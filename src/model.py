"""
Two-Tower MovieRecommender model v3 (4-pool user tower, full softmax, ReLU, L2 norm).

User tower:
  4×sum_pool(item_id_emb, 32) + [genome_ctx(32)] + [llm_ctx(32)] + [genre(32)] + [ts(4)]
  → proj MLP → L2-normalize → output_dim

  Pools (all raw item_embedding_lookup, each with its own LayerNorm; base_towers='idonly'
  keeps only `full`):
    full     — unweighted sum, all history
    liked    — unweighted sum, items with positive debiased rating
    disliked — unweighted sum, items with negative debiased rating
    weighted — rating-weighted sum, all history

Item tower:
  [genre(8)] + [tag(16)] + [genome(32)] + [llm(32)] + id(32) + [year(8)] → proj MLP → L2-norm → output_dim

Semantic-feature towers are set by feature_towers — genome tags and LLM features are each their
own named sub-tower (like genre/tag/year), independently switched on or off:
  'genome' — genome-tag tower only.                         [Model A / prod-v3]
  'llm'    — LLM-feature tower only.                         [Model B]
  None     — neither (towers/buffers omitted, concat shrinks). [Model C floor]
  'both'   — genome-tag AND LLM-feature towers, both         [Model D / new prod]
             concatenated in. Each has its own item-side and user-side sub-tower
             (item_genome_tag_tower/user_genome_context_tower and
             item_llm_feature_tower/user_llm_feature_tower) over its own buffer.

Base towers (genre, tag, year — historically always-on) are set by base_towers:
  'all'    — genre + tag + year towers included.            [every model above]
  'idonly' — stripped CF-base ablation: item/user genre, item tag, year, and timestamp
             towers (and their buffers) omitted, and the liked/disliked/rating-weighted
             pools dropped — concat shrinks. The user tower is ONLY the single full-history
             sum pool plus the semantic-feature context when that slot is on; the item
             tower is ONLY item_embedding_tower plus the semantic-feature slot — isolating
             what the content slot adds over pure collaborative filtering.

Dot product of L2-normalized vectors = cosine similarity.

item_embedding() looks up all features from registered buffers — call full_item_embedding()
to score all corpus items in one batched pass (required by full softmax training).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MovieRecommender(nn.Module):
    def __init__(self,
                 genres_len,
                 tags_len,
                 genome_tags_len,
                 top_movies_len,
                 all_years_len,
                 timestamp_num_bins,
                 user_context_size,
                 feature_towers='genome',
                 base_towers='all',
                 genome_context_buffer=None,
                 llm_feature_buffer=None,
                 llm_feature_len=None,
                 item_genre_embedding_size=8,
                 item_tag_embedding_size=16,
                 item_genome_embedding_size=32,
                 item_llm_embedding_size=32,
                 item_movieId_embedding_size=32,
                 item_year_embedding_size=8,
                 user_genre_embedding_size=32,
                 timestamp_feature_embedding_size=4,
                 user_genome_embedding_size=32,
                 user_llm_embedding_size=32,
                 genre_context_buffer=None,
                 tag_context_buffer=None,
                 year_context_buffer=None,
                 proj_hidden=256,
                 output_dim=128,
                ):
        """
        feature_towers: 'genome' | 'llm' | 'both' | None. Which semantic-feature sub-towers the
            model includes. genome and llm are independent; 'both' (Model D) builds both, None
            (Model C) builds neither and the concat dims shrink accordingly.

        base_towers: 'all' | 'idonly'. 'all' (default) keeps the historically always-on
            genre/tag/year/timestamp towers; 'idonly' (stripped CF-base ablation) omits
            them — towers and buffers — collapses the user history pools to the single
            full-history sum pool, and the concat dims shrink accordingly.

        genome_context_buffer: float32 (top_movies_len+1, genome_tags_len) [non-persistent].
            Row i = genome scores for movie at corpus index i. Last row = zeros (padding).
            Present for 'genome'/'both'.
        llm_feature_buffer:    float32 (top_movies_len+1, llm_feature_len) [non-persistent].
            Row i = LLM features for movie at corpus index i. Present for 'llm'/'both'.
        genome_tags_len / llm_feature_len: in_features of the genome / LLM sub-towers (1128 / 132).

        genre_context_buffer:  float32 (top_movies_len+1, genres_len)   [non-persistent]
        tag_context_buffer:    float32 (top_movies_len+1, tags_len)     [non-persistent]
        year_context_buffer:   int64   (top_movies_len+1,)              [non-persistent]
        """
        super().__init__()

        self.pad_idx = top_movies_len
        self.feature_towers = feature_towers
        # genome and llm sub-towers are independent; 'both' enables both, None neither.
        self.has_genome = feature_towers in ('genome', 'both')
        self.has_llm    = feature_towers in ('llm', 'both')
        # 'idonly' strips the always-on genre/tag/year/timestamp towers as a block (CF-base
        # ablation), and collapses the user tower to the single full-history pool — the
        # liked/disliked/rating-weighted pools are rating-derived signals, stripped with it.
        self.base_towers = base_towers
        self.has_genre = self.has_tag = self.has_year = (base_towers != 'idonly')
        self.has_timestamp    = (base_towers != 'idonly')
        self.has_rating_pools = (base_towers != 'idonly')

        # Non-persistent: rebuilt from FeatureStore on every load (saves checkpoint space).
        if self.has_genome:
            self.register_buffer('genome_context_buffer', genome_context_buffer, persistent=False)
        if self.has_llm:
            self.register_buffer('llm_feature_buffer', llm_feature_buffer, persistent=False)
        if self.has_genre:
            self.register_buffer('genre_context_buffer',  genre_context_buffer,  persistent=False)
        if self.has_tag:
            self.register_buffer('tag_context_buffer',    tag_context_buffer,    persistent=False)
        if self.has_year:
            self.register_buffer('year_context_buffer',   year_context_buffer,   persistent=False)

        # ── Shared item embedding (item tower + user history pool) ────────────
        self.item_embedding_lookup = nn.Embedding(
            top_movies_len + 1, item_movieId_embedding_size, padding_idx=top_movies_len
        )
        self.item_embedding_tower = nn.Sequential(
            nn.Linear(item_movieId_embedding_size, item_movieId_embedding_size),
            nn.ReLU(),
        )

        # ── Item feature towers ───────────────────────────────────────────────
        if self.has_genre:
            self.item_genre_tower = nn.Sequential(
                nn.Linear(genres_len, item_genre_embedding_size),
                nn.ReLU(),
            )
        if self.has_tag:
            self.item_tag_tower = nn.Sequential(
                nn.Linear(tags_len, item_tag_embedding_size),
                nn.ReLU(),
            )
        # Per-movie semantic-feature towers (genome scores 1128-dim, LLM features 132-dim).
        if self.has_genome:
            self.item_genome_tag_tower = nn.Sequential(
                nn.Linear(genome_tags_len, item_genome_embedding_size),
                nn.ReLU(),
            )
        if self.has_llm:
            self.item_llm_feature_tower = nn.Sequential(
                nn.Linear(llm_feature_len, item_llm_embedding_size),
                nn.ReLU(),
            )
        if self.has_year:
            self.year_embedding_lookup = nn.Embedding(all_years_len, item_year_embedding_size)
            self.year_embedding_tower = nn.Sequential(
                nn.Linear(item_year_embedding_size, item_year_embedding_size),
                nn.ReLU(),
            )

        # ── User towers ───────────────────────────────────────────────────────
        if self.has_genre:
            self.user_genre_tower = nn.Sequential(
                nn.Linear(user_context_size, user_genre_embedding_size),
                nn.ReLU(),
            )
        if self.has_timestamp:
            self.timestamp_embedding_lookup = nn.Embedding(
                timestamp_num_bins, timestamp_feature_embedding_size
            )
            self.timestamp_embedding_tower = nn.Sequential(
                nn.Linear(timestamp_feature_embedding_size, timestamp_feature_embedding_size),
                nn.ReLU(),
            )
        # User-side semantic-feature towers: run over the rating-weighted avg of the raw per-movie
        # feature across history (a compact taste fingerprint). Distinct from the item-side towers,
        # which process a single movie's scores.
        if self.has_genome:
            self.user_genome_context_tower = nn.Sequential(
                nn.Linear(genome_tags_len, user_genome_embedding_size),
                nn.ReLU(),
            )
        if self.has_llm:
            self.user_llm_feature_tower = nn.Sequential(
                nn.Linear(llm_feature_len, user_llm_embedding_size),
                nn.ReLU(),
            )

        # ── Quadruple history sum pools with LayerNorm ────────────────────────
        # ('idonly' keeps only the full pool.)
        self.hist_full_norm     = nn.LayerNorm(item_movieId_embedding_size)
        if self.has_rating_pools:
            self.hist_liked_norm    = nn.LayerNorm(item_movieId_embedding_size)
            self.hist_disliked_norm = nn.LayerNorm(item_movieId_embedding_size)
            self.hist_weighted_norm = nn.LayerNorm(item_movieId_embedding_size)

        # ── Projection MLPs ───────────────────────────────────────────────────
        # Each gated tower (semantic-feature or base) contributes 0 to its concat when off.
        genome_user_dim = user_genome_embedding_size if self.has_genome else 0
        llm_user_dim    = user_llm_embedding_size    if self.has_llm    else 0
        genome_item_dim = item_genome_embedding_size if self.has_genome else 0
        llm_item_dim    = item_llm_embedding_size    if self.has_llm    else 0
        genre_user_dim  = user_genre_embedding_size  if self.has_genre  else 0
        genre_item_dim  = item_genre_embedding_size  if self.has_genre  else 0
        tag_item_dim    = item_tag_embedding_size    if self.has_tag    else 0
        year_item_dim   = item_year_embedding_size   if self.has_year   else 0
        ts_user_dim     = timestamp_feature_embedding_size if self.has_timestamp else 0
        n_pools         = 4 if self.has_rating_pools else 1
        user_concat_dim = (n_pools * item_movieId_embedding_size
                           + genome_user_dim + llm_user_dim
                           + genre_user_dim + ts_user_dim)
        item_concat_dim = (genre_item_dim + tag_item_dim
                           + genome_item_dim + llm_item_dim
                           + item_movieId_embedding_size + year_item_dim)
        # No activation on the final linear — output is L2-normalized.
        self.user_projection = nn.Sequential(
            nn.Linear(user_concat_dim, proj_hidden),
            nn.ReLU(),
            nn.Linear(proj_hidden, output_dim),
        )
        self.item_projection = nn.Sequential(
            nn.Linear(item_concat_dim, proj_hidden),
            nn.ReLU(),
            nn.Linear(proj_hidden, output_dim),
        )

        # Sub-tower linears: gain=0.1 keeps initial sub-embeddings small.
        self.apply(self._init_weights)
        # Projection layers re-initialized at gain=1.0. Without this, gain=0.1^2 compounds
        # across sub-tower + projection and collapses dot products to near-zero at step 0.
        for proj in [self.user_projection, self.item_projection]:
            for m in proj.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.1)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight, gain=0.01)

    def _sum_pool(self, ids, norm_layer, weights=None):
        """Sum pool over raw item_embedding_lookup with optional per-item weights."""
        embs = self.item_embedding_lookup(ids)
        if weights is not None:
            embs = embs * weights.unsqueeze(-1)
        return norm_layer(embs.sum(dim=1))

    def _watched_context(self, buffer, history, ratings):
        """Rating-weighted average of a per-movie raw feature buffer over the user's history.

        Shared by the genome and LLM user-side towers — feed it the matching buffer."""
        pad_mask       = (history != self.pad_idx).float().unsqueeze(-1)
        rating_weights = ratings.unsqueeze(-1) * pad_mask
        weight_sum     = rating_weights.abs().sum(dim=1).clamp(min=1e-6)
        watched        = buffer[history]
        return (watched * rating_weights).sum(dim=1) / weight_sum

    def user_embedding(self, user_genre_contexts, user_watch_history,
                       user_hist_liked, user_hist_disliked,
                       user_watch_history_ratings, timestamps):
        """User tower: returns (batch, output_dim), L2-normalized."""
        # Sum pools over raw item ID embeddings ('idonly' keeps only the full pool)
        # Look up full history once; reuse for both full and weighted pools
        full_embs = self.item_embedding_lookup(user_watch_history)
        pool_full = self.hist_full_norm(full_embs.sum(dim=1))

        parts = [pool_full]
        if self.has_rating_pools:
            pool_liked    = self._sum_pool(user_hist_liked,    self.hist_liked_norm)
            pool_disliked = self._sum_pool(user_hist_disliked, self.hist_disliked_norm)
            pool_weighted = self.hist_weighted_norm(
                (full_embs * user_watch_history_ratings.unsqueeze(-1)).sum(dim=1))
            parts = [pool_full, pool_liked, pool_disliked, pool_weighted]

        # Rating-weighted avg of each per-movie raw feature → its user-side tower.
        if self.has_genome:
            genome_ctx = self._watched_context(self.genome_context_buffer,
                                               user_watch_history, user_watch_history_ratings)
            parts.append(self.user_genome_context_tower(genome_ctx))
        if self.has_llm:
            llm_ctx = self._watched_context(self.llm_feature_buffer,
                                            user_watch_history, user_watch_history_ratings)
            parts.append(self.user_llm_feature_tower(llm_ctx))

        if self.has_genre:
            parts.append(self.user_genre_tower(user_genre_contexts))
        if self.has_timestamp:
            parts.append(self.timestamp_embedding_tower(self.timestamp_embedding_lookup(timestamps)))

        concat = torch.cat(parts, dim=1)
        return F.normalize(self.user_projection(concat), p=2, dim=1)

    def item_embedding(self, target_movieId):
        """Item tower: looks up all features from buffers. Returns (batch, output_dim), L2-normalized."""
        item_emb = self.item_embedding_tower(self.item_embedding_lookup(target_movieId))

        parts = []
        if self.has_genre:
            parts.append(self.item_genre_tower(self.genre_context_buffer[target_movieId]))
        if self.has_tag:
            parts.append(self.item_tag_tower(self.tag_context_buffer[target_movieId]))
        if self.has_genome:
            parts.append(self.item_genome_tag_tower(self.genome_context_buffer[target_movieId]))
        if self.has_llm:
            parts.append(self.item_llm_feature_tower(self.llm_feature_buffer[target_movieId]))
        parts.append(item_emb)
        if self.has_year:
            parts.append(self.year_embedding_tower(
                             self.year_embedding_lookup(self.year_context_buffer[target_movieId])))

        concat = torch.cat(parts, dim=1)
        return F.normalize(self.item_projection(concat), p=2, dim=1)

    def full_item_embedding(self):
        """Returns all corpus item embeddings (n_movies, output_dim) in one batched pass.
        Used by full softmax training to build the (batch × n_movies) score matrix."""
        all_idxs = torch.arange(self.pad_idx, device=self.item_embedding_lookup.weight.device)
        return self.item_embedding(all_idxs)

    def forward(self, user_genre_contexts, user_watch_history,
                user_hist_liked, user_hist_disliked,
                user_watch_history_ratings, timestamps, target_movieId):
        user_emb = self.user_embedding(user_genre_contexts, user_watch_history,
                                       user_hist_liked, user_hist_disliked,
                                       user_watch_history_ratings, timestamps)
        item_emb = self.item_embedding(target_movieId)
        return torch.einsum('ij,ij->i', user_emb, item_emb)
