"""
Two-Tower MovieRecommender model v3 (4-pool user tower, full softmax, ReLU, L2 norm).
Serving: user concat dim = 196 (4×32 + 32 + 32 + 4).

User tower:
  4×sum_pool(item_id_emb, 32) + genome_ctx(32) + genre(32) + ts(4) = 196
  → proj MLP → L2-normalize → output_dim

  Pools (all raw item_embedding_lookup, each with its own LayerNorm):
    full     — unweighted sum, all history
    liked    — unweighted sum, items with positive debiased rating
    disliked — unweighted sum, items with negative debiased rating
    weighted — rating-weighted sum, all history

Item tower:
  genre(8) + tag(16) + genome(32) + id(32) + year(8) = 96 → proj MLP → L2-normalize → output_dim

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
                 genome_context_buffer,
                 item_genre_embedding_size=8,
                 item_tag_embedding_size=16,
                 item_genome_tag_embedding_size=32,
                 item_movieId_embedding_size=32,
                 item_year_embedding_size=8,
                 user_genre_embedding_size=32,
                 timestamp_feature_embedding_size=4,
                 user_genome_context_embedding_size=32,
                 genre_context_buffer=None,
                 tag_context_buffer=None,
                 year_context_buffer=None,
                 proj_hidden=256,
                 output_dim=128,
                ):
        """
        genome_context_buffer: float32 (top_movies_len+1, genome_tags_len) [persistent].
            Row i = genome scores for movie at corpus index i. Last row = zeros (padding).

        genre_context_buffer:  float32 (top_movies_len+1, genres_len)   [non-persistent]
        tag_context_buffer:    float32 (top_movies_len+1, tags_len)     [non-persistent]
        year_context_buffer:   int64   (top_movies_len+1,)              [non-persistent]
        """
        super().__init__()

        self.pad_idx = top_movies_len

        # Persistent: genome scores are large and dataset-specific; saved in checkpoint.
        self.register_buffer('genome_context_buffer', genome_context_buffer)
        # Non-persistent: rebuilt from FeatureStore on every load (saves checkpoint space).
        self.register_buffer('genre_context_buffer', genre_context_buffer, persistent=False)
        self.register_buffer('tag_context_buffer',   tag_context_buffer,   persistent=False)
        self.register_buffer('year_context_buffer',  year_context_buffer,  persistent=False)

        # ── Shared item embedding (item tower + user history pool) ────────────
        self.item_embedding_lookup = nn.Embedding(
            top_movies_len + 1, item_movieId_embedding_size, padding_idx=top_movies_len
        )
        self.item_embedding_tower = nn.Sequential(
            nn.Linear(item_movieId_embedding_size, item_movieId_embedding_size),
            nn.ReLU(),
        )

        # ── Item feature towers ───────────────────────────────────────────────
        self.item_genre_tower = nn.Sequential(
            nn.Linear(genres_len, item_genre_embedding_size),
            nn.ReLU(),
        )
        self.item_tag_tower = nn.Sequential(
            nn.Linear(tags_len, item_tag_embedding_size),
            nn.ReLU(),
        )
        # Shared: used for item-side genome embedding AND user genome pool.
        self.item_genome_tag_tower = nn.Sequential(
            nn.Linear(genome_tags_len, item_genome_tag_embedding_size),
            nn.ReLU(),
        )
        self.year_embedding_lookup = nn.Embedding(all_years_len, item_year_embedding_size)
        self.year_embedding_tower = nn.Sequential(
            nn.Linear(item_year_embedding_size, item_year_embedding_size),
            nn.ReLU(),
        )

        # ── User towers ───────────────────────────────────────────────────────
        self.user_genre_tower = nn.Sequential(
            nn.Linear(user_context_size, user_genre_embedding_size),
            nn.ReLU(),
        )
        self.timestamp_embedding_lookup = nn.Embedding(
            timestamp_num_bins, timestamp_feature_embedding_size
        )
        self.timestamp_embedding_tower = nn.Sequential(
            nn.Linear(timestamp_feature_embedding_size, timestamp_feature_embedding_size),
            nn.ReLU(),
        )
        # Separate tower: rating-weighted avg of raw genome scores (1128-dim) → compact vector.
        # Distinct from item_genome_tag_tower which processes per-movie genome scores.
        self.user_genome_context_tower = nn.Sequential(
            nn.Linear(genome_tags_len, user_genome_context_embedding_size),
            nn.ReLU(),
        )

        # ── Quadruple history sum pools with LayerNorm ────────────────────────
        self.hist_full_norm     = nn.LayerNorm(item_movieId_embedding_size)
        self.hist_liked_norm    = nn.LayerNorm(item_movieId_embedding_size)
        self.hist_disliked_norm = nn.LayerNorm(item_movieId_embedding_size)
        self.hist_weighted_norm = nn.LayerNorm(item_movieId_embedding_size)

        # ── Projection MLPs ───────────────────────────────────────────────────
        user_concat_dim = (4 * item_movieId_embedding_size
                           + user_genome_context_embedding_size
                           + user_genre_embedding_size + timestamp_feature_embedding_size)
        item_concat_dim = (item_genre_embedding_size + item_tag_embedding_size
                           + item_genome_tag_embedding_size
                           + item_movieId_embedding_size + item_year_embedding_size)
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

    def user_embedding(self, user_genre_contexts, user_watch_history,
                       user_hist_liked, user_hist_disliked,
                       user_watch_history_ratings, timestamps):
        """User tower: returns (batch, output_dim), L2-normalized."""
        # 4 sum pools over raw item ID embeddings
        # Look up full history once; reuse for both full and weighted pools
        full_embs     = self.item_embedding_lookup(user_watch_history)
        pool_full     = self.hist_full_norm(full_embs.sum(dim=1))
        pool_weighted = self.hist_weighted_norm(
            (full_embs * user_watch_history_ratings.unsqueeze(-1)).sum(dim=1))
        pool_liked    = self._sum_pool(user_hist_liked,    self.hist_liked_norm)
        pool_disliked = self._sum_pool(user_hist_disliked, self.hist_disliked_norm)

        # Rating-weighted avg of raw genome scores (1128-dim) → genome context tower
        pad_mask       = (user_watch_history != self.pad_idx).float().unsqueeze(-1)
        rating_weights = user_watch_history_ratings.unsqueeze(-1) * pad_mask
        weight_sum     = rating_weights.abs().sum(dim=1).clamp(min=1e-6)
        watched_genome = self.genome_context_buffer[user_watch_history]
        genome_ctx_raw = (watched_genome * rating_weights).sum(dim=1) / weight_sum
        genome_ctx_emb = self.user_genome_context_tower(genome_ctx_raw)

        genre_emb = self.user_genre_tower(user_genre_contexts)
        ts_emb    = self.timestamp_embedding_tower(self.timestamp_embedding_lookup(timestamps))

        concat = torch.cat([pool_full, pool_liked, pool_disliked, pool_weighted,
                            genome_ctx_emb, genre_emb, ts_emb], dim=1)
        return F.normalize(self.user_projection(concat), p=2, dim=1)

    def item_embedding(self, target_movieId):
        """Item tower: looks up all features from buffers. Returns (batch, output_dim), L2-normalized."""
        item_genre_emb  = self.item_genre_tower(self.genre_context_buffer[target_movieId])
        item_tag_emb    = self.item_tag_tower(self.tag_context_buffer[target_movieId])
        item_genome_emb = self.item_genome_tag_tower(self.genome_context_buffer[target_movieId])
        item_emb        = self.item_embedding_tower(self.item_embedding_lookup(target_movieId))
        year_emb        = self.year_embedding_tower(
                              self.year_embedding_lookup(self.year_context_buffer[target_movieId]))
        concat = torch.cat([item_genre_emb, item_tag_emb, item_genome_emb, item_emb, year_emb], dim=1)
        return F.normalize(self.item_projection(concat), p=2, dim=1)

    def full_item_embedding(self):
        """Returns all corpus item embeddings (n_movies, output_dim) in one batched pass.
        Used by full softmax training to build the (batch × n_movies) score matrix."""
        all_idxs = torch.arange(self.pad_idx, device=self.genome_context_buffer.device)
        return self.item_embedding(all_idxs)

    def forward(self, user_genre_contexts, user_watch_history,
                user_hist_liked, user_hist_disliked,
                user_watch_history_ratings, timestamps, target_movieId):
        user_emb = self.user_embedding(user_genre_contexts, user_watch_history,
                                       user_hist_liked, user_hist_disliked,
                                       user_watch_history_ratings, timestamps)
        item_emb = self.item_embedding(target_movieId)
        return torch.einsum('ij,ij->i', user_emb, item_emb)
