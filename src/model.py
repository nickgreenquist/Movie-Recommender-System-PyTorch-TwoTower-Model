"""
Two-Tower MovieRecommender model v3 (4-pool user tower, full softmax, ReLU, L2 norm).

User tower:
  4×sum_pool(item_id_emb, 32) + [genome_ctx(32)] + [llm_ctx(32)] + [genre(32)] + [ts(4)]
  → proj MLP → L2-normalize → output_dim

  Pools (all raw item_embedding_lookup, each with its own LayerNorm; base_towers='idonly'
  keeps only `full`):
    full         — unweighted sum, all history
    liked        — unweighted sum, items with positive debiased rating
    disliked     — unweighted sum, items with negative debiased rating
    weighted     — rating-weighted sum, all history
    last_liked   — single most-recent item with positive debiased rating (recency × valence)
    last_watched — single most-recent item, any rating (recency only; valence-free sibling)
    second_to_last_watched — the item one position before the most-recent watch (probes recency decay past the last item)

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

feature_towers/base_towers are the COARSE selectors that set defaults. For per-tower ablations,
three fine-grained knobs override them independently (each a subset of its canonical-order tuple,
None → the legacy set derived from feature_towers + base_towers):
  user_pools    ⊆ POOL_ORDER         — which history pools the user tower sums over
                                        (full/liked/disliked/weighted + `last`, the most-recent item).
  user_features ⊆ USER_FEATURE_ORDER — which non-pool user-side context towers run (genre/genome/llm/timestamp).
  item_features ⊆ ITEM_FEATURE_ORDER — which item-side feature towers (besides the always-on ID) run.
The order in each tuple is load-bearing (it fixes the concat column layout); src/checkpoint.py
resolves all three from the saved weight keys, so a fine-grained checkpoint rebuilds from weights alone.

Dot product of L2-normalized vectors = cosine similarity.

item_embedding() looks up all features from registered buffers — call full_item_embedding()
to score all corpus items in one batched pass (required by full softmax training).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Fine-grained tower/pool selection ─────────────────────────────────────────
# Three independent knobs select which sub-towers the model builds, decoupled from
# the coarse feature_towers / base_towers selectors (which only set the DEFAULTS):
#   user_pools    — which history pools the user tower sums over.
#   user_features — which NON-pool user-side context towers run.
#   item_features — which item-side feature towers (besides the always-on ID embedding) run.
# Each is a subset of its canonical-order tuple. The order is load-bearing: it fixes the
# concat column layout, so the resolver (src/checkpoint.py) and the forward pass must agree.
POOL_ORDER         = ('full', 'liked', 'disliked', 'weighted', 'last_liked', 'last_watched', 'second_to_last_watched')
USER_FEATURE_ORDER = ('genre', 'genome', 'llm', 'timestamp')
ITEM_FEATURE_ORDER = ('genre', 'tag', 'genome', 'llm', 'year')


def default_user_pools(base_towers):
    """Legacy pool set implied by base_towers (used when user_pools is unspecified)."""
    return ('full',) if base_towers == 'idonly' else ('full', 'liked', 'disliked', 'weighted')


def default_user_features(feature_towers, base_towers):
    """Legacy user-side non-pool context towers implied by feature_towers + base_towers."""
    feats = set()
    if feature_towers in ('genome', 'both'):
        feats.add('genome')
    if feature_towers in ('llm', 'both'):
        feats.add('llm')
    if base_towers != 'idonly':
        feats.update(('genre', 'timestamp'))
    return tuple(f for f in USER_FEATURE_ORDER if f in feats)


def default_item_features(feature_towers, base_towers):
    """Legacy item-side feature towers (besides ID) implied by feature_towers + base_towers."""
    feats = set()
    if feature_towers in ('genome', 'both'):
        feats.add('genome')
    if feature_towers in ('llm', 'both'):
        feats.add('llm')
    if base_towers != 'idonly':
        feats.update(('genre', 'tag', 'year'))
    return tuple(f for f in ITEM_FEATURE_ORDER if f in feats)


def _canon(values, order, kind):
    """Validate `values` against `order` and return them as a canonical-order tuple."""
    s   = set(values)
    bad = s - set(order)
    if bad:
        raise ValueError(f"Unknown {kind}: {sorted(bad)}; allowed: {list(order)}")
    return tuple(x for x in order if x in s)


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
                 user_pools=None,
                 user_features=None,
                 item_features=None,
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

        user_pools / user_features / item_features: fine-grained overrides that INDEPENDENTLY
            select which history pools, user-side context towers, and item-side feature towers
            are active (subsets of POOL_ORDER / USER_FEATURE_ORDER / ITEM_FEATURE_ORDER). Each
            defaults to None, in which case the legacy set is derived from feature_towers +
            base_towers (default_user_pools / default_user_features / default_item_features) so
            existing checkpoints, the serving path, and every old config build the identical
            model. When set, they override feature_towers/base_towers for that side — e.g.
            user_pools=('full','liked') is a 2-pool user tower; item_features=() is an
            ID-embedding-only item tower; user_features=() drops every user-side context tower.
            The ID embedding (item) and a non-empty user_pools set are always required.

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
        self.base_towers    = base_towers

        # Resolve the three fine-grained selectors. None → derive the legacy set from
        # feature_towers + base_towers, so an unspecified knob reproduces today's model exactly
        # (existing checkpoints, the serving path, and old configs are all unaffected). When a
        # knob is given, it overrides that side independently of feature_towers/base_towers.
        self.user_pools = (_canon(user_pools, POOL_ORDER, 'user_pools')
                           if user_pools is not None else default_user_pools(base_towers))
        if not self.user_pools:
            raise ValueError("user_pools must select at least one history pool")
        self.user_features = (_canon(user_features, USER_FEATURE_ORDER, 'user_features')
                              if user_features is not None
                              else default_user_features(feature_towers, base_towers))
        self.item_features = (_canon(item_features, ITEM_FEATURE_ORDER, 'item_features')
                              if item_features is not None
                              else default_item_features(feature_towers, base_towers))

        # Per-side presence flags — the single source of truth for tower/buffer/concat decisions.
        # genome/llm/genre are now independent per side (item vs user); tag/year are item-only;
        # timestamp is user-only.
        self.has_user_genre  = 'genre'  in self.user_features
        self.has_user_genome = 'genome' in self.user_features
        self.has_user_llm    = 'llm'    in self.user_features
        self.has_timestamp   = 'timestamp' in self.user_features
        self.has_item_genre  = 'genre'  in self.item_features
        self.has_item_tag    = 'tag'    in self.item_features
        self.has_item_genome = 'genome' in self.item_features
        self.has_item_llm    = 'llm'    in self.item_features
        self.has_item_year   = 'year'   in self.item_features

        # Derived "either-side" aliases — kept for external consumers that only ask whether a
        # feature is present anywhere (src/export.py, tools/). Context buffers are shared across
        # sides, so these also gate buffer registration.
        self.has_genome = self.has_item_genome or self.has_user_genome
        self.has_llm    = self.has_item_llm    or self.has_user_llm
        self.has_genre  = self.has_item_genre  or self.has_user_genre
        self.has_tag    = self.has_item_tag
        self.has_year   = self.has_item_year
        self.has_rating_pools = any(p in self.user_pools for p in ('liked', 'disliked', 'weighted'))

        # Non-persistent: rebuilt from FeatureStore on every load (saves checkpoint space). The
        # genome/llm context buffers are shared by the item AND user towers (register if either
        # side uses them); genre/tag/year buffers feed only the item towers (the user_genre_tower
        # reads the X_genre input vector, not a buffer).
        if self.has_genome:
            self.register_buffer('genome_context_buffer', genome_context_buffer, persistent=False)
        if self.has_llm:
            self.register_buffer('llm_feature_buffer', llm_feature_buffer, persistent=False)
        if self.has_item_genre:
            self.register_buffer('genre_context_buffer',  genre_context_buffer,  persistent=False)
        if self.has_item_tag:
            self.register_buffer('tag_context_buffer',    tag_context_buffer,    persistent=False)
        if self.has_item_year:
            self.register_buffer('year_context_buffer',   year_context_buffer,   persistent=False)

        # ── Shared item embedding (item tower + user history pool) ────────────
        self.item_embedding_lookup = nn.Embedding(
            top_movies_len + 1, item_movieId_embedding_size, padding_idx=top_movies_len
        )
        self.item_embedding_tower = nn.Sequential(
            nn.Linear(item_movieId_embedding_size, item_movieId_embedding_size),
            nn.ReLU(),
        )

        # ── Item feature towers (each gated by its item_features membership) ──
        if self.has_item_genre:
            self.item_genre_tower = nn.Sequential(
                nn.Linear(genres_len, item_genre_embedding_size),
                nn.ReLU(),
            )
        if self.has_item_tag:
            self.item_tag_tower = nn.Sequential(
                nn.Linear(tags_len, item_tag_embedding_size),
                nn.ReLU(),
            )
        # Per-movie semantic-feature towers (genome scores 1128-dim, LLM features 132-dim).
        if self.has_item_genome:
            self.item_genome_tag_tower = nn.Sequential(
                nn.Linear(genome_tags_len, item_genome_embedding_size),
                nn.ReLU(),
            )
        if self.has_item_llm:
            self.item_llm_feature_tower = nn.Sequential(
                nn.Linear(llm_feature_len, item_llm_embedding_size),
                nn.ReLU(),
            )
        if self.has_item_year:
            self.year_embedding_lookup = nn.Embedding(all_years_len, item_year_embedding_size)
            self.year_embedding_tower = nn.Sequential(
                nn.Linear(item_year_embedding_size, item_year_embedding_size),
                nn.ReLU(),
            )

        # ── User towers (each gated by its user_features membership) ──────────
        if self.has_user_genre:
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
        if self.has_user_genome:
            self.user_genome_context_tower = nn.Sequential(
                nn.Linear(genome_tags_len, user_genome_embedding_size),
                nn.ReLU(),
            )
        if self.has_user_llm:
            self.user_llm_feature_tower = nn.Sequential(
                nn.Linear(llm_feature_len, user_llm_embedding_size),
                nn.ReLU(),
            )

        # ── History sum pools with LayerNorm (one per active user_pool) ───────
        # full/liked/disliked/weighted sum over the raw item-ID embedding; `last_liked` is the single
        # most-recent *liked* item's embedding (a recency×valence channel). Each pool is gated by
        # user_pools, so a subset (e.g. weighted-only, or 4-pool + last_liked) builds exactly its norms.
        if 'full' in self.user_pools:
            self.hist_full_norm         = nn.LayerNorm(item_movieId_embedding_size)
        if 'liked' in self.user_pools:
            self.hist_liked_norm        = nn.LayerNorm(item_movieId_embedding_size)
        if 'disliked' in self.user_pools:
            self.hist_disliked_norm     = nn.LayerNorm(item_movieId_embedding_size)
        if 'weighted' in self.user_pools:
            self.hist_weighted_norm     = nn.LayerNorm(item_movieId_embedding_size)
        if 'last_liked' in self.user_pools:
            self.hist_last_liked_norm   = nn.LayerNorm(item_movieId_embedding_size)
        if 'last_watched' in self.user_pools:
            self.hist_last_watched_norm = nn.LayerNorm(item_movieId_embedding_size)
        if 'second_to_last_watched' in self.user_pools:
            self.hist_second_to_last_watched_norm = nn.LayerNorm(item_movieId_embedding_size)

        # ── Projection MLPs ───────────────────────────────────────────────────
        # Each gated tower/pool contributes 0 to its concat when off. Dims are split per side so
        # genome/llm/genre can be on for the item tower but off for the user tower (or vice-versa).
        genome_user_dim = user_genome_embedding_size if self.has_user_genome else 0
        llm_user_dim    = user_llm_embedding_size    if self.has_user_llm    else 0
        genome_item_dim = item_genome_embedding_size if self.has_item_genome else 0
        llm_item_dim    = item_llm_embedding_size    if self.has_item_llm    else 0
        genre_user_dim  = user_genre_embedding_size  if self.has_user_genre  else 0
        genre_item_dim  = item_genre_embedding_size  if self.has_item_genre  else 0
        tag_item_dim    = item_tag_embedding_size    if self.has_item_tag    else 0
        year_item_dim   = item_year_embedding_size   if self.has_item_year   else 0
        ts_user_dim     = timestamp_feature_embedding_size if self.has_timestamp else 0
        n_pools         = len(self.user_pools)
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

    def _last_liked_ids(self, history, ratings):
        """Index of each row's most-recent *liked* item (rightmost position with debiased rating > 0).

        Alignment-independent: the rollback eval pads histories LEFT-aligned (items first) while
        the training dataset pads them RIGHT-aligned (items last); both order history oldest→newest,
        so the rightmost liked column is the most-recent liked item in either layout. `ratings > 0`
        is the same liked-set definition used by the `liked` pool and `X_hist_liked` (pad positions
        carry rating 0.0, so they're excluded). A row with no liked item (or an all-pad row) resolves
        to a pad index (→ zero embedding → the LayerNorm bias β), the same benign null as an empty pool."""
        liked_mask = (ratings > 0)
        length     = history.shape[1]
        has_liked  = liked_mask.any(dim=1)
        # argmax on the column-reversed mask = first liked from the right = last liked position.
        last_pos   = length - 1 - torch.argmax(liked_mask.flip(dims=[1]).int(), dim=1)
        ids        = history.gather(1, last_pos.unsqueeze(1)).squeeze(1)
        # No-likes rows: argmax of an all-False mask is 0 (a meaningless column) — force the pad index.
        return torch.where(has_liked, ids, torch.full_like(ids, self.pad_idx))

    def _last_watched_ids(self, history):
        """Index of each row's most-recent watched item (rightmost non-pad position, ANY rating).

        The valence-free sibling of `_last_liked_ids`: identical rightmost-gather, but the mask is
        `history != pad_idx` instead of `ratings > 0`, so it returns the most-recent watch even when
        it was a low-rated / hate-watch. Alignment-independent for the same reason (both the train
        right-aligned and eval left-aligned layouts order history oldest→newest, so the rightmost
        real column is the most recent). An all-pad row resolves to a pad index (→ zero embedding →
        the LayerNorm bias β), the same benign null as an empty pool."""
        watched_mask = (history != self.pad_idx)
        length       = history.shape[1]
        has_watched  = watched_mask.any(dim=1)
        # argmax on the column-reversed mask = first real item from the right = last watched position.
        last_pos     = length - 1 - torch.argmax(watched_mask.flip(dims=[1]).int(), dim=1)
        ids          = history.gather(1, last_pos.unsqueeze(1)).squeeze(1)
        # All-pad rows: argmax of an all-False mask is 0 (a meaningless column) — force the pad index.
        return torch.where(has_watched, ids, torch.full_like(ids, self.pad_idx))

    def _second_to_last_watched_ids(self, history):
        """Index of each row's SECOND-most-recent watched item (the non-pad position one column left
        of the most recent). Because real items are always a contiguous block (training right-aligns,
        eval left-aligns — neither leaves interior pads), the second-to-last watch is simply one
        position left of the last; rows with fewer than two watched items resolve to a pad index (→
        zero embedding → the LayerNorm bias β). Alignment-independent for the same reason as
        `_last_watched_ids`."""
        watched_mask = (history != self.pad_idx)
        length       = history.shape[1]
        n_watched    = watched_mask.sum(dim=1)
        has_two      = n_watched >= 2
        last_pos     = length - 1 - torch.argmax(watched_mask.flip(dims=[1]).int(), dim=1)
        second_pos   = (last_pos - 1).clamp(min=0)
        ids          = history.gather(1, second_pos.unsqueeze(1)).squeeze(1)
        return torch.where(has_two, ids, torch.full_like(ids, self.pad_idx))

    def user_embedding(self, user_genre_contexts, user_watch_history,
                       user_hist_liked, user_hist_disliked,
                       user_watch_history_ratings, timestamps):
        """User tower: returns (batch, output_dim), L2-normalized."""
        # History sum pools over raw item-ID embeddings, appended in canonical POOL_ORDER so the
        # concat column layout matches user_concat_dim. Look up the full history once (reused by
        # both the full and weighted pools); only compute it when one of those pools is active.
        full_embs = (self.item_embedding_lookup(user_watch_history)
                     if ('full' in self.user_pools or 'weighted' in self.user_pools) else None)

        parts = []
        if 'full' in self.user_pools:
            parts.append(self.hist_full_norm(full_embs.sum(dim=1)))
        if 'liked' in self.user_pools:
            parts.append(self._sum_pool(user_hist_liked, self.hist_liked_norm))
        if 'disliked' in self.user_pools:
            parts.append(self._sum_pool(user_hist_disliked, self.hist_disliked_norm))
        if 'weighted' in self.user_pools:
            parts.append(self.hist_weighted_norm(
                (full_embs * user_watch_history_ratings.unsqueeze(-1)).sum(dim=1)))
        if 'last_liked' in self.user_pools:
            last_ids = self._last_liked_ids(user_watch_history, user_watch_history_ratings)
            parts.append(self.hist_last_liked_norm(self.item_embedding_lookup(last_ids)))
        if 'last_watched' in self.user_pools:
            last_watched_ids = self._last_watched_ids(user_watch_history)
            parts.append(self.hist_last_watched_norm(self.item_embedding_lookup(last_watched_ids)))
        if 'second_to_last_watched' in self.user_pools:
            second_to_last_ids = self._second_to_last_watched_ids(user_watch_history)
            parts.append(self.hist_second_to_last_watched_norm(self.item_embedding_lookup(second_to_last_ids)))

        # Rating-weighted avg of each per-movie raw feature → its user-side tower.
        if self.has_user_genome:
            genome_ctx = self._watched_context(self.genome_context_buffer,
                                               user_watch_history, user_watch_history_ratings)
            parts.append(self.user_genome_context_tower(genome_ctx))
        if self.has_user_llm:
            llm_ctx = self._watched_context(self.llm_feature_buffer,
                                            user_watch_history, user_watch_history_ratings)
            parts.append(self.user_llm_feature_tower(llm_ctx))

        if self.has_user_genre:
            parts.append(self.user_genre_tower(user_genre_contexts))
        if self.has_timestamp:
            parts.append(self.timestamp_embedding_tower(self.timestamp_embedding_lookup(timestamps)))

        concat = torch.cat(parts, dim=1)
        return F.normalize(self.user_projection(concat), p=2, dim=1)

    def item_embedding(self, target_movieId):
        """Item tower: looks up all features from buffers. Returns (batch, output_dim), L2-normalized."""
        item_emb = self.item_embedding_tower(self.item_embedding_lookup(target_movieId))

        parts = []
        if self.has_item_genre:
            parts.append(self.item_genre_tower(self.genre_context_buffer[target_movieId]))
        if self.has_item_tag:
            parts.append(self.item_tag_tower(self.tag_context_buffer[target_movieId]))
        if self.has_item_genome:
            parts.append(self.item_genome_tag_tower(self.genome_context_buffer[target_movieId]))
        if self.has_item_llm:
            parts.append(self.item_llm_feature_tower(self.llm_feature_buffer[target_movieId]))
        parts.append(item_emb)
        if self.has_item_year:
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
