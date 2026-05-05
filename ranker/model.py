"""
Stage 2 — Wide & Deep ranker with full CG feature parity.

Mirrors the CG (src/model.py) feature set: same towers, same shared parameters
(item_id embedding shared between item-side tower and user history pool;
item_genome_tower shared between item side and user genome pool), same buffer
lookups for genre / tag / genome / year. All parameters are ranker-owned —
the buffers carry static data (genome scores, genre one-hot, etc.) but no CG
weights are loaded.

Key difference from CG:
  - No projection MLP. CG projects user/item to L2-normalized 128-dim vectors
    for cosine softmax. The ranker concatenates user_concat + item_concat
    directly into a deep MLP, then a head with cross-feature wide bypass.
  - No L2 normalization. We score with BCE + raw logits, not cosine.

User concat (132): history_pool(32) + genome_pool(32) + genome_ctx(32)
                   + genre_emb(32) + ts_emb(4)
Item concat  (96): item_genre_emb(8) + item_tag_emb(16) + item_genome_emb(32)
                   + item_id_emb(32) + year_emb(8)
Deep MLP    (228 → [256, 128, 64]) → 64
Head: cat(deep_out(64), cross_features(n_cross)) → Linear(64+n_cross, 1)

ZERO src/ imports — buffers built externally from FeatureStore and passed in.
"""
import torch
import torch.nn as nn


class WideDeepRanker(nn.Module):
    def __init__(self,
                 # Vocab / corpus sizes
                 n_movies: int,
                 n_genres: int,
                 n_tags: int,
                 n_genome_tags: int,
                 n_years: int,
                 n_ts_bins: int,
                 user_context_size: int,            # 2 * n_genres = 40
                 # Feature buffers (with padding row appended — index n_movies = pad)
                 genome_buffer: torch.Tensor,       # (n_movies+1, n_genome_tags) float32
                 genre_buffer: torch.Tensor,        # (n_movies+1, n_genres) float32
                 tag_buffer: torch.Tensor,          # (n_movies+1, n_tags) float32
                 year_buffer: torch.Tensor,         # (n_movies+1,) int64 — year index
                 # Sub-tower output dims (CG defaults)
                 item_id_emb_dim: int = 32,
                 item_genre_emb_dim: int = 8,
                 item_tag_emb_dim: int = 16,
                 item_genome_emb_dim: int = 32,
                 item_year_emb_dim: int = 8,
                 user_genre_emb_dim: int = 32,
                 user_genome_ctx_emb_dim: int = 32,
                 ts_emb_dim: int = 4,
                 # Deep MLP
                 hidden_dims: list = None,
                 dropout: float = 0.0,
                 # Wide bypass
                 n_cross_features: int = 0,
                 # User-side genome toggles. When BOTH are False, the model skips
                 # genome_buffer[X_history] entirely → major speedup on MPS (no
                 # 920 MB lookup per batch). The user side keeps history_pool +
                 # genre + ts; user-item genome similarity still flows in via
                 # the genome_cosine cross feature (wide bypass) + the item-side
                 # item_genome_tower on the candidate.
                 use_user_watch_history_genome_pool: bool = True,   # tower-then-pool over history (CG-style)
                 use_user_genome_context: bool = True,              # pool-then-tower over raw history genome
                 ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        self.pad_idx          = n_movies
        self.n_genome_tags    = n_genome_tags
        self.n_cross_features = n_cross_features
        self.use_user_watch_history_genome_pool = use_user_watch_history_genome_pool
        self.use_user_genome_context            = use_user_genome_context
        self._uses_history_genome = (use_user_watch_history_genome_pool
                                     or use_user_genome_context)

        # ── Buffers (lookup-by-corpus-idx) ────────────────────────────────────
        # Persistent: genome scores travel with the checkpoint (large, dataset-specific).
        self.register_buffer('genome_buffer', genome_buffer)
        # Non-persistent: rebuilt from FeatureStore on every load.
        self.register_buffer('genre_buffer', genre_buffer, persistent=False)
        self.register_buffer('tag_buffer',   tag_buffer,   persistent=False)
        self.register_buffer('year_buffer',  year_buffer,  persistent=False)

        # ── Embedding lookups ─────────────────────────────────────────────────
        # Item ID lookup — shared between item-side tower and user history pool.
        self.item_id_lookup = nn.Embedding(n_movies + 1, item_id_emb_dim, padding_idx=n_movies)
        self.year_lookup    = nn.Embedding(n_years, item_year_emb_dim)
        self.ts_lookup      = nn.Embedding(n_ts_bins, ts_emb_dim)

        # ── Sub-towers ────────────────────────────────────────────────────────
        # Item side
        self.item_id_tower = nn.Sequential(
            nn.Linear(item_id_emb_dim, item_id_emb_dim),
            nn.ReLU(),
        )
        self.item_genre_tower = nn.Sequential(
            nn.Linear(n_genres, item_genre_emb_dim),
            nn.ReLU(),
        )
        self.item_tag_tower = nn.Sequential(
            nn.Linear(n_tags, item_tag_emb_dim),
            nn.ReLU(),
        )
        # SHARED: applied on both item side and user genome pool side.
        self.item_genome_tower = nn.Sequential(
            nn.Linear(n_genome_tags, item_genome_emb_dim),
            nn.ReLU(),
        )
        self.year_tower = nn.Sequential(
            nn.Linear(item_year_emb_dim, item_year_emb_dim),
            nn.ReLU(),
        )

        # User side
        self.user_genre_tower = nn.Sequential(
            nn.Linear(user_context_size, user_genre_emb_dim),
            nn.ReLU(),
        )
        self.ts_tower = nn.Sequential(
            nn.Linear(ts_emb_dim, ts_emb_dim),
            nn.ReLU(),
        )
        # Distinct from item_genome_tower: takes the rating-weighted avg of raw
        # genome scores over watch history (not per-movie genome).
        self.user_genome_ctx_tower = nn.Sequential(
            nn.Linear(n_genome_tags, user_genome_ctx_emb_dim),
            nn.ReLU(),
        )

        # ── Deep MLP ──────────────────────────────────────────────────────────
        user_concat_dim = (item_id_emb_dim
                           + (item_genome_emb_dim     if use_user_watch_history_genome_pool else 0)
                           + (user_genome_ctx_emb_dim if use_user_genome_context            else 0)
                           + user_genre_emb_dim + ts_emb_dim)
        item_concat_dim = (item_genre_emb_dim + item_tag_emb_dim
                           + item_genome_emb_dim
                           + item_id_emb_dim + item_year_emb_dim)
        deep_in = user_concat_dim + item_concat_dim

        deep_layers = []
        in_dim = deep_in
        for h in hidden_dims:
            deep_layers.append(nn.Linear(in_dim, h))
            deep_layers.append(nn.ReLU())
            if dropout > 0:
                deep_layers.append(nn.Dropout(dropout))
            in_dim = h
        self.deep = nn.Sequential(*deep_layers)
        last_hidden = hidden_dims[-1]

        # ── Head: deep representation + cross-feature wide bypass ────────────
        self.head = nn.Linear(last_hidden + n_cross_features, 1)

        # Public dims (used by train.py for the summary line)
        self.user_concat_dim = user_concat_dim
        self.item_concat_dim = item_concat_dim
        self.deep_in         = deep_in

        # ── Init ──────────────────────────────────────────────────────────────
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight, gain=0.01)

    # ── User tower ────────────────────────────────────────────────────────────

    def user_embedding(self, user_genre_ctx, X_history, X_hist_ratings, timestamp):
        """
        user_genre_ctx:  (B, user_context_size)
        X_history:       (B, max_hist) int64 — corpus indices, padding = self.pad_idx
        X_hist_ratings:  (B, max_hist) float — debiased ratings
        timestamp:       (B,) int64 — timestamp bin
        Returns: (B, user_concat_dim)
        """
        pad_mask     = (X_history != self.pad_idx).float().unsqueeze(-1)        # (B, H, 1)
        rat_w        = X_hist_ratings.unsqueeze(-1) * pad_mask                  # (B, H, 1)
        weight_sum   = rat_w.abs().sum(dim=1).clamp(min=1e-6)                   # (B, 1)

        # Rating-weighted avg pool over item ID embeddings (shared with item side).
        history_embs = self.item_id_lookup(X_history)                           # (B, H, id_dim)
        history_pool = (history_embs * rat_w).sum(dim=1) / weight_sum           # (B, id_dim)

        parts = [history_pool]

        # Skip the entire genome_buffer[X_history] lookup (920 MB at B=4096) when
        # neither user-side genome feature is enabled. Major speedup on MPS.
        if self._uses_history_genome:
            watched_genome = self.genome_buffer[X_history]                       # (B, H, n_genome)

            if self.use_user_watch_history_genome_pool:
                # Tower-then-pool (CG-style). Per-item Linear(1128→32)+ReLU then average.
                genome_embs = self.item_genome_tower(watched_genome)             # (B, H, genome_dim)
                parts.append((genome_embs * rat_w).sum(dim=1) / weight_sum)      # (B, genome_dim)

            if self.use_user_genome_context:
                # Pool-then-tower over raw genome. Use bmm to avoid the (B,H,1128)
                # broadcast-multiply intermediate that (raw * rat_w).sum(dim=1) creates.
                # rat_w shape: (B, H, 1) → transpose to (B, 1, H) for bmm.
                genome_ctx_raw = torch.bmm(rat_w.transpose(1, 2),
                                           watched_genome).squeeze(1) / weight_sum  # (B, n_genome)
                parts.append(self.user_genome_ctx_tower(genome_ctx_raw))         # (B, ctx_dim)

        parts.append(self.user_genre_tower(user_genre_ctx))                     # (B, user_genre_dim)
        parts.append(self.ts_tower(self.ts_lookup(timestamp)))                  # (B, ts_dim)

        return torch.cat(parts, dim=1)

    # ── Item tower ────────────────────────────────────────────────────────────

    def item_embedding(self, cand_idx):
        """
        cand_idx: (B,) int64 — corpus index of candidate item
        Returns: (B, item_concat_dim)
        """
        item_genre_emb  = self.item_genre_tower(self.genre_buffer[cand_idx])
        item_tag_emb    = self.item_tag_tower(self.tag_buffer[cand_idx])
        item_genome_emb = self.item_genome_tower(self.genome_buffer[cand_idx])
        item_id_emb     = self.item_id_tower(self.item_id_lookup(cand_idx))
        year_emb        = self.year_tower(self.year_lookup(self.year_buffer[cand_idx]))
        return torch.cat([item_genre_emb, item_tag_emb, item_genome_emb, item_id_emb, year_emb], dim=1)

    # ── Score pairs (deep MLP + head with wide bypass) ───────────────────────

    def score_pairs(self,
                    user_concat: torch.Tensor,
                    item_concat: torch.Tensor,
                    cross_features: torch.Tensor) -> torch.Tensor:
        """
        Score pre-computed user / item / cross feature blocks. All tensors must have
        the same leading batch dim B.

          user_concat:    (B, user_concat_dim)
          item_concat:    (B, item_concat_dim)
          cross_features: (B, n_cross_features)

        Returns: (B,) raw logits.

        Pulled out of `forward` so eval code can compute user_embedding once per row,
        item_embedding per candidate, and call score_pairs with cheap broadcast expansions
        of user_concat (avoids 250× redundant user-tower work in compute_label_ranks).
        """
        deep_out = self.deep(torch.cat([user_concat, item_concat], dim=1))
        if self.n_cross_features > 0:
            combined = torch.cat([deep_out, cross_features], dim=1)
        else:
            combined = deep_out
        return self.head(combined).squeeze(-1)

    # ── Forward: tower outputs → deep MLP → head with wide bypass ────────────

    def forward(self,
                user_genre_ctx,
                X_history,
                X_hist_ratings,
                timestamp,
                cand_idx,
                cross_features):
        """
        cross_features: (B, n_cross_features) — wide-bypass scalars (genome_cosine,
                        genre_affinity, era_gap, rating_cal, pop_match)
        Returns: (B,) raw logits.
        """
        user_concat = self.user_embedding(user_genre_ctx, X_history, X_hist_ratings, timestamp)
        item_concat = self.item_embedding(cand_idx)
        return self.score_pairs(user_concat, item_concat, cross_features)
