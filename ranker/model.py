"""
Stage 2 — Wide & Deep ranker with full v3 CG feature parity.

Mirrors v3 src/model.py feature set: same per-feature towers, same dimensions,
same shared item_id_lookup between user pools and item-side tower, same
LayerNorm-then-sum 4-pool user history (full / liked / disliked / weighted).
All parameters are ranker-owned — buffers carry static data (genome scores,
genre one-hot, etc.) but no CG weights are referenced at runtime. Init-time
warm-start from a CG checkpoint is handled in train.py and does not change
ownership: copies values into the ranker state_dict, then the ranker fine-tunes.

Key difference from CG:
  - No projection MLP and no L2 normalization. CG projects user/item to
    L2-normalized 128-dim vectors for cosine softmax. The ranker concatenates
    user_concat + item_concat directly into a deep MLP, then a head with
    cross-feature wide bypass. We score with BCE + raw logits, not cosine.
  - Cross features in the wide bypass (genome cosine + planned cross features)
    are the ranker's actual edge; the deep MLP is the CG-equivalent baseline.

User concat (196): pool_full(32) + pool_liked(32) + pool_disliked(32)
                   + pool_weighted(32) + genome_ctx(32) + genre_emb(32) + ts_emb(4)
Item concat  (96): item_genre(8) + item_tag(16) + item_genome(32)
                   + item_id(32) + year(8)
Deep MLP   (292 → [256, 128, 64]) → 64
Head: cat(deep_out(64), cross_features(n_cross)) → Linear(64+n_cross, 1)

ZERO src/ imports — buffers built externally from FeatureStore and passed in.
"""
from collections import namedtuple

import torch
import torch.nn as nn


# Slice layout of `user_concat` returned by WideDeepRanker.user_embedding() / user_forward().
# Callers (eval / canary / cross-feature computation) use this to extract specific pools
# without hardcoding offsets. Keep in sync with the order in user_concat construction.
USER_CONCAT_LAYOUT = {
    'pool_full':     (0, 32),
    'pool_liked':    (32, 64),
    'pool_disliked': (64, 96),
    'pool_weighted': (96, 128),
    'genome_ctx':    (128, 160),
    'genre_emb':     (160, 192),
    'ts_emb':        (192, 196),
}


# Bundled result of WideDeepRanker.user_forward() — the unified user-side pass that
# shares one (B, H, n_genome) genome_buffer lookup across all auxiliary signals.
UserForwardResult = namedtuple('UserForwardResult', [
    'user_concat',           # (B, user_concat_dim) — main user embedding
    'profile',               # (B, n_genome)        — rating-weighted avg of history genome
    'baseline',              # (B,)                 — mean cosine of profile vs history items
    'last_item_emb',         # (B, item_id_emb_dim) — most recent watch's item ID embedding
    'last_k_mean_genome',    # (B, n_genome)        — mean genome over last k real watches
])


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
                 # Sub-tower output dims (CG v3 defaults)
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
                 ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        self.pad_idx          = n_movies
        self.n_genome_tags    = n_genome_tags
        self.n_cross_features = n_cross_features

        # ── Buffers (lookup-by-corpus-idx) ────────────────────────────────────
        # Persistent: genome scores travel with the checkpoint (large, dataset-specific).
        self.register_buffer('genome_buffer', genome_buffer)
        # Non-persistent: rebuilt from FeatureStore on every load.
        self.register_buffer('genre_buffer', genre_buffer, persistent=False)
        self.register_buffer('tag_buffer',   tag_buffer,   persistent=False)
        self.register_buffer('year_buffer',  year_buffer,  persistent=False)

        # ── Embedding lookups ─────────────────────────────────────────────────
        # Item ID lookup — shared between item-side tower and all 4 user history pools.
        self.item_id_lookup = nn.Embedding(n_movies + 1, item_id_emb_dim, padding_idx=n_movies)
        self.year_lookup    = nn.Embedding(n_years, item_year_emb_dim)
        self.ts_lookup      = nn.Embedding(n_ts_bins, ts_emb_dim)

        # ── Sub-towers ────────────────────────────────────────────────────────
        # Item side (mirrors CG item tower)
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
        # Item-side genome compression. (CG v3 calls this item_genome_tag_tower.)
        self.item_genome_tower = nn.Sequential(
            nn.Linear(n_genome_tags, item_genome_emb_dim),
            nn.ReLU(),
        )
        self.year_tower = nn.Sequential(
            nn.Linear(item_year_emb_dim, item_year_emb_dim),
            nn.ReLU(),
        )

        # User side (mirrors CG user tower)
        self.user_genre_tower = nn.Sequential(
            nn.Linear(user_context_size, user_genre_emb_dim),
            nn.ReLU(),
        )
        self.ts_tower = nn.Sequential(
            nn.Linear(ts_emb_dim, ts_emb_dim),
            nn.ReLU(),
        )
        # Pool-then-tower over raw genome scores: rating-weighted avg → Linear(1128→32).
        # Distinct from item_genome_tower (per-movie genome compression on item side).
        self.user_genome_ctx_tower = nn.Sequential(
            nn.Linear(n_genome_tags, user_genome_ctx_emb_dim),
            nn.ReLU(),
        )

        # ── 4 sum pools with LayerNorm (mirrors CG v3 user history) ──────────
        self.hist_full_norm     = nn.LayerNorm(item_id_emb_dim)
        self.hist_liked_norm    = nn.LayerNorm(item_id_emb_dim)
        self.hist_disliked_norm = nn.LayerNorm(item_id_emb_dim)
        self.hist_weighted_norm = nn.LayerNorm(item_id_emb_dim)

        # ── Deep MLP ──────────────────────────────────────────────────────────
        user_concat_dim = (4 * item_id_emb_dim
                           + user_genome_ctx_emb_dim
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

        # ── Init (mirrors CG: gain=0.1 on sub-towers, gain=1.0 on deep + head) ─
        self.apply(self._init_weights)
        for module in [self.deep, self.head]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.1)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight, gain=0.01)

    # ── User tower ────────────────────────────────────────────────────────────

    def user_forward(self, user_genre_ctx, X_history, X_hist_ratings, timestamp, k: int = 5):
        """
        Unified user-side forward pass. Computes user_concat AND every auxiliary signal
        the cross features need (profile, baseline, last_item_emb, last_k_mean_genome)
        in ONE pass with a SINGLE genome_buffer[X_history] lookup.

        The (B, H, n_genome) lookup allocates ~922 MB at B=4096, H=50, n_genome=1128.
        Doing it 3x (one per signal: user_embedding's genome_ctx + user_genome_profile +
        user_genome_baseline) was the dominant per-batch cost on MPS — this collapses
        all three into one. The profile is also reused for genome_ctx_emb (avoiding
        recomputation) and for baseline (via bmm). Recent signals slice from hist_genome
        (zero-copy view).

        Returns UserForwardResult NamedTuple. Empty-history users get safe defaults
        (zero last_item_emb, zero baseline).
        """
        pad           = self.pad_idx
        is_real       = (X_history != pad)
        is_real_f     = is_real.float()                                             # (B, H)

        # ── 4 user pools (item_id_lookup over full / liked / disliked) ───────
        liked_mask    = is_real & (X_hist_ratings > 0)
        disliked_mask = is_real & (X_hist_ratings < 0)
        X_hist_liked    = torch.where(liked_mask,    X_history, torch.full_like(X_history, pad))
        X_hist_disliked = torch.where(disliked_mask, X_history, torch.full_like(X_history, pad))

        full_embs     = self.item_id_lookup(X_history)                              # (B, H, id_dim)
        pool_full     = self.hist_full_norm(full_embs.sum(dim=1))
        pool_weighted = self.hist_weighted_norm(
            (full_embs * X_hist_ratings.unsqueeze(-1)).sum(dim=1))
        pool_liked    = self.hist_liked_norm(self.item_id_lookup(X_hist_liked).sum(dim=1))
        pool_disliked = self.hist_disliked_norm(self.item_id_lookup(X_hist_disliked).sum(dim=1))

        # ── ONE genome lookup, shared across genome_ctx / profile / baseline / recent ──
        hist_genome    = self.genome_buffer[X_history]                              # (B, H, n_genome)

        # Profile: rating-weighted avg of history genome (also feeds genome_ctx_tower).
        rating_weights = X_hist_ratings.unsqueeze(-1) * is_real_f.unsqueeze(-1)     # (B, H, 1)
        weight_sum     = rating_weights.abs().sum(dim=1).clamp(min=1e-6)            # (B, 1)
        profile        = torch.bmm(rating_weights.transpose(1, 2),
                                    hist_genome).squeeze(1) / weight_sum            # (B, n_genome)
        genome_ctx_emb = self.user_genome_ctx_tower(profile)                        # (B, ctx_dim)

        # ── Genre / TS sub-towers ────────────────────────────────────────────
        genre_emb = self.user_genre_tower(user_genre_ctx)                           # (B, user_genre_dim)
        ts_emb    = self.ts_tower(self.ts_lookup(timestamp))                        # (B, ts_dim)

        user_concat = torch.cat([pool_full, pool_liked, pool_disliked, pool_weighted,
                                  genome_ctx_emb, genre_emb, ts_emb], dim=1)

        # ── Baseline: mean cosine of profile vs each history item via bmm ───
        # (Avoids materializing a (B, H, n_genome) normalized intermediate.)
        unnorm_dot   = torch.bmm(profile.unsqueeze(1),
                                  hist_genome.transpose(1, 2)).squeeze(1)            # (B, H)
        profile_norm = profile.norm(dim=1).clamp(min=1e-8)                           # (B,)
        hist_norm    = hist_genome.norm(dim=2).clamp(min=1e-8)                       # (B, H)
        cos_per_item = unnorm_dot / (profile_norm.unsqueeze(1) * hist_norm)          # (B, H)
        n_real       = is_real_f.sum(dim=1).clamp(min=1.0)                           # (B,)
        baseline     = (cos_per_item * is_real_f).sum(dim=1) / n_real                # (B,)

        # ── Recent signals: slice from hist_genome (zero-copy view) ──────────
        last_idx       = X_history[:, -1]                                            # (B,)
        last_real_mask = (last_idx != pad).float().unsqueeze(-1)                     # (B, 1)
        last_item_emb  = self.item_id_lookup(last_idx) * last_real_mask              # (B, item_id_dim)

        last_k_genome  = hist_genome[:, -k:, :]                                      # (B, k, n_genome) view
        last_k_real    = is_real_f[:, -k:]                                           # (B, k) view
        last_k_n       = last_k_real.sum(dim=1).clamp(min=1.0).unsqueeze(-1)         # (B, 1)
        last_k_mean    = (last_k_genome * last_k_real.unsqueeze(-1)).sum(dim=1) / last_k_n

        return UserForwardResult(user_concat, profile, baseline, last_item_emb, last_k_mean)

    def user_embedding(self, user_genre_ctx, X_history, X_hist_ratings, timestamp):
        """
        user_genre_ctx:  (B, user_context_size)
        X_history:       (B, max_hist) int64 — corpus indices, padding = self.pad_idx
        X_hist_ratings:  (B, max_hist) float — debiased ratings
        timestamp:       (B,) int64 — timestamp bin
        Returns: (B, user_concat_dim) — concat of 4 pools + genome_ctx + genre + ts.

        Liked/disliked subsets are derived on the fly from (X_history, X_hist_ratings)
        rather than passed as separate tensors — keeps the dataset/eval call signature
        unchanged. Equivalent to CG's precomputed X_hist_liked / X_hist_disliked.
        """
        pad           = self.pad_idx
        # Liked: positions where rating > 0 (and not padding); else map to pad_idx.
        # Disliked: positions where rating < 0 (and not padding); else map to pad_idx.
        # padding_idx on item_id_lookup zeroes out pad rows in the sum.
        is_real       = (X_history != pad)
        liked_mask    = is_real & (X_hist_ratings > 0)
        disliked_mask = is_real & (X_hist_ratings < 0)
        X_hist_liked    = torch.where(liked_mask,    X_history, torch.full_like(X_history, pad))
        X_hist_disliked = torch.where(disliked_mask, X_history, torch.full_like(X_history, pad))

        # Look up full history once; reuse for both pool_full and pool_weighted.
        full_embs = self.item_id_lookup(X_history)                              # (B, H, id_dim)
        pool_full     = self.hist_full_norm(full_embs.sum(dim=1))
        pool_weighted = self.hist_weighted_norm(
            (full_embs * X_hist_ratings.unsqueeze(-1)).sum(dim=1))
        pool_liked    = self.hist_liked_norm(self.item_id_lookup(X_hist_liked).sum(dim=1))
        pool_disliked = self.hist_disliked_norm(self.item_id_lookup(X_hist_disliked).sum(dim=1))

        # Rating-weighted avg of raw genome scores → user_genome_ctx_tower.
        # bmm avoids the (B, H, n_genome) broadcast-multiply intermediate.
        rating_weights = X_hist_ratings.unsqueeze(-1) * is_real.float().unsqueeze(-1)  # (B, H, 1)
        weight_sum     = rating_weights.abs().sum(dim=1).clamp(min=1e-6)               # (B, 1)
        watched_genome = self.genome_buffer[X_history]                                  # (B, H, n_genome)
        genome_ctx_raw = torch.bmm(rating_weights.transpose(1, 2),
                                    watched_genome).squeeze(1) / weight_sum            # (B, n_genome)
        genome_ctx_emb = self.user_genome_ctx_tower(genome_ctx_raw)                    # (B, ctx_dim)

        genre_emb = self.user_genre_tower(user_genre_ctx)                              # (B, user_genre_dim)
        ts_emb    = self.ts_tower(self.ts_lookup(timestamp))                           # (B, ts_dim)

        return torch.cat([pool_full, pool_liked, pool_disliked, pool_weighted,
                          genome_ctx_emb, genre_emb, ts_emb], dim=1)

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
