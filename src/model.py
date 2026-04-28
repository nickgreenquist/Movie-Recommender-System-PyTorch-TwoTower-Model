"""
Two-Tower MovieRecommender model.

User tower:
  id_pool(32) + genome_pool(32) + genre(32) + ts(4) [+ genome_ctx(32)] → proj MLP → 128

Item tower:
  genre(8) + tag(16) + genome(32) + id(32) + year(8) = 96 → proj MLP → 128

genre_context_buffer, tag_context_buffer, year_context_buffer are non-persistent buffers
(not saved in state_dict) — rebuilt from the feature store on every load.
"""
import torch
import torch.nn as nn


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
                 item_genome_tag_embedding_size=16,
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
        genome_context_buffer: float32 tensor (top_movies_len + 1, genome_tags_len).
            Row i = genome tag context for movie at embedding index i.
            Last row (index top_movies_len) = zeros — used as padding.

        genre_context_buffer: float32 (top_movies_len + 1, genres_len), non-persistent.
        tag_context_buffer:   float32 (top_movies_len + 1, tags_len), non-persistent.
        year_context_buffer:  int64   (top_movies_len + 1,), non-persistent.
        """
        super().__init__()

        self.pad_idx = top_movies_len

        self.register_buffer('genome_context_buffer', genome_context_buffer)
        # Non-persistent: not saved in state_dict, rebuilt from feature store on every load.
        self.register_buffer('genre_context_buffer', genre_context_buffer, persistent=False)
        self.register_buffer('tag_context_buffer',   tag_context_buffer,   persistent=False)
        self.register_buffer('year_context_buffer',  year_context_buffer,  persistent=False)

        # ── Shared item embedding ─────────────────────────────────────────────
        self.item_embedding_lookup = nn.Embedding(
            top_movies_len + 1, item_movieId_embedding_size, padding_idx=top_movies_len
        )
        self.item_embedding_tower = nn.Sequential(
            nn.Linear(item_movieId_embedding_size, item_movieId_embedding_size),
            nn.Tanh()
        )

        # ── Item feature towers ───────────────────────────────────────────────
        self.item_genre_tower = nn.Sequential(
            nn.Linear(genres_len, item_genre_embedding_size),
            nn.Tanh()
        )
        self.item_tag_tower = nn.Sequential(
            nn.Linear(tags_len, item_tag_embedding_size),
            nn.Tanh()
        )
        # Shared between item side (target movie) and user side (genome pooling)
        self.item_genome_tag_tower = nn.Sequential(
            nn.Linear(genome_tags_len, item_genome_tag_embedding_size),
            nn.Tanh()
        )
        self.year_embedding_lookup = nn.Embedding(all_years_len, item_year_embedding_size)
        self.year_embedding_tower = nn.Sequential(
            nn.Linear(item_year_embedding_size, item_year_embedding_size),
            nn.Tanh()
        )

        # ── User towers ───────────────────────────────────────────────────────
        self.user_genre_tower = nn.Sequential(
            nn.Linear(user_context_size, user_genre_embedding_size),
            nn.Tanh()
        )
        self.timestamp_embedding_lookup = nn.Embedding(
            timestamp_num_bins, timestamp_feature_embedding_size
        )
        self.timestamp_embedding_tower = nn.Sequential(
            nn.Linear(timestamp_feature_embedding_size, timestamp_feature_embedding_size),
            nn.Tanh()
        )
        # Separate tower for the pre-aggregated user genome taste vector (1128 → gctx_dim).
        # Distinct from item_genome_tag_tower which processes per-movie genome scores.
        self.user_genome_context_tower = nn.Sequential(
            nn.Linear(genome_tags_len, user_genome_context_embedding_size),
            nn.Tanh()
        )

        # ── Projection MLPs ───────────────────────────────────────────────────
        user_concat_dim = (item_movieId_embedding_size + item_genome_tag_embedding_size
                           + user_genre_embedding_size + timestamp_feature_embedding_size
                           + user_genome_context_embedding_size)
        item_concat_dim = (item_genre_embedding_size + item_tag_embedding_size
                           + item_genome_tag_embedding_size
                           + item_movieId_embedding_size + item_year_embedding_size)
        # No activation on the final linear — feeds directly into dot product.
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

        # Sub-tower linears: gain=0.1 (not 0.01 — projection adds extra layers)
        self.apply(self._init_weights)
        # Projection layers re-initialized at gain=1.0 after the rest.
        # Without this, gain=0.1^2 compounds across sub-tower + projection and
        # collapses dot products to zero before training starts.
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

    def _genome_ctx_emb(self, user_watch_history, rating_weights, weight_sum):
        """Rating-weighted avg of raw genome scores → user_genome_context_tower. Returns (B, gctx_dim)."""
        watched_genome_raw = self.genome_context_buffer[user_watch_history]
        genome_ctx_raw     = (watched_genome_raw * rating_weights).sum(dim=1) / weight_sum
        return self.user_genome_context_tower(genome_ctx_raw)

    def user_embedding(self, user_genre_contexts, user_watch_history,
                       user_watch_history_ratings, timestamps):
        """User tower: returns (batch, output_dim)."""
        pad_mask       = (user_watch_history != self.pad_idx).float().unsqueeze(-1)
        rating_weights = user_watch_history_ratings.unsqueeze(-1) * pad_mask
        weight_sum     = rating_weights.abs().sum(dim=1).clamp(min=1e-6)

        genre_emb    = self.user_genre_tower(user_genre_contexts)
        ts_emb       = self.timestamp_embedding_tower(self.timestamp_embedding_lookup(timestamps))
        history_embs = self.item_embedding_lookup(user_watch_history)
        history_emb  = (history_embs * rating_weights).sum(dim=1) / weight_sum

        watched_genome = self.genome_context_buffer[user_watch_history]
        genome_embs    = self.item_genome_tag_tower(watched_genome)
        genome_emb     = (genome_embs * rating_weights).sum(dim=1) / weight_sum
        parts = [history_emb, genome_emb, genre_emb, ts_emb]

        parts.append(self._genome_ctx_emb(user_watch_history, rating_weights, weight_sum))
        return self.user_projection(torch.cat(parts, dim=1))

    def item_embedding(self, movie_genres, movie_tags, movie_genome_tags, years, target_movieId):
        """Item tower: returns (batch, output_dim)."""
        item_genre_emb  = self.item_genre_tower(movie_genres)
        item_tag_emb    = self.item_tag_tower(movie_tags)
        item_genome_emb = self.item_genome_tag_tower(movie_genome_tags)
        item_emb        = self.item_embedding_tower(self.item_embedding_lookup(target_movieId))
        year_emb        = self.year_embedding_tower(self.year_embedding_lookup(years))
        concat = torch.cat([item_genre_emb, item_tag_emb, item_genome_emb, item_emb, year_emb], dim=1)
        return self.item_projection(concat)

    def forward(self, user_genre_contexts, user_watch_history,
                user_watch_history_ratings, timestamps, target_movieId):
        user_combined = self.user_embedding(user_genre_contexts, user_watch_history,
                                            user_watch_history_ratings, timestamps)
        item_combined = self.item_embedding(
            self.genre_context_buffer[target_movieId],
            self.tag_context_buffer[target_movieId],
            self.genome_context_buffer[target_movieId],
            self.year_context_buffer[target_movieId],
            target_movieId,
        )
        return torch.einsum('ij, ij -> i', user_combined, item_combined)
