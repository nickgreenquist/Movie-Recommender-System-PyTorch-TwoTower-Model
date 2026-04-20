"""
Two-Tower MovieRecommender model.
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
                 item_genre_embedding_size=20,
                 item_tag_embedding_size=15,
                 item_genome_tag_embedding_size=35,
                 item_movieId_embedding_size=40,
                 item_year_embedding_size=10,
                 user_genre_embedding_size=35,
                 timestamp_feature_embedding_size=10,
                ):
        """
        Fixed architecture — item tag and genome tag towers are always active,
        user tag tower is not used.

        User genome pooling reuses item_genome_tag_tower (shared tower):
            watched movies → item_genome_tag_tower(each movie's genome context)
                           → rating-weighted avg pool → user genome embedding
        This keeps the user and item genome embeddings in the same space.

        Dimension constraint (must be equal for dot product):
            user: item_movieId_embedding_size + item_genome_tag_embedding_size
                  + user_genre_embedding_size + timestamp_feature_embedding_size
            item: item_genre_embedding_size + item_tag_embedding_size
                  + item_genome_tag_embedding_size
                  + item_movieId_embedding_size + item_year_embedding_size

        genome_context_buffer: float32 tensor of shape (top_movies_len + 1, genome_tags_len).
            Row i is the genome tag context for the movie at embedding index i.
            Last row (index top_movies_len) is zeros — used as padding.
            Built in build_model() from fs.movieId_to_genome_tag_context.
        """
        super().__init__()

        self.pad_idx = top_movies_len

        # genome_context_buffer: non-trainable, device-portable, saved in state_dict
        self.register_buffer('genome_context_buffer', genome_context_buffer)

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
        self.timestamp_embedding_lookup = nn.Embedding(timestamp_num_bins, timestamp_feature_embedding_size)
        self.timestamp_embedding_tower = nn.Sequential(
            nn.Linear(timestamp_feature_embedding_size, timestamp_feature_embedding_size),
            nn.Tanh()
        )

        # ── Dimension check ───────────────────────────────────────────────────
        user_side = (item_movieId_embedding_size + item_genome_tag_embedding_size
                     + user_genre_embedding_size + timestamp_feature_embedding_size)
        item_side = (item_genre_embedding_size + item_tag_embedding_size
                     + item_genome_tag_embedding_size
                     + item_movieId_embedding_size + item_year_embedding_size)
        if user_side != item_side:
            raise ValueError(
                f"User embedding size ({user_side} = history {item_movieId_embedding_size} + "
                f"genome {item_genome_tag_embedding_size} + "
                f"genre {user_genre_embedding_size} + timestamp {timestamp_feature_embedding_size}) "
                f"must match item embedding size ({item_side} = genre {item_genre_embedding_size} + "
                f"tag {item_tag_embedding_size} + genome_tag {item_genome_tag_embedding_size} + "
                f"movieId {item_movieId_embedding_size} + year {item_year_embedding_size}). "
                f"Adjust embedding sizes."
            )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight, gain=0.01)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.xavier_uniform_(module.weight, gain=0.01)

    def user_embedding(self, user_genre_contexts, user_watch_history,
                       user_watch_history_ratings, timestamps):
        """User tower: returns (batch, embedding_dim)."""
        history_embs   = self.item_embedding_lookup(user_watch_history)
        pad_mask       = (user_watch_history != self.pad_idx).float().unsqueeze(-1)
        rating_weights = user_watch_history_ratings.unsqueeze(-1) * pad_mask
        weight_sum     = rating_weights.abs().sum(dim=1).clamp(min=1e-6)
        history_emb    = (history_embs * rating_weights).sum(dim=1) / weight_sum

        # nn.Linear + Tanh broadcast: (batch, hist_len, 1128) → (batch, hist_len, 35)
        watched_genome = self.genome_context_buffer[user_watch_history]
        genome_embs    = self.item_genome_tag_tower(watched_genome)
        genome_emb     = (genome_embs * rating_weights).sum(dim=1) / weight_sum

        genre_emb = self.user_genre_tower(user_genre_contexts)
        ts_emb    = self.timestamp_embedding_tower(self.timestamp_embedding_lookup(timestamps))
        return torch.cat([history_emb, genome_emb, genre_emb, ts_emb], dim=1)

    def item_embedding(self, movie_genres, movie_tags, movie_genome_tags, years, target_movieId):
        """Item tower: returns (batch, embedding_dim)."""
        item_genre_emb  = self.item_genre_tower(movie_genres)
        item_tag_emb    = self.item_tag_tower(movie_tags)
        item_genome_emb = self.item_genome_tag_tower(movie_genome_tags)
        item_emb        = self.item_embedding_tower(self.item_embedding_lookup(target_movieId))
        year_emb        = self.year_embedding_tower(self.year_embedding_lookup(years))
        return torch.cat([item_genre_emb, item_tag_emb, item_genome_emb, item_emb, year_emb], dim=1)

    def forward(self, user_genre_contexts, user_watch_history,
                user_watch_history_ratings, timestamps,
                movie_genres, movie_tags, movie_genome_tags, years, target_movieId):
        """
        Args:
            user_genre_contexts        (Tensor): (batch, user_context_size)
            user_watch_history         (Tensor): (batch, max_hist_len)      padded movie ID indices
            user_watch_history_ratings (Tensor): (batch, max_hist_len)      debiased ratings; 0 at padding
            timestamps                 (Tensor): (batch,)
            movie_genres               (Tensor): (batch, genres_len)
            movie_tags                 (Tensor): (batch, tags_len)
            movie_genome_tags          (Tensor): (batch, genome_tags_len)
            years                      (Tensor): (batch,)
            target_movieId             (Tensor): (batch,)
        """
        user_combined = self.user_embedding(user_genre_contexts, user_watch_history,
                                            user_watch_history_ratings, timestamps)
        item_combined = self.item_embedding(movie_genres, movie_tags, movie_genome_tags,
                                            years, target_movieId)
        return torch.einsum('ij, ij -> i', user_combined, item_combined)
