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
                 item_genre_embedding_size=10,
                 item_tag_embedding_size=20,
                 item_genome_tag_embedding_size=20,
                 item_movieId_embedding_size=40,
                 item_year_embedding_size=10,
                 user_genre_embedding_size=50,
                 user_tag_embedding_size=0,
                 timestamp_feature_embedding_size=10,
                 use_user_tag_tower=False,
                 use_item_tag_tower=True,
                 use_item_genome_tag_tower=True,
                ):
        """
        Dimension constraint:
            user: item_movieId_embedding_size + user_genre_embedding_size
                  + timestamp_feature_embedding_size + (user_tag_embedding_size if use_user_tag_tower else 0)
            item: item_genre_embedding_size + (item_tag_embedding_size if use_item_tag_tower else 0)
                  + (item_genome_tag_embedding_size if use_item_genome_tag_tower else 0)
                  + item_movieId_embedding_size + item_year_embedding_size
            Both must be equal.
        """
        super().__init__()

        self.use_user_tag_tower        = use_user_tag_tower
        self.use_item_tag_tower        = use_item_tag_tower
        self.use_item_genome_tag_tower = use_item_genome_tag_tower

        self.pad_idx = top_movies_len

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
        if use_item_tag_tower:
            self.item_tag_tower = nn.Sequential(
                nn.Linear(tags_len, item_tag_embedding_size),
                nn.Tanh()
            )
        if use_item_genome_tag_tower:
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
        if use_user_tag_tower:
            self.user_tag_tower = nn.Sequential(
                nn.Linear(tags_len, user_tag_embedding_size),
                nn.Tanh()
            )
        self.timestamp_embedding_lookup = nn.Embedding(timestamp_num_bins, timestamp_feature_embedding_size)
        self.timestamp_embedding_tower = nn.Sequential(
            nn.Linear(timestamp_feature_embedding_size, timestamp_feature_embedding_size),
            nn.Tanh()
        )

        # ── Dimension check ───────────────────────────────────────────────────
        effective_user_tag   = user_tag_embedding_size        if use_user_tag_tower        else 0
        effective_item_tag   = item_tag_embedding_size        if use_item_tag_tower        else 0
        effective_genome_tag = item_genome_tag_embedding_size if use_item_genome_tag_tower else 0
        user_side = (item_movieId_embedding_size + user_genre_embedding_size
                     + timestamp_feature_embedding_size + effective_user_tag)
        item_side = (item_genre_embedding_size + effective_item_tag + effective_genome_tag
                     + item_movieId_embedding_size + item_year_embedding_size)
        if user_side != item_side:
            raise ValueError(
                f"User embedding size ({user_side} = history {item_movieId_embedding_size} + "
                f"genre {user_genre_embedding_size} + timestamp {timestamp_feature_embedding_size} + "
                f"tag {effective_user_tag}) "
                f"must match item embedding size ({item_side} = genre {item_genre_embedding_size} + "
                f"tag {effective_item_tag} + genome_tag {effective_genome_tag} + "
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

    def forward(self, user_genre_contexts, user_tag_contexts, user_watch_history,
                user_watch_history_ratings, timestamps,
                movie_genres, movie_tags, movie_genome_tags, years, target_movieId):
        """
        Args:
            user_genre_contexts        (Tensor): (batch, user_context_size)
            user_tag_contexts          (Tensor): (batch, tags_len)
            user_watch_history         (Tensor): (batch, max_hist_len)      padded movie ID indices
            user_watch_history_ratings (Tensor): (batch, max_hist_len)      debiased ratings; 0 at padding
            timestamps                 (Tensor): (batch,)
            movie_genres               (Tensor): (batch, genres_len)
            movie_tags                 (Tensor): (batch, tags_len)
            movie_genome_tags          (Tensor): (batch, genome_tags_len)
            years                      (Tensor): (batch,)
            target_movieId             (Tensor): (batch,)
        """
        # ── Rating-weighted avg pool over watch history ───────────────────────
        history_embs   = self.item_embedding_lookup(user_watch_history)
        pad_mask       = (user_watch_history != self.pad_idx).float().unsqueeze(-1)
        rating_weights = user_watch_history_ratings.unsqueeze(-1) * pad_mask
        weight_sum     = rating_weights.abs().sum(dim=1).clamp(min=1e-6)
        history_emb    = (history_embs * rating_weights).sum(dim=1) / weight_sum

        # ── User towers ───────────────────────────────────────────────────────
        genre_emb = self.user_genre_tower(user_genre_contexts)
        ts_emb    = self.timestamp_embedding_tower(self.timestamp_embedding_lookup(timestamps))

        if self.use_user_tag_tower:
            tag_emb       = self.user_tag_tower(user_tag_contexts)
            user_combined = torch.cat([history_emb, genre_emb, tag_emb, ts_emb], dim=1)
        else:
            user_combined = torch.cat([history_emb, genre_emb, ts_emb], dim=1)

        # ── Item towers ───────────────────────────────────────────────────────
        item_genre_emb = self.item_genre_tower(movie_genres)
        item_emb       = self.item_embedding_tower(self.item_embedding_lookup(target_movieId))
        year_emb       = self.year_embedding_tower(self.year_embedding_lookup(years))

        item_parts = [item_genre_emb]
        if self.use_item_tag_tower:
            item_parts.append(self.item_tag_tower(movie_tags))
        if self.use_item_genome_tag_tower:
            item_parts.append(self.item_genome_tag_tower(movie_genome_tags))
        item_parts += [item_emb, year_emb]
        item_combined = torch.cat(item_parts, dim=1)

        # ── Dot product prediction ────────────────────────────────────────────
        return torch.einsum('ij, ij -> i', user_combined, item_combined)
