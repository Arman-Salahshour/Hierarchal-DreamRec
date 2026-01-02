import argparse
import pandas as pd
from tqdm import tqdm


class MoviesDataBuilder:
    def __init__(
        self,
        rating_path='movie_dataset/rating.csv',
        meta_data_path='movie_dataset/movie.csv',
        number_of_users=None,
        output_path='movie_dataset/processed_dataset.csv',
        min_rating=4.0,
        history_len=11,
        min_history=2,
    ):
        self.rating_path = rating_path
        self.meta_data_path = meta_data_path
        self.number_of_users = number_of_users
        self.output_path = output_path
        self.min_rating = min_rating
        self.history_len = history_len
        self.min_history = min_history

        self.df_rating = pd.read_csv(self.rating_path)
        self.df_meta = pd.read_csv(self.meta_data_path)

        self.df_rating = pd.merge(self.df_rating, self.df_meta, on='movieId')
        self.df_rating = self.df_rating[self.df_rating.rating >= self.min_rating]

        self.users = self.df_rating.userId.unique()

    def build_user(self):
        self.users_pref = []
        self.movies_genres = []

        users = self.users
        if self.number_of_users is not None:
            users = users[: int(self.number_of_users)]

        for user in tqdm(users, desc='Building user preferences'):
            user_df = (
                self.df_rating[self.df_rating.userId == user]
                .sort_values('timestamp')
                .tail(self.history_len)
            )

            if len(user_df) < self.min_history:
                continue

            movie_ids = user_df.movieId.tolist()
            self.users_pref.append(movie_ids)

            genres = user_df.genres.tolist()[:-1]
            genres = [",".join(g.split('|')) for g in genres]
            self.movies_genres.append("|".join(genres))

    def build_dataset(self, seq_len=10, pad_token=0):
        features, targets, lengths, meta = [], [], [], []

        for idx, movies in enumerate(self.users_pref):
            if len(movies) < 2:
                continue

            seq = movies[:-1]
            target = movies[-1]

            if len(seq) > seq_len:
                seq = seq[-seq_len:]

            lengths.append(len(seq))
            targets.append(target)
            meta.append(self.movies_genres[idx])

            while len(seq) < seq_len:
                seq.append(pad_token)

            features.append(seq)

        self.dataset = pd.DataFrame({
            'seq': features,
            'len_seq': lengths,
            'target': targets,
            'genres': meta
        })
        self.dataset.to_csv(self.output_path, index=False)
