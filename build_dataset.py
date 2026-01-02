import argparse
import pandas as pd
from tqdm import tqdm
import random
from pathlib import Path


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









class GenresDataBuilder:
    def __init__(
        self,
        processed_data_file='movie_dataset/processed_dataset.csv',
        movie_meta_data_path='movie_dataset/movie.csv',
        seq_len=10,
    ):
        self.seq_len = seq_len

        df = pd.read_csv(processed_data_file)
        self.movie_meta = pd.read_csv(movie_meta_data_path)

        genres = set()
        for row in df.genres:
            for g in row.split('|'):
                for sub in g.split(','):
                    genres.add(sub)

        self.genres_meta = pd.DataFrame(
            enumerate(sorted(genres), start=1),
            columns=['genre_id', 'genre']
        )

        out_path = Path(processed_data_file).parent / 'genres.csv'
        self.genres_meta.to_csv(out_path, index=False)

        df = df.apply(self.encode_genres, axis=1)
        df.to_csv(processed_data_file, index=False)

    def encode_genres(self, row):
        seq_genres = []
        for block in row.genres.split('|'):
            choices = block.split(',')
            genre = random.choice(choices)
            gid = self.genres_meta[self.genres_meta.genre == genre].genre_id.values[0]
            seq_genres.append(int(gid))

        while len(seq_genres) < self.seq_len:
            seq_genres.append(seq_genres[-1])

        row['seq_genres'] = seq_genres

        target_genres = self.movie_meta[self.movie_meta.movieId == row.target].genres.values[0]
        target = random.choice(target_genres.split('|'))
        row['target_genre'] = int(
            self.genres_meta[self.genres_meta.genre == target].genre_id.values[0]
        )

        return row

