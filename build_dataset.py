import argparse
import random
from pathlib import Path

import pandas as pd
from tqdm import tqdm


class MoviesDataBuilder:
    """
    Builds a sequence dataset for next-item prediction from:
      - rating.csv  (userId, movieId, rating, timestamp, ...)
      - movie.csv   (movieId, title, genres, ...)

    Output columns:
      - seq        : list[int] fixed-length padded sequence of movieIds
      - len_seq    : true length before padding
      - target     : next movieId to predict
      - genres     : a string describing user history genres (excluding target movie genres)
    """

    def __init__(
        self,
        rating_path="movie_dataset/rating.csv",
        meta_data_path="movie_dataset/movie.csv",
        number_of_users=None,
        output_path="movie_dataset/processed_dataset.csv",
        min_rating=4.0,
        history_len=11,          # total interactions used per user = history_len (last item is target)
        min_history=2,           # minimum interactions required to create at least 1 feature + 1 target
        dedupe_user_history=False,
        seed=42,
    ):
        self.rating_path = rating_path
        self.meta_data_path = meta_data_path
        self.number_of_users = number_of_users
        self.output_path = output_path

        self.min_rating = float(min_rating)
        self.history_len = int(history_len)
        self.min_history = int(min_history)
        self.dedupe_user_history = bool(dedupe_user_history)

        random.seed(seed)

        # Load CSVs
        self.df_rating = pd.read_csv(self.rating_path)
        self.df_meta = pd.read_csv(self.meta_data_path)

        # Merge metadata onto ratings so each rating row has genres/title/etc.
        self.df_rating = pd.merge(self.df_rating, self.df_meta, on="movieId", how="inner")

        # Filter by rating threshold (implicit "liked" items)
        self.df_rating = self.df_rating[self.df_rating.rating >= self.min_rating].copy()

        # Cache unique users after filtering
        self.users = self.df_rating.userId.unique()

        # Will be filled by build_user()
        self.users_pref = []
        self.movies_genres = []

    def build_user(self):
        """
        Builds per-user sequences:
          - users_pref[i] contains a list of movieIds in time order (last item is target)
          - movies_genres[i] contains a genre-history string excluding the target movie
        """
        self.users_pref = []
        self.movies_genres = []

        # Optionally limit number of users
        users = self.users
        if self.number_of_users is not None:
            users = users[: int(self.number_of_users)]

        for user in tqdm(users, desc="Building user preferences"):
            # Sort user interactions chronologically; take the last `history_len`
            user_df = self.df_rating[self.df_rating.userId == user].sort_values("timestamp")
            user_df = user_df.tail(self.history_len)

            if len(user_df) < self.min_history:
                # Not enough interactions to form (feature, target)
                continue

            # Optionally dedupe repeated movies in the sequence (keeps last occurrence order)
            if self.dedupe_user_history:
                # Keep order but remove duplicates by scanning from end
                seen = set()
                movie_ids = []
                genres_col = []
                for _, row in user_df[::-1].iterrows():
                    mid = int(row.movieId)
                    if mid in seen:
                        continue
                    seen.add(mid)
                    movie_ids.append(mid)
                    genres_col.append(row.genres)
                # Reverse back to chronological order
                movie_ids = list(reversed(movie_ids))
                genres_col = list(reversed(genres_col))
            else:
                movie_ids = user_df.movieId.astype(int).to_list()
                genres_col = user_df.genres.astype(str).to_list()

            if len(movie_ids) < self.min_history:
                continue

            self.users_pref.append(movie_ids)

            # Prevent target leakage: exclude genres of the final (target) movie
            history_genres = genres_col[:-1]

            # Convert "Action|Comedy" into "Action,Comedy" per movie, then join across history
            history_genres = [",".join(g.split("|")) for g in history_genres]
            history_genres = "|".join(history_genres)

            self.movies_genres.append(history_genres)

    def build_dataset(self, seq_len=10, pad_strategy="repeat_last", pad_token=0):
        """
        Creates fixed-length sequences for training.

        Parameters
        ----------
        seq_len : int
            Length of the model input sequence.
        pad_strategy : str
            'repeat_last' repeats the last real item.
            'pad_token' pads using `pad_token`.
        pad_token : int
            Used only when pad_strategy='pad_token'.
        """
        seq_len = int(seq_len)
        pad_token = int(pad_token)

        features = []
        target = []
        len_seq = []
        meta_data = []

        for idx, movies_set in tqdm(list(enumerate(self.users_pref)), desc="Building dataset"):
            length = len(movies_set)
            if length < 2:
                continue

            meta_data.append(self.movies_genres[idx])

            # Feature sequence excludes the last item; last item is the target
            feature_vector = movies_set[:-1]
            target.append(int(movies_set[-1]))

            true_len = len(feature_vector)
            len_seq.append(true_len)

            # Truncate if longer than seq_len (keep most recent seq_len items)
            if true_len > seq_len:
                feature_vector = feature_vector[-seq_len:]
                true_len = seq_len
                len_seq[-1] = true_len

            # Pad if shorter than seq_len
            if len(feature_vector) < seq_len:
                diff = seq_len - len(feature_vector)
                if pad_strategy == "repeat_last" and len(feature_vector) > 0:
                    feature_vector.extend([feature_vector[-1]] * diff)
                elif pad_strategy == "pad_token":
                    feature_vector.extend([pad_token] * diff)
                else:
                    raise ValueError("pad_strategy must be 'repeat_last' or 'pad_token'")

            # Safety check: every row has exact seq_len
            assert len(feature_vector) == seq_len, f"All rows must contain {seq_len} items"

            features.append([int(x) for x in feature_vector])

        self.dataset = pd.DataFrame(
            {
                "seq": features,
                "len_seq": len_seq,
                "target": target,
                "genres": meta_data,
            }
        )

        # Write output
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        self.dataset.to_csv(self.output_path, index=False)


class GenresDataBuilder:
    """
    Adds genre-based features to an existing processed dataset.

    Creates:
      - genres.csv : genre_id -> genre mapping
    Adds columns to processed dataset:
      - seq_genres    : list[int] per timestep (one genre id per history movie)
      - target_genre  : int genre id for the target movie
      - (optional) multi_hot features if genre_strategy='multi_hot'
    """

    def __init__(
        self,
        processed_data_file="movie_dataset/processed_dataset.csv",
        movie_meta_data_path="movie_dataset/movie.csv",
        seq_len=10,
        genre_strategy="random",   # random | first | multi_hot
        seed=42,
    ):
        self.processed_data_file = processed_data_file
        self.movie_meta_data_path = movie_meta_data_path
        self.seq_len = int(seq_len)
        self.genre_strategy = genre_strategy

        random.seed(seed)

        # Load processed dataset and movie metadata
        df = pd.read_csv(processed_data_file)
        self.movie_meta_data = pd.read_csv(movie_meta_data_path)

        # Build a unique set of genres from the history-genre strings
        genres_set = set()
        for genres in df["genres"].astype(str).to_list():
            # history string looks like: "Action,Comedy|Drama,Romance|..."
            for sub_genre in genres.split("|"):
                for genre in sub_genre.split(","):
                    if genre:
                        genres_set.add(genre)

        genres_list = sorted(list(genres_set))
        genres_id = list(zip(range(1, len(genres_list) + 1), genres_list))
        self.genres_meta_data = pd.DataFrame(genres_id, columns=["genre_id", "genre"])

        # Write genres.csv next to processed_data_file
        out_dir = Path(processed_data_file).parent
        genres_output = out_dir / "genres.csv"
        self.genres_meta_data.to_csv(genres_output, index=False)

        # Apply encoding row-wise
        df = df.apply(self.encode_genres, axis=1)

        # Save back to the same file (or adjust if you want a new file)
        df.to_csv(processed_data_file, index=False)

    def _pick_genre(self, pipe_separated_genres: str) -> str:
        """
        Pick a single genre from 'Action|Comedy|Drama' according to genre_strategy.
        """
        parts = [g for g in pipe_separated_genres.split("|") if g]
        if not parts:
            return ""

        if self.genre_strategy == "first":
            return parts[0]
        # default: random
        return random.choice(parts)

    def encode_genres(self, row: pd.Series) -> pd.Series:
        """
        Encodes:
          - seq_genres: one genre id per timestep (based on the 'genres' history string)
          - target_genre: one genre id from the target movie's genre list
          - optionally multi-hot vectors
        """
        history = str(row["genres"])

        # history is like: "Action,Comedy|Drama|Romance,Sci-Fi|..."
        # Each timestep may contain multiple candidate genres separated by commas.
        seq_genres = []
        for sub in history.split("|"):
            candidates = [g for g in sub.split(",") if g]
            if not candidates:
                continue

            if self.genre_strategy == "first":
                chosen = candidates[0]
            else:  # random
                chosen = random.choice(candidates)

            genre_id = int(self.genres_meta_data[self.genres_meta_data.genre == chosen].genre_id.values[0])
            seq_genres.append(genre_id)

        # Pad/truncate seq_genres to seq_len
        if len(seq_genres) > self.seq_len:
            seq_genres = seq_genres[-self.seq_len:]
        while len(seq_genres) < self.seq_len and len(seq_genres) > 0:
            seq_genres.append(seq_genres[-1])

        row["seq_genres"] = seq_genres

        # Target genre from the actual target movie metadata
        target_movie_id = row["target"]
        target_genres_str = self.movie_meta_data[self.movie_meta_data.movieId == target_movie_id].genres.values[0]
        chosen_target_genre = self._pick_genre(str(target_genres_str))

        if chosen_target_genre:
            target_genre_id = int(
                self.genres_meta_data[self.genres_meta_data.genre == chosen_target_genre].genre_id.values[0]
            )
        else:
            target_genre_id = 0

        row["target_genre"] = target_genre_id

        # Optional: multi-hot encoding of target genres (useful for multi-label tasks)
        if self.genre_strategy == "multi_hot":
            all_genres = self.genres_meta_data["genre"].tolist()
            genre_to_idx = {g: i for i, g in enumerate(all_genres)}

            # Multi-hot for target movie
            vec = [0] * len(all_genres)
            for g in str(target_genres_str).split("|"):
                g = g.strip()
                if g in genre_to_idx:
                    vec[genre_to_idx[g]] = 1
            row["target_genres_multi_hot"] = vec

        return row


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a movie sequence dataset and optionally add genre encodings."
    )

    # Build base dataset
    parser.add_argument("--user_movie", action="store_true", default=False, help="Generate base user->movie dataset.")
    parser.add_argument("--rating_file", default="movie_dataset/rating.csv")
    parser.add_argument("--metadata_file", default="movie_dataset/movie.csv")
    parser.add_argument("-o", "--output", default="movie_dataset/processed_dataset.csv")
    parser.add_argument("--users", type=int, default=None, help="Limit number of users processed.")
    parser.add_argument("--min_rating", type=float, default=4.0, help="Keep ratings >= this threshold.")
    parser.add_argument("--history_len", type=int, default=11, help="Last N interactions per user (includes target).")
    parser.add_argument("--min_history", type=int, default=2, help="Minimum interactions needed per user.")
    parser.add_argument("--seq_len", type=int, default=10, help="Fixed length of the input sequence.")
    parser.add_argument("--pad_strategy", choices=["repeat_last", "pad_token"], default="repeat_last")
    parser.add_argument("--pad_token", type=int, default=0)
    parser.add_argument("--dedupe_user_history", action="store_true", help="Remove repeated movies per user sequence.")
    parser.add_argument("--seed", type=int, default=42)

    # Add genres
    parser.add_argument("--user_genres", action="store_true", default=False, help="Add genre encodings.")
    parser.add_argument(
        "--genre_strategy",
        choices=["random", "first", "multi_hot"],
        default="random",
        help="How to pick/encode genres.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.user_movie:
        data_builder = MoviesDataBuilder(
            rating_path=args.rating_file,
            meta_data_path=args.metadata_file,
            number_of_users=args.users,
            output_path=args.output,
            min_rating=args.min_rating,
            history_len=args.history_len,
            min_history=args.min_history,
            dedupe_user_history=args.dedupe_user_history,
            seed=args.seed,
        )
        data_builder.build_user()
        data_builder.build_dataset(
            seq_len=args.seq_len,
            pad_strategy=args.pad_strategy,
            pad_token=args.pad_token,
        )
        print(f"Base dataset created: {args.output}", flush=True)

    if args.user_genres:
        GenresDataBuilder(
            processed_data_file=args.output,
            movie_meta_data_path=args.metadata_file,
            seq_len=args.seq_len,
            genre_strategy=args.genre_strategy,
            seed=args.seed,
        )
        print(f"Genre features added: {args.output}", flush=True)
