from pathlib import Path
import pandas as pd
from typing import Tuple


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def preprocess_movielens_1m(input_dir: str, out_dir: str = "data/processed", fmt: str = "csv") -> Tuple[Path, Path, Path]:
    """Read MovieLens 1M raw .dat files and write cleaned CSV or Parquet files.

    Parameters
    - input_dir: path to extracted ml-1m folder containing movies.dat, ratings.dat, users.dat
    - out_dir: directory to write processed files
    - fmt: 'csv' or 'parquet'

    Returns paths to written (movies_path, ratings_path, users_path)
    """
    inp = Path(input_dir)
    out = Path(out_dir)
    _ensure_dir(out)

    movies_path = inp / "movies.dat"
    ratings_path = inp / "ratings.dat"
    users_path = inp / "users.dat"

    if not movies_path.exists() or not ratings_path.exists() or not users_path.exists():
        raise FileNotFoundError(f"Expected movies.dat, ratings.dat, users.dat in {input_dir}")

    # MovieLens 1M uses '::' as separator; files contain Latin-1 encoded characters
    movies = pd.read_csv(movies_path, sep="::", engine="python", names=["movieId", "title", "genres"], encoding="latin-1")
    ratings = pd.read_csv(ratings_path, sep="::", engine="python", names=["userId", "movieId", "rating", "timestamp"], encoding="latin-1")
    users = pd.read_csv(users_path, sep="::", engine="python", names=["userId", "gender", "age", "occupation", "zip_code"], encoding="latin-1")

    # Basic cleaning/type-casting
    movies["movieId"] = movies["movieId"].astype(int)
    ratings["userId"] = ratings["userId"].astype(int)
    ratings["movieId"] = ratings["movieId"].astype(int)
    ratings["rating"] = ratings["rating"].astype(float)
    users["userId"] = users["userId"].astype(int)

    # Output file paths
    if fmt == "csv":
        movies_out = out / "movies.csv"
        ratings_out = out / "ratings.csv"
        users_out = out / "users.csv"

        movies.to_csv(movies_out, index=False)
        ratings.to_csv(ratings_out, index=False)
        users.to_csv(users_out, index=False)
    elif fmt == "parquet":
        movies_out = out / "movies.parquet"
        ratings_out = out / "ratings.parquet"
        users_out = out / "users.parquet"

        movies.to_parquet(movies_out, index=False)
        ratings.to_parquet(ratings_out, index=False)
        users.to_parquet(users_out, index=False)
    else:
        raise ValueError("fmt must be 'csv' or 'parquet'")

    return movies_out, ratings_out, users_out


def load_processed(out_dir: str = "data/processed", fmt: str = "csv") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load processed movies, ratings, users files as DataFrames."""
    out = Path(out_dir)
    if fmt == "csv":
        movies = pd.read_csv(out / "movies.csv")
        ratings = pd.read_csv(out / "ratings.csv")
        users = pd.read_csv(out / "users.csv")
    elif fmt == "parquet":
        movies = pd.read_parquet(out / "movies.parquet")
        ratings = pd.read_parquet(out / "ratings.parquet")
        users = pd.read_parquet(out / "users.parquet")
    else:
        raise ValueError("fmt must be 'csv' or 'parquet'")

    return movies, ratings, users


__all__ = ["preprocess_movielens_1m", "load_processed"]
