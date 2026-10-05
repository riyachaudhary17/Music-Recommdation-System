import pandas as pd
import numpy as np


def train_test_split_by_user(ratings: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """Split ratings DataFrame into train and test by sampling each user's ratings.

    For each user, samples floor(test_size * n_ratings) ratings to test (at least 1 if possible).
    Returns (train_df, test_df).
    """
    rng = np.random.default_rng(seed)
    train_parts = []
    test_parts = []

    for uid, group in ratings.groupby("userId"):
        n = len(group)
        if n == 1:
            # Can't split, keep in train
            train_parts.append(group)
            continue
        n_test = max(1, int(np.floor(n * test_size)))
        idx = np.arange(n)
        test_idx = rng.choice(idx, size=n_test, replace=False)
        mask = np.zeros(n, dtype=bool)
        mask[test_idx] = True
        test_parts.append(group.iloc[mask])
        train_parts.append(group.iloc[~mask])

    train = pd.concat(train_parts).reset_index(drop=True)
    test = pd.concat(test_parts).reset_index(drop=True) if test_parts else pd.DataFrame(columns=ratings.columns)
    return train, test


__all__ = ["train_test_split_by_user"]
