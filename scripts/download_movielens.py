#!/usr/bin/env python
"""Simple script wrapper to download MovieLens 1M using the package API."""
from movie_recs.data.download import download_movielens_1m


if __name__ == "__main__":
    download_movielens_1m("data/raw")
    print("Done")
