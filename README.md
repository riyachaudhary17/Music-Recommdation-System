# Movie Recommendation System (scaffold)

This repository is a scaffold for a Movie Recommendation System Python package.

Usage (development):
- Create a virtualenv and install dependencies:

  python -m venv .venv
  .venv\Scripts\activate
  pip install -e .

- Download MovieLens 1M dataset:

  python -m movie_recs.cli download_dataset --dest data\raw

- Preprocess MovieLens data:

  python -m movie_recs.cli preprocess --input data\raw\ml-1m --out data\processed --format csv

- Persist trained models (SVD + TF-IDF):

  python -m movie_recs.cli persist --processed data\processed --out models --n_factors 20 --fields title genres

- Serve recommendations via local HTTP API (requires fastapi + uvicorn):

  # Install server deps if not already installed
  pip install fastapi uvicorn[standard]

  # Start server (defaults to models/ and data/processed)
  python -m movie_recs.cli serve --host 127.0.0.1 --port 8000 --models models --processed data\processed

  # Example request (using curl):
  curl -X POST "http://127.0.0.1:8000/recommend" -H "Content-Type: application/json" -d "{\"user_id\":1, \"method\":\"hybrid\", \"top_n\":10}"

Package entry point (after install):
  movie-recs <command> ...

Next steps:
- Implement models in movie_recs.models
- Add evaluation code in movie_recs.eval
- Add tests for each component
