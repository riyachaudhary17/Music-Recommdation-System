import argparse
import sys
from movie_recs.data.download import download_movielens_1m
from movie_recs.data.preprocess import preprocess_movielens_1m


def main(argv=None):
    parser = argparse.ArgumentParser(description="Movie Recommendation System CLI")
    sub = parser.add_subparsers(dest="cmd")

    dl = sub.add_parser("download_dataset", help="Download MovieLens 1M dataset")
    dl.add_argument("--dest", default="data/raw", help="Destination folder (relative to cwd)")

    tr = sub.add_parser("train_baseline", help="Train baseline models or run baseline recommendations")
    tr.add_argument("--processed", default="data/processed", help="Path to processed data folder")
    tr.add_argument("--user", type=int, default=1, help="User id to produce recommendations for")
    tr.add_argument("--top_n", type=int, default=10, help="Number of recommendations to show")

    svd_p = sub.add_parser("svd", help="Train SVD model and produce recommendations for a user")
    svd_p.add_argument("--processed", default="data/processed", help="Path to processed data folder")
    svd_p.add_argument("--user", type=int, default=1, help="User id to recommend for")
    svd_p.add_argument("--n_factors", type=int, default=20, help="Number of SVD latent factors")
    svd_p.add_argument("--top_n", type=int, default=10, help="Number of recommendations to show")

    content_p = sub.add_parser("content", help="Run content-based recommendations for a user")
    content_p.add_argument("--processed", default="data/processed", help="Path to processed data folder")
    content_p.add_argument("--user", type=int, default=1, help="User id to recommend for")
    content_p.add_argument("--top_n", type=int, default=10, help="Number of recommendations to show")
    content_p.add_argument("--fields", nargs="+", default=["title", "genres"], help="Movie fields to use for content (space-separated)")

    hybrid_p = sub.add_parser("hybrid", help="Run hybrid recommendations combining CF and content")
    hybrid_p.add_argument("--processed", default="data/processed", help="Path to processed data folder")
    hybrid_p.add_argument("--user", type=int, default=1, help="User id to recommend for")
    hybrid_p.add_argument("--top_n", type=int, default=10, help="Number of recommendations to show")
    hybrid_p.add_argument("--alpha", type=float, default=0.5, help="Weight for CF (0..1), content weight = 1-alpha")
    hybrid_p.add_argument("--cf_method", choices=["svd", "item"], default="svd", help="CF method to use: svd or item-based")
    hybrid_p.add_argument("--n_factors", type=int, default=20, help="Number of SVD latent factors (if using svd)")
    hybrid_p.add_argument("--candidate_pool", type=int, default=100, help="Number of candidates to consider from each method before merging")

    persist_p = sub.add_parser("persist", help="Train/build models and persist them to disk")
    persist_p.add_argument("--processed", default="data/processed", help="Path to processed data folder")
    persist_p.add_argument("--out", default="models", help="Output folder/prefix to write model artifacts")
    persist_p.add_argument("--n_factors", type=int, default=20, help="Number of SVD latent factors to train and persist")
    persist_p.add_argument("--fields", nargs="+", default=["title", "genres"], help="Fields to use for content TF-IDF")

    load_p = sub.add_parser("load_models", help="Load persisted models and produce recommendations without retraining")
    load_p.add_argument("--models", default="models", help="Models folder where artifacts were saved")
    load_p.add_argument("--method", choices=["svd", "content", "hybrid"], default="svd", help="Which persisted model to use")
    load_p.add_argument("--user", type=int, default=1, help="User id to recommend for")
    load_p.add_argument("--top_n", type=int, default=10, help="Number of recommendations to show")
    load_p.add_argument("--alpha", type=float, default=0.5, help="Hybrid alpha (only for hybrid method)")

    serve_p = sub.add_parser("serve", help="Run a local FastAPI server to serve recommendations")
    serve_p.add_argument("--host", default="127.0.0.1", help="Host to bind")
    serve_p.add_argument("--port", type=int, default=8000, help="Port to listen on")
    serve_p.add_argument("--models", default="models", help="Models folder where artifacts were saved")
    serve_p.add_argument("--processed", default="data/processed", help="Processed data folder")
    serve_p.add_argument("--reload", action="store_true", help="Enable uvicorn reload (development)")

    enrich_p = sub.add_parser("enrich", help="Enrich recommendations with TMDb metadata (requires TMDB_API_KEY env var)")
    enrich_p.add_argument("--models", default="models", help="Models folder where artifacts were saved")
    enrich_p.add_argument("--processed", default="data/processed", help="Processed data folder")
    enrich_p.add_argument("--method", choices=["svd", "content", "hybrid"], default="hybrid", help="Which model to use to produce base recommendations")
    enrich_p.add_argument("--user", type=int, default=1, help="User id to recommend for")
    enrich_p.add_argument("--top_n", type=int, default=10, help="Number of recommendations to enrich")
    enrich_p.add_argument("--out", default=None, help="Optional output path (JSON) to save enriched results")
    enrich_p.add_argument("--tmdb_env", default="TMDB_API_KEY", help="Environment variable name containing TMDb API key")

    ev = sub.add_parser("evaluate", help="Evaluate models (placeholder)")
    ev.add_argument("--model", default="baseline", help="Model name to evaluate")

    pr = sub.add_parser("preprocess", help="Preprocess MovieLens 1M into CSV/Parquet")
    pr.add_argument("--input", default="data/raw/ml-1m", help="Path to extracted ml-1m folder")
    pr.add_argument("--out", default="data/processed", help="Output folder for processed data")
    pr.add_argument("--format", choices=["csv", "parquet"], default="csv", help="Output format")

    args = parser.parse_args(argv)

    if args.cmd == "download_dataset":
        dest = args.dest
        print(f"Downloading MovieLens 1M to {dest} ...")
        download_movielens_1m(dest)
        print("Download complete.")
    elif args.cmd == "train_baseline":
        # Run popularity and item-based baseline and show recommendations for the requested user
        from movie_recs.data.preprocess import load_processed
        from movie_recs.models.baselines import top_n_popular, item_based_recommendations

        print(f"Loading processed data from {args.processed} ...")
        movies, ratings, users = load_processed(args.processed, fmt="csv")
        print("Computing popularity baseline...")
        pop = top_n_popular(ratings, movies, n=args.top_n)
        print("Top popular movies:")
        for _, r in pop.iterrows():
            print(f"{int(r.movieId)} - {r.title} (count={int(r.count)}, avg={r.avg_rating:.2f})")

        print(f"\nComputing item-based recommendations for user {args.user} ...")
        recs = item_based_recommendations(args.user, ratings, movies, top_n=args.top_n)
        if recs.empty:
            print("No recommendations (user not found or no data).")
        else:
            print("Recommendations:")
            for _, r in recs.iterrows():
                print(f"{int(r.movieId)} - {r.title} (score={r.score:.4f})")

    elif args.cmd == "svd":
        from movie_recs.data.preprocess import load_processed
        from movie_recs.models.svd import train_svd, recommend_svd

        print(f"Loading processed data from {args.processed} ...")
        movies, ratings, users = load_processed(args.processed, fmt="csv")
        print(f"Training SVD with n_factors={args.n_factors} ...")
        u, s, vt, user_ids, item_ids, gmean = train_svd(ratings, n_factors=args.n_factors)
        print(f"Trained SVD: users={len(user_ids)}, items={len(item_ids)}, global_mean={float(gmean):.3f}")
        print(f"Computing recommendations for user {args.user} ...")
        recs = recommend_svd(args.user, u, s, vt, user_ids, item_ids, ratings, movies, top_n=args.top_n)
        if recs.empty:
            print("No recommendations (user not found or no data).")
        else:
            print("SVD Recommendations:")
            for _, r in recs.iterrows():
                print(f"{int(r.movieId)} - {r.title} (score={r.score:.4f})")

    elif args.cmd == "content":
        from movie_recs.data.preprocess import load_processed
        from movie_recs.models.content import recommend_content, build_item_tfidf

        print(f"Loading processed data from {args.processed} ...")
        movies, ratings, users = load_processed(args.processed, fmt="csv")
        print(f"Building content item vectors using fields={args.fields} ...")
        vec, item_vectors, item_ids = build_item_tfidf(movies, fields=args.fields)
        print(f"Computing content-based recommendations for user {args.user} ...")
        recs = recommend_content(args.user, ratings, movies, top_n=args.top_n, vectorizer=vec, item_vectors=item_vectors, fields=args.fields)
        if recs.empty:
            print("No recommendations (user not found or no data).")
        else:
            print("Content-based Recommendations:")
            for _, r in recs.iterrows():
                print(f"{int(r.movieId)} - {r.title} (score={r.score:.4f})")

    elif args.cmd == "hybrid":
        from movie_recs.data.preprocess import load_processed
        from movie_recs.models.hybrid import hybrid_recommend

        print(f"Loading processed data from {args.processed} ...")
        movies, ratings, users = load_processed(args.processed, fmt="csv")
        print(f"Computing hybrid recommendations for user {args.user} (alpha={args.alpha}) ...")
        recs = hybrid_recommend(
            args.user,
            ratings,
            movies,
            top_n=args.top_n,
            alpha=args.alpha,
            cf_method=args.cf_method,
            n_factors=args.n_factors,
            candidate_pool=args.candidate_pool,
        )
        if recs.empty:
            print("No recommendations (user not found or no data).")
        else:
            print("Hybrid Recommendations:")
            for _, r in recs.iterrows():
                print(f"{int(r.movieId)} - {r.title} (score={r.score:.4f})")

    elif args.cmd == "persist":
        from movie_recs.data.preprocess import load_processed
        from movie_recs.models.svd import train_svd
        from movie_recs.models.content import build_item_tfidf
        from movie_recs.models.persistence import save_svd, save_tfidf
        import numpy as np
        from pathlib import Path

        print(f"Loading processed data from {args.processed} ...")
        movies, ratings, users = load_processed(args.processed, fmt="csv")
        out_prefix = Path(args.out)
        out_prefix.mkdir(parents=True, exist_ok=True)

        print(f"Training SVD with n_factors={args.n_factors} ...")
        u, s, vt, user_ids, item_ids, gmean = train_svd(ratings, n_factors=args.n_factors)
        svd_path = out_prefix / "svd_factors.npz"
        print(f"Saving SVD to {svd_path} ...")
        save_svd(str(svd_path), u, s, vt, user_ids, item_ids, gmean)

        print(f"Building TF-IDF item vectors using fields={args.fields} ...")
        vec, item_vectors, item_ids = build_item_tfidf(movies, fields=args.fields)
        tf_prefix = out_prefix / "tfidf"
        print(f"Saving TF-IDF artifacts with prefix {tf_prefix} ...")
        save_tfidf(str(tf_prefix), vec, item_vectors)

        print(f"Persisted models to {out_prefix}")

    elif args.cmd == "load_models":
        from movie_recs.data.preprocess import load_processed
        from movie_recs.models.loader import load_models_and_recommend

        print(f"Loading processed data from {args.processed} ...")
        movies, ratings, users = load_processed(args.processed, fmt="csv")
        print(f"Loading persisted models from {args.models} and producing recommendations (method={args.method}) ...")
        try:
            recs = load_models_and_recommend(args.models, args.method, args.user, ratings, movies, top_n=args.top_n, alpha=args.alpha)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return
        if recs.empty:
            print("No recommendations (user not found or no data).")
        else:
            print("Loaded-model Recommendations:")
            for _, r in recs.iterrows():
                print(f"{int(r.movieId)} - {r.title} (score={r.score:.4f})")

    elif args.cmd == "serve":
        # Start a FastAPI server using uvicorn
        try:
            import uvicorn
        except Exception:
            print("uvicorn not installed. Install fastapi and uvicorn to run the server: pip install fastapi uvicorn[standard]")
            return

        host = args.host
        port = args.port
        reload = args.reload
        models = args.models
        processed = args.processed

        # Set default environment/config by writing simple wrapper that passes args via env isn't necessary — server reads processed and models from request body
        print(f"Starting server on {host}:{port} (models={models}, processed={processed})")
        uvicorn.run("movie_recs.api.server:app", host=host, port=port, reload=reload)

    elif args.cmd == "enrich":
        from movie_recs.data.preprocess import load_processed
        from movie_recs.models.loader import load_models_and_recommend
        from movie_recs.external.tmdb import enrich_movie_by_title
        import json
        from pathlib import Path

        print(f"Loading processed data from {args.processed} ...")
        movies, ratings, users = load_processed(args.processed, fmt="csv")
        print(f"Producing base recommendations using method={args.method} ...")
        try:
            recs = load_models_and_recommend(args.models, args.method, args.user, ratings, movies, top_n=args.top_n, alpha=args.alpha)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return
        if recs.empty:
            print("No recommendations to enrich.")
            return

        print(f"Enriching {len(recs)} recommendations using TMDb (env var: {args.tmdb_env}) ...")
        enriched = []
        for _, row in recs.iterrows():
            title = row.title
            base = {"movieId": int(row.movieId), "title": title, "score": float(row.score)}
            try:
                meta = enrich_movie_by_title(title, api_key_env=args.tmdb_env)
            except EnvironmentError as e:
                print(f"TMDb API key error: {e}")
                return
            if not meta:
                base["tmdb"] = None
            else:
                base["tmdb"] = meta
            enriched.append(base)
            print(f"- {title}: {'found' if base['tmdb'] else 'not found'}")

        if args.out:
            outp = Path(args.out)
            outp.parent.mkdir(parents=True, exist_ok=True)
            with open(outp, "w", encoding="utf-8") as f:
                json.dump(enriched, f, ensure_ascii=False, indent=2)
            print(f"Wrote enriched recommendations to {outp}")
        else:
            import json
            print(json.dumps(enriched, ensure_ascii=False, indent=2))

    elif args.cmd == "serve":
        # Start a FastAPI server using uvicorn
        try:
            import uvicorn
        except Exception:
            print("uvicorn not installed. Install fastapi and uvicorn to run the server: pip install fastapi uvicorn[standard]")
            return

        host = args.host
        port = args.port
        reload = args.reload
        models = args.models
        processed = args.processed

        # Set default environment/config by writing simple wrapper that passes args via env isn't necessary — server reads processed and models from request body
        print(f"Starting server on {host}:{port} (models={models}, processed={processed})")
        uvicorn.run("movie_recs.api.server:app", host=host, port=port, reload=reload)

    elif args.cmd == "evaluate":
        print("evaluate is a placeholder. Implement evaluation routines in movie_recs.eval.")
    elif args.cmd == "preprocess":
        print(f"Preprocessing MovieLens from {args.input} to {args.out} as {args.format} ...")
        movies_out, ratings_out, users_out = preprocess_movielens_1m(args.input, args.out, fmt=args.format)
        print(f"Wrote: {movies_out}, {ratings_out}, {users_out}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main(sys.argv[1:])
