import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path

from movie_recs.models.loader import load_models_and_recommend
from movie_recs.data.preprocess import load_processed
from movie_recs.models.persistence import load_svd, load_tfidf
from movie_recs.models.svd import recommend_svd
from movie_recs.models.content import recommend_content
from movie_recs.nlp.sentiment import analyze_sentiment, sentiment_label

app = FastAPI(title="Movie Recommendation System API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8001",
        "http://127.0.0.1:8001",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RecommendRequest(BaseModel):
    user_id: int
    method: Optional[str] = "svd"  # 'svd', 'content', or 'hybrid'
    top_n: Optional[int] = 10
    alpha: Optional[float] = 0.5
    models_dir: Optional[str] = None
    processed_dir: Optional[str] = None


@app.on_event("startup")
def load_artifacts_on_startup():
    """Attempt to load processed data and persisted models into app.state for fast serving.

    Uses environment variables MOVIE_RECS_MODELS_DIR and MOVIE_RECS_PROCESSED_DIR if set,
    otherwise defaults to 'models' and 'data/processed'. Non-fatal: server will still start
    even if artifacts are missing; requests will return errors accordingly.
    """
    models_dir = os.getenv("MOVIE_RECS_MODELS_DIR", "models")
    processed_dir = os.getenv("MOVIE_RECS_PROCESSED_DIR", "data/processed")

    app.state.models_dir = models_dir
    app.state.processed_dir = processed_dir

    # Load processed data
    try:
        movies, ratings, users = load_processed(processed_dir, fmt="csv")
        app.state.movies = movies
        app.state.ratings = ratings
        app.state.users = users
        print(f"Loaded processed data from {processed_dir}: movies={len(movies)}, ratings={len(ratings)}, users={len(users)}")
    except Exception as e:
        app.state.movies = None
        app.state.ratings = None
        app.state.users = None
        print(f"Warning: Failed to load processed data from {processed_dir}: {e}")

    # Load SVD if available
    try:
        svd_path = Path(models_dir) / "svd_factors.npz"
        if svd_path.exists():
            u, s, vt, user_ids, item_ids, gmean = load_svd(str(svd_path))
            app.state.svd = dict(u=u, s=s, vt=vt, user_ids=user_ids, item_ids=item_ids, gmean=gmean)
            print(f"Loaded SVD factors from {svd_path}")
        else:
            app.state.svd = None
            print(f"No SVD file at {svd_path}")
    except Exception as e:
        app.state.svd = None
        print(f"Warning: Failed to load SVD factors: {e}")

    # Load TF-IDF artifacts if available
    try:
        tf_prefix = Path(models_dir) / "tfidf"
        vec_path = tf_prefix.with_name(tf_prefix.name + "_vectorizer.joblib")
        items_path = tf_prefix.with_name(tf_prefix.name + "_item_vectors.npz")
        if vec_path.exists() and items_path.exists():
            vec, item_vectors = load_tfidf(str(tf_prefix))
            app.state.tfidf = dict(vectorizer=vec, item_vectors=item_vectors)
            print(f"Loaded TF-IDF artifacts from {tf_prefix}")
        else:
            app.state.tfidf = None
            print(f"No TF-IDF artifacts at {tf_prefix}")
    except Exception as e:
        app.state.tfidf = None
        print(f"Warning: Failed to load TF-IDF artifacts: {e}")

    # Warm-up: precompute popular movies and cached recommendations for top users
    try:
        # Precompute popularity list (top 100)
        if app.state.ratings is not None:
            agg = app.state.ratings.groupby("movieId").rating.agg(["count", "mean"]).rename(columns={"mean": "avg_rating"})
            agg = agg.sort_values(["count", "avg_rating"], ascending=[False, False])
            popular = agg.reset_index().head(100)
            # join titles if movies available
            if app.state.movies is not None:
                popular = popular.merge(app.state.movies[["movieId", "title"]], on="movieId", how="left")
            app.state.popular = popular
            print(f"Warmup: computed top-{len(popular)} popular movies")
        else:
            app.state.popular = None

        # Cache recommendations for top active users for methods 'svd','content','hybrid'
        top_users_count = int(os.getenv("MOVIE_RECS_WARMUP_TOP_USERS", "100"))
        top_users_k = int(os.getenv("MOVIE_RECS_WARMUP_K", "10"))
        app.state.user_rec_cache = {"svd": {}, "content": {}, "hybrid": {}}
        if app.state.ratings is not None:
            user_counts = app.state.ratings.groupby("userId").size().sort_values(ascending=False)
            top_users = user_counts.head(top_users_count).index.tolist()
            print(f"Warmup: caching recommendations for top {len(top_users)} users")
            for uid in top_users:
                try:
                    # SVD
                    if app.state.svd is not None:
                        recs = recommend_svd(uid, app.state.svd["u"], app.state.svd["s"], app.state.svd["vt"], app.state.svd["user_ids"], app.state.svd["item_ids"], app.state.ratings, app.state.movies, top_n=top_users_k)
                        app.state.user_rec_cache["svd"][uid] = recs
                    # Content
                    if app.state.tfidf is not None:
                        recs = recommend_content(uid, app.state.ratings, app.state.movies, top_n=top_users_k, vectorizer=app.state.tfidf["vectorizer"], item_vectors=app.state.tfidf["item_vectors"])
                        app.state.user_rec_cache["content"][uid] = recs
                    # Hybrid: use loader for hybrid merging
                    try:
                        h = load_models_and_recommend(models_dir, "hybrid", uid, app.state.ratings, app.state.movies, top_n=top_users_k)
                        app.state.user_rec_cache["hybrid"][uid] = h
                    except Exception:
                        pass
                except Exception:
                    # continue on errors per-user
                    continue
        print("Warmup: user recommendation cache populated (partial)")
    except Exception as e:
        print(f"Warning: Warmup failed: {e}")


@app.get("/", tags=["root"])
async def root():
    return {"service": "movie-recs", "status": "ok"}


@app.post("/recommend", tags=["recommendations"] )
def recommend(req: RecommendRequest):
    # Use processed_dir from request if provided, else from app.state
    processed_dir = req.processed_dir or getattr(app.state, "processed_dir", None) or os.getenv("MOVIE_RECS_PROCESSED_DIR", "data/processed")
    models_dir = req.models_dir or getattr(app.state, "models_dir", None) or os.getenv("MOVIE_RECS_MODELS_DIR", "models")

    # Prefer cached data
    movies = getattr(app.state, "movies", None)
    ratings = getattr(app.state, "ratings", None)

    if movies is None or ratings is None:
        # Try loading from requested path
        try:
            movies, ratings, users = load_processed(processed_dir, fmt="csv")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Processed data not found at {processed_dir}. Run preprocess first. ({e})")

    # If client requested to use models_dir different from cached, attempt to load them on-demand
    svd_loaded = getattr(app.state, "svd", None)
    tf_loaded = getattr(app.state, "tfidf", None)
    if models_dir != getattr(app.state, "models_dir", None):
        # try load new models
        try:
            svd_path = Path(models_dir) / "svd_factors.npz"
            if svd_path.exists():
                u, s, vt, user_ids, item_ids, gmean = load_svd(str(svd_path))
                svd_loaded = dict(u=u, s=s, vt=vt, user_ids=user_ids, item_ids=item_ids, gmean=gmean)
            else:
                svd_loaded = None
            tf_prefix = Path(models_dir) / "tfidf"
            vec_path = tf_prefix.with_name(tf_prefix.name + "_vectorizer.joblib")
            items_path = tf_prefix.with_name(tf_prefix.name + "_item_vectors.npz")
            if vec_path.exists() and items_path.exists():
                vec, item_vectors = load_tfidf(str(tf_prefix))
                tf_loaded = dict(vectorizer=vec, item_vectors=item_vectors)
            else:
                tf_loaded = None
        except Exception as e:
            # ignore and fall back to any cached artifacts
            print(f"Warning: failed to load models from {models_dir}: {e}")

    # Produce recommendations using cached or on-demand artifacts
    try:
        if req.method == "svd":
            if svd_loaded is None:
                raise HTTPException(status_code=400, detail=f"SVD model not found in {models_dir}")
            recs = recommend_svd(req.user_id, svd_loaded["u"], svd_loaded["s"], svd_loaded["vt"], svd_loaded["user_ids"], svd_loaded["item_ids"], ratings, movies, top_n=req.top_n)
        elif req.method == "content":
            if tf_loaded is None:
                raise HTTPException(status_code=400, detail=f"TF-IDF artifacts not found in {models_dir}")
            recs = recommend_content(req.user_id, ratings, movies, top_n=req.top_n, vectorizer=tf_loaded["vectorizer"], item_vectors=tf_loaded["item_vectors"])
        elif req.method == "hybrid":
            # delegate to loader hybrid path for simplicity
            recs = load_models_and_recommend(models_dir, "hybrid", req.user_id, ratings, movies, top_n=req.top_n, alpha=req.alpha)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown method: {req.method}")
    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation error: {e}")

    # Convert to JSON-friendly structure; include sentiment for movie overviews if available in cached enriched metadata
    results = []
    for _, r in recs.iterrows():
        item = {"movieId": int(r.movieId), "title": str(r.title), "score": float(r.score)}
        # If we have TMDb enriched data cached (app.state.popular or user cache may include), try attach sentiment
        # Otherwise, attempt lightweight sentiment on title (not ideal)
        try:
            # If tmdb details cached in tmdb cache folder? Skip heavy lookups here
            overview = None
            # If movies df has a title and we have TF-IDF or other data, we avoid doing network calls here
            # As a simple integration, compute sentiment on the movie title as placeholder
            s = analyze_sentiment(item["title"])
            item["sentiment"] = {"scores": s, "label": sentiment_label(s.get("compound", 0.0))}
        except Exception:
            item["sentiment"] = None
        results.append(item)

    return {"user_id": req.user_id, "method": req.method, "results": results}


class SentimentRequest(BaseModel):
    text: str


@app.post("/sentiment", tags=["nlp"])
def sentiment(req: SentimentRequest):
    """Return sentiment scores and label for provided text."""
    try:
        scores = analyze_sentiment(req.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sentiment analysis failed: {e}")
    label = sentiment_label(scores.get("compound", 0.0))
    return {"text": req.text, "scores": scores, "label": label}


class ReloadRequest(BaseModel):
    models_dir: Optional[str] = None
    processed_dir: Optional[str] = None


@app.post("/reload", tags=["admin"])
def reload(req: ReloadRequest):
    """Reload persisted models and processed data and recompute warmup cache.

    Optional JSON body: {"models_dir": "models", "processed_dir": "data/processed"}
    """
    if req.models_dir:
        os.environ["MOVIE_RECS_MODELS_DIR"] = req.models_dir
    if req.processed_dir:
        os.environ["MOVIE_RECS_PROCESSED_DIR"] = req.processed_dir

    try:
        # Call the same startup loader to refresh app.state
        load_artifacts_on_startup()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reload failed: {e}")

    # Return summary of loaded artifacts
    return {
        "models_dir": app.state.models_dir,
        "processed_dir": app.state.processed_dir,
        "has_svd": bool(app.state.svd),
        "has_tfidf": bool(app.state.tfidf),
        "popular_count": len(app.state.popular) if getattr(app.state, "popular", None) is not None else 0,
        "cached_users_svd": len(app.state.user_rec_cache.get("svd", {})) if getattr(app.state, "user_rec_cache", None) is not None else 0,
        "cached_users_content": len(app.state.user_rec_cache.get("content", {})) if getattr(app.state, "user_rec_cache", None) is not None else 0,
        "cached_users_hybrid": len(app.state.user_rec_cache.get("hybrid", {})) if getattr(app.state, "user_rec_cache", None) is not None else 0,
    }


@app.get("/health", tags=["root"]) 
def health():
    return {"status": "ok"}
