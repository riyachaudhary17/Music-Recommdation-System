import os
import json
import requests
from typing import Optional, Dict, Any, List
from pathlib import Path
import re

TMDB_API_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p"


def _get_api_key(env_var: str = "TMDB_API_KEY") -> str:
    key = os.getenv(env_var)
    if not key:
        raise EnvironmentError(f"TMDb API key not found. Set the {env_var} environment variable.")
    return key


def _safe_get(url: str, params: dict = None, timeout: int = 5) -> Optional[requests.Response]:
    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r
    except requests.exceptions.HTTPError:
        # For 4xx/5xx including 404/429, return None
        return None
    except requests.exceptions.RequestException:
        # Network error, timeout, DNS, etc.
        return None


# Caching helpers -----------------------------------------------------------------

def _cache_dir() -> Path:
    p = Path(os.getenv("TMDB_CACHE_DIR", "data/tmdb_cache"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def _slugify(text: str) -> str:
    # Keep alphanumeric and underscore
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:200]


def _cache_read_search(title: str) -> Optional[Dict[str, Any]]:
    cd = _cache_dir()
    key = _slugify(title)
    p = cd / f"search_{key}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _cache_write_search(title: str, data: Dict[str, Any]):
    cd = _cache_dir()
    key = _slugify(title)
    p = cd / f"search_{key}.json"
    try:
        p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def _cache_read_details(tmdb_id: int) -> Optional[Dict[str, Any]]:
    cd = _cache_dir()
    p = cd / f"details_{tmdb_id}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _cache_write_details(tmdb_id: int, data: Dict[str, Any]):
    cd = _cache_dir()
    p = cd / f"details_{tmdb_id}.json"
    try:
        p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


# API helpers ---------------------------------------------------------------------

def search_movie(title: str, api_key_env: str = "TMDB_API_KEY") -> Optional[Dict[str, Any]]:
    """Search TMDb for a movie by title. Uses cache first. Returns the first result dict or None."""
    # Try cache
    cached = _cache_read_search(title)
    if cached:
        return cached

    key = _get_api_key(api_key_env)
    url = f"{TMDB_API_URL}/search/movie"
    params = {"api_key": key, "query": title}
    r = _safe_get(url, params=params)
    if not r:
        return None
    data = r.json()
    results = data.get("results") or []
    if not results:
        return None
    first = results[0]
    # write to cache
    _cache_write_search(title, first)
    return first


def get_movie_details(tmdb_id: int, api_key_env: str = "TMDB_API_KEY") -> Optional[Dict[str, Any]]:
    """Get movie details and credits for a TMDb movie id. Uses cache first. Returns dict or None."""
    cached = _cache_read_details(tmdb_id)
    if cached:
        return cached

    key = _get_api_key(api_key_env)
    url = f"{TMDB_API_URL}/movie/{tmdb_id}"
    params = {"api_key": key, "append_to_response": "credits"}
    r = _safe_get(url, params=params)
    if not r:
        return None
    details = r.json()
    _cache_write_details(tmdb_id, details)
    return details


def poster_url(path: Optional[str], size: str = "w500") -> Optional[str]:
    if not path:
        return None
    return f"{TMDB_IMAGE_BASE}/{size}{path}"


def extract_metadata_from_details(details: Dict[str, Any], cast_limit: int = 5) -> Dict[str, Any]:
    """Extract recommended fields from TMDb movie details JSON."""
    if not details:
        return {}
    overview = details.get("overview")
    genres = [g.get("name") for g in details.get("genres", []) if g.get("name")] if details.get("genres") else []
    poster = poster_url(details.get("poster_path"))
    release_date = details.get("release_date")
    rating = details.get("vote_average")

    director = None
    cast_list: List[str] = []
    credits = details.get("credits") or {}
    for crew in credits.get("crew", []) if credits.get("crew") else []:
        if crew.get("job") == "Director":
            director = crew.get("name")
            break
    for c in (credits.get("cast") or [])[:cast_limit]:
        name = c.get("name")
        if name:
            cast_list.append(name)

    return {
        "overview": overview,
        "genres": genres,
        "poster_url": poster,
        "release_date": release_date,
        "rating": rating,
        "director": director,
        "cast": cast_list,
    }


def enrich_movie_by_title(title: str, api_key_env: str = "TMDB_API_KEY") -> Dict[str, Any]:
    """Search by title and return enriched metadata. Returns dict with tmdb_id (if found) and extracted fields; on failure returns empty dict.

    This function handles network errors and missing API key by raising EnvironmentError for missing key and returning an empty dict for other failures.
    """
    try:
        res = search_movie(title, api_key_env=api_key_env)
    except EnvironmentError:
        raise
    if not res:
        return {}
    tmdb_id = res.get("id")
    details = get_movie_details(tmdb_id, api_key_env=api_key_env)
    if not details:
        return {"tmdb_id": tmdb_id}
    meta = extract_metadata_from_details(details)
    meta["tmdb_id"] = tmdb_id
    meta["tmdb_title"] = details.get("title")
    # Add sentiment of the overview if available
    try:
        from movie_recs.nlp.sentiment import analyze_sentiment, sentiment_label

        overview = meta.get("overview")
        if overview:
            scores = analyze_sentiment(overview)
            meta["sentiment"] = {"scores": scores, "label": sentiment_label(scores.get("compound", 0.0))}
        else:
            meta["sentiment"] = None
    except Exception:
        # If sentiment module or analysis fails, leave it out
        meta["sentiment"] = None
    return meta


__all__ = ["enrich_movie_by_title", "search_movie", "get_movie_details", "extract_metadata_from_details"]
