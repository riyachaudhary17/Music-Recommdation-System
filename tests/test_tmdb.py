import os
import json
import pytest
from unittest.mock import patch, Mock
from movie_recs.external.tmdb import search_movie, get_movie_details, extract_metadata_from_details, enrich_movie_by_title


@patch("movie_recs.external.tmdb._safe_get")
def test_search_and_enrich(mock_safe_get):
    # Mock search response
    mock_search_resp = Mock()
    mock_search_resp.json.return_value = {"results": [{"id": 123, "title": "Toy Story"}]}
    # Mock details response
    mock_details_resp = Mock()
    mock_details_resp.json.return_value = {
        "id": 123,
        "title": "Toy Story",
        "overview": "A story about toys",
        "genres": [{"id": 1, "name": "Animation"}],
        "poster_path": "/poster.jpg",
        "release_date": "1995-11-22",
        "vote_average": 7.9,
        "credits": {"cast": [{"name": "Tom Hanks"}], "crew": [{"name": "John Lasseter", "job": "Director"}]},
    }

    # Configure safe_get to return search then details; repeat so subsequent calls still work
    # Configure safe_get to return search then details; repeat so subsequent calls still work
    mock_safe_get.side_effect = [mock_search_resp, mock_details_resp, mock_search_resp, mock_details_resp]

    # Ensure API key is present in env for function to proceed
    os.environ["TMDB_API_KEY"] = "fake-key"
    # Use a temporary cache dir for test
    import tempfile
    tmp = tempfile.TemporaryDirectory()
    os.environ["TMDB_CACHE_DIR"] = tmp.name

    res = search_movie("Toy Story")
    assert res and res.get("id") == 123

    # Second call should hit cache (safe_get will still be called as we've mocked but cache prevents network)
    details = get_movie_details(123)
    assert details and details.get("id") == 123

    meta = extract_metadata_from_details(details)
    assert meta["overview"] == "A story about toys"
    assert meta["poster_url"].endswith("/poster.jpg")
    assert meta["director"] == "John Lasseter"

    enriched = enrich_movie_by_title("Toy Story")
    assert enriched.get("tmdb_id") == 123
    assert enriched.get("director") == "John Lasseter"

    tmp.cleanup()


@patch("movie_recs.external.tmdb._safe_get")
def test_missing_api_key(mock_safe_get):
    # Ensure env var missing
    if "TMDB_API_KEY" in os.environ:
        del os.environ["TMDB_API_KEY"]
    with pytest.raises(EnvironmentError):
        search_movie("Any")
