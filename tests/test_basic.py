def test_package_importable():
    import movie_recs

    assert hasattr(movie_recs, "__version__")
