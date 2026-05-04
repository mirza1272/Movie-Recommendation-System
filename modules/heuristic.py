def compute_heuristic(df, genre=None, preferred_year=2000):
    df = df.copy()
    rating_norm = (df["IMDB_Rating"] - df["IMDB_Rating"].min()) / \
                  (df["IMDB_Rating"].max() - df["IMDB_Rating"].min() + 1e-9)
    year_diff = (df["Released_Year"] - preferred_year).abs()
    year_score = 1 - (year_diff / (year_diff.max() + 1e-9))
    if genre and genre != "Any":
        genre_match = df["Genre"].str.contains(genre, case=False, na=False).astype(float)
    else:
        genre_match = 1.0
    df["heuristic_score"] = (rating_norm * 0.5) + (year_score * 0.2) + (genre_match * 0.3)
    return df
