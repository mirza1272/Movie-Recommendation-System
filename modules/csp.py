def apply_csp(df, genre=None, year_min=1920, year_max=2023,
              rating_min=0.0, duration_min=60, duration_max=240):
    filtered = df.copy()
    if genre and genre != "Any":
        filtered = filtered[filtered["Genre"].str.contains(genre, case=False, na=False)]
    filtered = filtered[
        (filtered["Released_Year"] >= year_min) &
        (filtered["Released_Year"] <= year_max)
    ]
    filtered = filtered[filtered["IMDB_Rating"] >= rating_min]
    filtered = filtered[
        (filtered["Runtime_min"] >= duration_min) &
        (filtered["Runtime_min"] <= duration_max)
    ]
    return filtered.reset_index(drop=True)
