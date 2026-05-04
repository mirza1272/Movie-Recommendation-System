"""CSP Module: filters movies based on user constraints."""

def apply_csp(df, genre=None, year_min=1920, year_max=2023,
              rating_min=0.0, duration_min=60, duration_max=240):
    """
    Constraint Satisfaction Problem filtering.
    Each constraint must be satisfied for a movie to pass.
    """
    filtered = df.copy()

    # Constraint 1: Genre match
    if genre and genre != "Any":
        filtered = filtered[filtered["Genre"].str.contains(genre, case=False, na=False)]

    # Constraint 2: Year range
    filtered = filtered[
        (filtered["Released_Year"] >= year_min) &
        (filtered["Released_Year"] <= year_max)
    ]

    # Constraint 3: Minimum IMDB rating
    filtered = filtered[filtered["IMDB_Rating"] >= rating_min]

    # Constraint 4: Duration range
    filtered = filtered[
        (filtered["Runtime_min"] >= duration_min) &
        (filtered["Runtime_min"] <= duration_max)
    ]

    return filtered.reset_index(drop=True)
