import pandas as pd
import re
DATA_PATH = "data/imdb_top_1000.csv"
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["Runtime_min"] = df["Runtime"].str.extract(r"(\d+)").astype(float)
    df["Released_Year"] = pd.to_numeric(df["Released_Year"], errors="coerce")
    df["Meta_score"] = df["Meta_score"].fillna(df["Meta_score"].median())
    df = df.dropna(subset=["Released_Year", "IMDB_Rating", "Runtime_min"])
    df["Released_Year"] = df["Released_Year"].astype(int)
    return df
def get_all_genres(df):
    genres = set()
    for g in df["Genre"].dropna():
        for part in g.split(","):
            genres.add(part.strip())
    return sorted(genres)
