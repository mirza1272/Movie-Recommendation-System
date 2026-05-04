"""ML Module: K-Means Clustering + ANN (PyTorch) for rating prediction."""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

MODEL_PATH = "models/ann_model.pt"
SCALER_PATH = "models/scaler.pkl"
KMEANS_PATH = "models/kmeans.pkl"
os.makedirs("models", exist_ok=True)

# ---------- ANN Architecture ----------
class RatingPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)


# ---------- Training ----------
def train_ann(df):
    features = df[["Released_Year", "Runtime_min", "Meta_score"]].dropna()
    targets = df.loc[features.index, "IMDB_Rating"]

    scaler = StandardScaler()
    X = scaler.fit_transform(features.values).astype(np.float32)
    y = targets.values.astype(np.float32).reshape(-1, 1)

    X_t = torch.tensor(X)
    y_t = torch.tensor(y)

    model = RatingPredictor()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        pred = model(X_t)
        loss = criterion(pred, y_t)
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    return model, scaler


def train_kmeans(df, n_clusters=5):
    features = df[["Released_Year", "Runtime_min", "Meta_score", "IMDB_Rating"]].dropna()
    scaler = StandardScaler()
    X = scaler.fit_transform(features.values)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X)
    joblib.dump(kmeans, KMEANS_PATH)
    return kmeans


# ---------- Inference ----------
def load_ann():
    model = RatingPredictor()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def predict_ratings(df, model, scaler):
    features = df[["Released_Year", "Runtime_min", "Meta_score"]].fillna(df[["Released_Year", "Runtime_min", "Meta_score"]].median())
    X = scaler.transform(features.values).astype(np.float32)
    with torch.no_grad():
        preds = model(torch.tensor(X)).numpy().flatten()
    # Clip to realistic range
    preds = np.clip(preds, 1.0, 10.0)
    return preds


def get_cluster_labels(df, kmeans):
    features = df[["Released_Year", "Runtime_min", "Meta_score", "IMDB_Rating"]].fillna(
        df[["Released_Year", "Runtime_min", "Meta_score", "IMDB_Rating"]].median()
    )
    scaler = StandardScaler()
    X = scaler.fit_transform(features.values)
    return kmeans.predict(X)


def is_trained():
    return os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)
