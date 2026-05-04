import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils.helpers import load_data, get_all_genres
from modules.csp import apply_csp
from modules.heuristic import compute_heuristic
from modules.search import run_search
from modules.ml_model import (
    train_ann, train_kmeans, load_ann, predict_ratings,
    get_cluster_labels, is_trained, KMEANS_PATH
)
import joblib
import os
st.set_page_config(
    page_title="Hybrid AI Movie Recommender",
    page_icon="MR",
    layout="wide"
)
st.markdown("""
<style>
    :root {
        --bg: #07111f;
        --panel: #111d30;
        --panel-border: #f2b134;
        --text-main: #f6f7fb;
        --text-sub: #c1c9d6;
        --accent: #f2b134;
        --accent-2: #d88f16;
    }
    .stApp {
        background: var(--bg);
        color: var(--text-main);
    }
    section[data-testid="stSidebar"] {
        background: var(--panel);
        border-right: 1px solid var(--panel-border);
    }
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span {
        color: var(--text-main) !important;
    }
    .main-title { font-size: 2.2rem; font-weight: 700; color: var(--accent); }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        line-height: 1.15;
        color: var(--text-main);
        margin-bottom: 0.35rem;
    }
    .hero-subtitle {
        font-size: 1.02rem;
        color: var(--text-sub);
        max-width: 900px;
        margin-bottom: 1.2rem;
    }
    .hero-panel {
        background: var(--panel);
        border: 1px solid var(--panel-border);
        border-radius: 20px;
        padding: 1.4rem 1.5rem;
        margin-bottom: 1.1rem;
        box-shadow: 0 18px 60px rgba(0, 0, 0, 0.28);
    }
    .sub-title  { font-size: 1rem; color: #888; margin-bottom: 1.5rem; }
    .section-label {
        font-size: 0.9rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--accent-2);
        margin-bottom: 0.4rem;
        font-weight: 700;
    }
    .metric-card, .movie-card, .info-card {
        background: var(--panel);
        border: 1px solid var(--panel-border);
        border-radius: 18px;
        backdrop-filter: blur(12px);
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.22);
    }
    .metric-card {
        padding: 1rem 1.1rem;
        min-height: 108px;
    }
    .metric-label {
        color: var(--text-sub);
        font-size: 0.9rem;
        margin-bottom: 0.2rem;
    }
    .metric-value {
        color: var(--text-main);
        font-size: 1.55rem;
        font-weight: 800;
    }
    .metric-note {
        color: var(--text-sub);
        font-size: 0.82rem;
        margin-top: 0.25rem;
    }
    .movie-card {
        padding: 18px 18px 16px;
        margin-bottom: 14px;
        border-left: 5px solid var(--accent);
    }
    .badge {
        display: inline-block; background: var(--accent); color: #07111f;
        border-radius: 6px; padding: 2px 8px; font-size: 0.8rem;
        margin-right: 4px;
    }
    .badge.soft {
        background: rgba(255, 255, 255, 0.08);
        color: var(--text-main);
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .score-box {
        background: #16213e; border-radius: 8px; padding: 10px;
        text-align: center;
    }
    .recommendation-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
        gap: 12px;
        margin-bottom: 0.5rem;
    }
    .info-card {
        padding: 1rem 1.1rem;
    }
    .info-title {
        font-size: 0.9rem;
        font-weight: 700;
        color: var(--accent-2);
        margin-bottom: 0.35rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .info-value {
        color: var(--text-main);
        font-size: 1.25rem;
        font-weight: 800;
    }
    .info-caption {
        color: var(--text-sub);
        font-size: 0.85rem;
        margin-top: 0.2rem;
    }
    .stButton > button {
        background: var(--panel);
        color: var(--accent);
        border: 1px solid var(--accent);
        border-radius: 16px;
        font-weight: 700;
        width: 100%;
        padding: 0.82rem 1rem;
        box-shadow: 0 10px 24px rgba(0, 0, 0, 0.24);
        transition: transform 0.18s ease, box-shadow 0.18s ease, background 0.18s ease, border-color 0.18s ease;
    }
    .stButton > button:hover {
        background: #15233a;
        color: var(--accent);
        border: 1px solid var(--accent-2);
        transform: translateY(-1px);
        box-shadow: 0 12px 28px rgba(0, 0, 0, 0.30);
    }
    .stButton > button:active {
        transform: translateY(0);
        box-shadow: 0 8px 18px rgba(0, 0, 0, 0.22);
    }
</style>
""", unsafe_allow_html=True)
@st.cache_data
def get_data():
    return load_data()
df = get_data()
all_genres = get_all_genres(df)
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/6/69/IMDB_Logo_2016.svg", width=120)
    st.markdown("## Filters")
    genre = st.selectbox("Genre", ["Any"] + all_genres)
    year_range = st.slider("Year Range", 1920, 2023, (1990, 2023))
    rating_min = st.slider("Minimum IMDB Rating", 1.0, 9.5, 7.0, step=0.1)
    duration_range = st.slider("Duration (min)", 60, 300, (90, 180))
    algorithm = st.selectbox("Search Algorithm", ["A*", "BFS", "DFS"])
    top_n = st.slider("Top N Results", 3, 10, 5)
    st.markdown("---")
    if st.button("Train AI Model"):
        with st.spinner("Training ANN and K-Means..."):
            train_ann(df)
            train_kmeans(df)
        st.success("Model trained and saved successfully.")
    recommend_btn = st.button("Recommend Movies")
st.markdown(
    '<div class="hero-panel">'
    '<div class="section-label">Hybrid recommendation dashboard</div>'
    '<div class="hero-title">Hybrid AI Movie Recommender</div>'
    '<div class="hero-subtitle">Combine simple constraints, search strategies, heuristic scoring, K-Means clustering, and a small ANN to produce clear movie recommendations with explanations.</div>'
    '</div>',
    unsafe_allow_html=True
)
hero_col1, hero_col2, hero_col3 = st.columns(3)
with hero_col1:
    st.markdown(
        f'<div class="info-card"><div class="info-title">Dataset</div><div class="info-value">{len(df)}</div><div class="info-caption">Movies loaded from the local CSV</div></div>',
        unsafe_allow_html=True
    )
with hero_col2:
    st.markdown(
        f'<div class="info-card"><div class="info-title">Genres</div><div class="info-value">{len(all_genres)}</div><div class="info-caption">Unique genre combinations available</div></div>',
        unsafe_allow_html=True
    )
with hero_col3:
    st.markdown(
        f'<div class="info-card"><div class="info-title">Rating average</div><div class="info-value">{df["IMDB_Rating"].mean():.2f}</div><div class="info-caption">Average IMDB score across the dataset</div></div>',
        unsafe_allow_html=True
    )
if recommend_btn:
    with st.spinner("Running AI pipeline..."):
        csp_result = apply_csp(
            df,
            genre=genre,
            year_min=year_range[0],
            year_max=year_range[1],
            rating_min=rating_min,
            duration_min=duration_range[0],
            duration_max=duration_range[1]
        )
        if csp_result.empty:
            st.warning("No movies matched your constraints. Try relaxing the filters.")
            st.stop()
        preferred_year = (year_range[0] + year_range[1]) // 2
        scored = compute_heuristic(csp_result, genre=genre, preferred_year=preferred_year)
        search_result = run_search(scored, algorithm=algorithm, top_n=min(50, len(scored)))
        if is_trained():
            model, scaler = load_ann()
            predicted_ratings = predict_ratings(search_result, model, scaler)
            search_result = search_result.copy()
            search_result["predicted_rating"] = predicted_ratings
        else:
            search_result = search_result.copy()
            search_result["predicted_rating"] = search_result["IMDB_Rating"]
        if os.path.exists(KMEANS_PATH):
            kmeans = joblib.load(KMEANS_PATH)
            search_result["cluster"] = get_cluster_labels(search_result, kmeans)
        else:
            search_result["cluster"] = 0
        search_result["final_score"] = (
            search_result["heuristic_score"] * 0.6 +
            (search_result["predicted_rating"] / 10.0) * 0.4
        )
        final = search_result.sort_values("final_score", ascending=False).head(top_n)
    st.markdown(f"### Top {top_n} Recommendations")
    st.caption(f"CSP filtered: **{len(csp_result)}** movies -> Search ({algorithm}): **{len(search_result)}** -> Final: **{top_n}**")
    for i, (_, row) in enumerate(final.iterrows(), 1):
        with st.container():
            st.markdown(f"""
            <div class="movie-card">
                <h4>#{i} &nbsp; {row['Series_Title']} ({int(row['Released_Year'])})</h4>
                <span class="badge">IMDB {row['IMDB_Rating']}</span>
                <span class="badge soft">Predicted {row['predicted_rating']:.1f}</span>
                <span class="badge soft">Genre {row['Genre'][:30]}</span>
                <span class="badge soft">Runtime {int(row['Runtime_min'])} min</span>
                <span class="badge soft">Cluster {row['cluster']}</span>
                <br><br>
                <small>{row.get('Overview', '')[:200]}...</small>
                <br><br>
                <b>Match Score: {row['final_score']:.3f}</b> &nbsp;|&nbsp;
                <b>Algorithm: {algorithm}</b> &nbsp;|&nbsp;
                <b>Heuristic: {row['heuristic_score']:.3f}</b>
                <br>
                <small style="color:#aaa;">
                Recommended because: IMDB {row['IMDB_Rating']}/10,
                {'genre matches your preference, ' if genre != 'Any' else ''}
                year {int(row['Released_Year'])} fits range,
                runtime {int(row['Runtime_min'])} min fits preference.
                </small>
            </div>
            """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### Visual Analysis")
    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.bar(
            final, x="Series_Title", y=["IMDB_Rating", "predicted_rating"],
            barmode="group", title="IMDB vs Predicted Rating",
            labels={"value": "Rating", "variable": "Type"},
            color_discrete_map={"IMDB_Rating": "#f2b134", "predicted_rating": "#07111f"}
        )
        fig1.update_xaxes(tickangle=30)
        st.plotly_chart(fig1, width="stretch")
    with col2:
        fig2 = px.scatter(
            final, x="Released_Year", y="IMDB_Rating",
            size="final_score", color="cluster",
            hover_name="Series_Title",
            title="Year vs Rating (bubble = match score, color = cluster)"
        )
        st.plotly_chart(fig2, width="stretch")
    fig3 = px.bar(
        final.sort_values("final_score"),
        x="final_score", y="Series_Title",
        orientation="h", title="Final Match Scores",
        color="final_score", color_continuous_scale="Reds"
    )
    st.plotly_chart(fig3, width="stretch")
    with st.expander("Show Full Data Table"):
        cols = ["Series_Title", "Released_Year", "Genre", "IMDB_Rating",
                "predicted_rating", "Runtime_min", "heuristic_score", "final_score", "cluster"]
        st.dataframe(final[cols].reset_index(drop=True), width="stretch")
else:
    st.info("Set your preferences in the sidebar and click Recommend Movies.\n\n"
            "Tip: Click Train AI Model first to enable ANN predictions.")
    st.markdown("### Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Movies", len(df))
    col2.metric("Avg IMDB Rating", f"{df['IMDB_Rating'].mean():.2f}")
    col3.metric("Year Range", f"{df['Released_Year'].min()}-{df['Released_Year'].max()}")
    col4.metric("Genres", len(all_genres))

    fig = px.histogram(df, x="IMDB_Rating", nbins=30,
                       title="Distribution of IMDB Ratings",
                       color_discrete_sequence=["#f2b134"])
    st.plotly_chart(fig, width="stretch")
