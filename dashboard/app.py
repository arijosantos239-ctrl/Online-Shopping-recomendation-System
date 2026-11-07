
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(
    page_title="Online Shopping Recommendation System",
    page_icon="🛍️",
    layout="wide"
)

st.title("🛍️ Online Shopping Recommendation System")
st.write("Dashboard ba analiza atividade utilizadór no rekomendasaun produtu.")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "product_recommendations.csv"
INTERACTIONS_FILE = BASE_DIR / "data" / "interactions.csv"

if not DATA_FILE.exists():
    st.error("Arquivo product_recommendations.csv la hetan.")
    st.stop()

recommendations = pd.read_csv(DATA_FILE)

col1, col2, col3 = st.columns(3)
col1.metric("Total Utilizadór", f"{recommendations['user_id'].nunique():,}")
col2.metric("Total Rekomendasaun", f"{len(recommendations):,}")
col3.metric("Total Produtu Úniku", f"{recommendations['product_id'].nunique():,}")

st.divider()

st.subheader("🔎 Buka Rekomendasaun ba Utilizadór")

user_ids = sorted(recommendations["user_id"].unique())
selected_user = st.selectbox("Hili ID Utilizadór", user_ids)

user_recommendations = recommendations[
    recommendations["user_id"] == selected_user
].sort_values("rank")

st.write(f"Rekomendasaun ba utilizadór: **{selected_user}**")
st.dataframe(user_recommendations, use_container_width=True, hide_index=True)

st.divider()

st.subheader("📊 Distribuisaun Score Rekomendasaun")

score_counts = recommendations["score"].value_counts().sort_index()
fig, ax = plt.subplots()
score_counts.plot(kind="bar", ax=ax)
ax.set_xlabel("Score")
ax.set_ylabel("Total Rekomendasaun")
ax.set_title("Total Rekomendasaun tuir Score")
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.divider()

st.subheader("📁 Dadus Interasaun")

if INTERACTIONS_FILE.exists():
    interactions = pd.read_csv(INTERACTIONS_FILE)
    st.write(f"Total interasaun: **{len(interactions):,}**")

    if "event_type" in interactions.columns:
        event_counts = interactions["event_type"].value_counts()
        st.bar_chart(event_counts)

    st.dataframe(interactions.head(20), use_container_width=True)
else:
    st.warning("Arquivo interactions.csv la hetan.")

st.caption("Projeto Big Data Capstone 2026 — Online Shopping Recommendation System")
