import pandas as pd
from pathlib import Path

df = pd.read_csv("data/clickstream.csv")

df["event_time"] = pd.to_datetime(df["event_time"], utc=True)
df["event_type"] = df["event_type"].str.strip().str.lower()

Path("reports").mkdir(exist_ok=True)

summary = df.groupby("event_type").agg(
    total_events=("event_type", "size"),
    unique_users=("user_id", "nunique"),
    unique_products=("product_id", "nunique")
).reset_index()

summary.to_csv("reports/event_summary.csv", index=False)

interactions = df[df["event_type"].isin(["view", "cart", "purchase"])].copy()
interactions.to_csv("data/interactions.csv", index=False)

print("Dataset prepara ho susesu!")
print("Event summary: reports/event_summary.csv")
print("Interactions: data/interactions.csv")
print("Total interasaun:", len(interactions))
print(summary.to_string(index=False))
