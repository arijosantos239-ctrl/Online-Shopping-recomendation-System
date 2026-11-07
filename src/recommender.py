import pandas as pd
from pathlib import Path

INPUT_FILE = "data/interactions.csv"
OUTPUT_FILE = "data/product_recommendations.csv"

EVENT_WEIGHTS = {
    "view": 1,
    "cart": 3,
    "purchase": 5,
}

def main():
    df = pd.read_csv(INPUT_FILE)

    df["score"] = df["event_type"].map(EVENT_WEIGHTS).fillna(0)

    user_product = (
        df.groupby(["user_id", "product_id"], as_index=False)
        .agg(score=("score", "sum"))
    )

    user_product["rank"] = (
        user_product.groupby("user_id")["score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    recommendations = user_product[
        user_product["rank"] <= 5
    ].sort_values(["user_id", "rank"])

    Path("data").mkdir(exist_ok=True)
    recommendations.to_csv(OUTPUT_FILE, index=False)

    print("Modelu rekomendasaun inisiál remata!")
    print("Total utilizadór:", recommendations["user_id"].nunique())
    print("Total rekomendasaun:", len(recommendations))
    print("Arquivo:", OUTPUT_FILE)
    print(recommendations.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
