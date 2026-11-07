from pathlib import Path
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.ml.recommendation import ALS

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "data" / "interactions.csv"
OUTPUT_FILE = BASE_DIR / "data" / "spark_recommendations.csv"
MODEL_DIR = BASE_DIR / "models" / "als_recommender"

EVENT_WEIGHTS = {"view": 1.0, "cart": 3.0, "purchase": 5.0}


def main():
    spark = (
        SparkSession.builder
        .appName("OnlineShoppingALS")
        .master("local[*]")
        .config("spark.driver.memory", "2g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    try:
        print("1. Lee dadus interasaun...")
        df = spark.read.csv(str(INPUT_FILE), header=True, inferSchema=True)

        required = {"user_id", "product_id", "event_type"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Koluna falta: {sorted(missing)}")

        weights = F.create_map(
            *[x for pair in EVENT_WEIGHTS.items()
              for x in (F.lit(pair[0]), F.lit(pair[1]))]
        )

        interactions = (
            df.withColumn("event_type", F.lower(F.col("event_type")))
            .withColumn("rating", weights[F.col("event_type")])
            .filter(F.col("rating").isNotNull())
            .groupBy("user_id", "product_id")
            .agg(F.sum("rating").alias("rating"))
        )

        user_window = Window.orderBy("user_id")
        product_window = Window.orderBy("product_id")

        user_index = (
            interactions.select("user_id").distinct()
            .withColumn("user_idx", F.row_number().over(user_window) - 1)
        )
        product_index = (
            interactions.select("product_id").distinct()
            .withColumn("product_idx", F.row_number().over(product_window) - 1)
        )

        training = (
            interactions.join(user_index, "user_id")
            .join(product_index, "product_id")
            .select(
                F.col("user_idx").cast("int"),
                F.col("product_idx").cast("int"),
                F.col("rating").cast("float")
            )
        )

        print("2. Treina modelu Spark MLlib ALS...")
        model = ALS(
            userCol="user_idx",
            itemCol="product_idx",
            ratingCol="rating",

            coldStartStrategy="drop",
            rank=10,
            maxIter=5,
            regParam=0.1,
            seed=42
        ).fit(training)

        print("3. Gera rekomendasaun...")
        predictions = model.transform(training)

        result = (
            predictions.join(user_index, "user_idx")
            .join(product_index, "product_idx")
            .select("user_id", "product_id", "predicted_score")
        )

        rank_window = Window.partitionBy("user_id").orderBy(
            F.desc("predicted_score")
        )
        result = (
            result.withColumn("rank", F.row_number().over(rank_window))
            .filter(F.col("rank") <= 5)
            .orderBy("user_id", "rank")
        )

        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        result.toPandas().to_csv(OUTPUT_FILE, index=False)

        print("4. Rai modelu...")
        try:
            MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
            model.write().overwrite().save(str(MODEL_DIR))
            print("Modelu rai iha:", MODEL_DIR)
        except Exception as e:
            print("Avisu: la konsege rai modelu ALS:", e)

        print("Spark MLlib ALS remata.")
        print("Arquivo rekomendasaun:", OUTPUT_FILE)
        result.show(10, truncate=False)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
