from __future__ import annotations

import argparse

from pyspark.ml import PipelineModel
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.pipeline.processor import clean_and_label
from src.utils.config import load_config
from src.utils.logging import get_logger
from src.utils.spark_utils import create_spark_session

LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score reviews with trained Spark model")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config")
    return parser.parse_args()


def build_outputs(predictions: DataFrame) -> tuple[DataFrame, DataFrame, DataFrame]:
    """Build predictions and simple aggregate outputs."""
    pred = predictions.select(
        "review_id",
        "business_id",
        "date",
        "label",
        F.col("prediction").cast("int").alias("prediction"),
        F.col("probability").cast("string").alias("probability"),
    )

    daily_stats = pred.groupBy("date", "prediction").count()

    total = pred.count()
    class_dist = pred.groupBy("prediction").count().withColumn(
        "pct", F.col("count") / F.lit(total if total > 0 else 1)
    )

    return pred, daily_stats, class_dist


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    spark = create_spark_session(config)
    try:
        paths = config["paths"]
        preprocessing_cfg = config["preprocessing"]
        runtime_cfg = config.get("runtime", {})

        input_df = spark.read.parquet(paths["curated_parquet"])
        # Keep scoring resilient when curated data is absent or stale.
        if "text_clean" not in input_df.columns or "label" not in input_df.columns:
            input_df = clean_and_label(
                spark.read.parquet(paths["raw_parquet"]),
                min_words=int(preprocessing_cfg["min_words"]),
            )

        model = PipelineModel.load(paths["model_dir"])
        predictions = model.transform(input_df)

        pred_df, daily_df, class_df = build_outputs(predictions)

        write_mode = runtime_cfg.get("write_mode", "overwrite")
        pred_df.write.mode(write_mode).parquet(paths["predictions_dir"])
        daily_df.write.mode(write_mode).parquet(paths["daily_stats_dir"])
        class_df.write.mode(write_mode).parquet(paths["class_distribution_dir"])

        LOGGER.info("Predictions saved at: %s", paths["predictions_dir"])
        LOGGER.info("Daily stats saved at: %s", paths["daily_stats_dir"])
        LOGGER.info("Class distribution saved at: %s", paths["class_distribution_dir"])
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
