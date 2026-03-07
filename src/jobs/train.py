from __future__ import annotations

import argparse
from typing import Dict, List

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.pipeline.processor import build_feature_stages, clean_and_label
from src.utils.config import load_config
from src.utils.io_utils import write_json_payload
from src.utils.logging import get_logger
from src.utils.spark_utils import create_spark_session

LOGGER = get_logger(__name__)


def add_class_weights(df: DataFrame) -> DataFrame:
    """Add a class_weight column to reduce imbalance impact."""
    counts = {int(row["label"]): row["count"] for row in df.groupBy("label").count().collect()}
    class_count = len(counts)
    total = sum(counts.values())

    if class_count == 0 or total == 0:
        return df.withColumn("class_weight", F.lit(1.0))

    weights = {label: float(total) / (class_count * count) for label, count in counts.items()}

    weight_expr = F.lit(1.0)
    for label, weight in weights.items():
        weight_expr = F.when(F.col("label") == F.lit(label), F.lit(weight)).otherwise(weight_expr)

    return df.withColumn("class_weight", weight_expr)


def compute_metrics(predictions: DataFrame) -> Dict[str, object]:
    """Compute key classification metrics and confusion matrix."""
    evaluator_accuracy = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy"
    )
    evaluator_weighted_f1 = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="f1"
    )

    accuracy = float(evaluator_accuracy.evaluate(predictions))
    weighted_f1 = float(evaluator_weighted_f1.evaluate(predictions))

    pred_labels_rdd = predictions.select("prediction", "label").rdd.map(
        lambda row: (float(row["prediction"]), float(row["label"]))
    )
    metrics = MulticlassMetrics(pred_labels_rdd)

    labels = [0.0, 1.0, 2.0]
    macro_f1 = float(sum(metrics.fMeasure(label) for label in labels) / len(labels))

    return {
        "accuracy": accuracy,
        "weighted_f1": weighted_f1,
        "macro_f1": macro_f1,
        "confusion_matrix": metrics.confusionMatrix().toArray().tolist(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train sentiment model with Spark ML")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    spark = create_spark_session(config)
    try:
        paths = config["paths"]
        preprocessing_cfg = config["preprocessing"]
        model_cfg = config["model"]
        runtime_cfg = config.get("runtime", {})

        raw_df = spark.read.parquet(paths["raw_parquet"])
        full_curated_df = clean_and_label(raw_df, min_words=int(preprocessing_cfg["min_words"]))

        # Always persist the full curated dataset so score can run on all available rows.
        full_curated_df.write.mode(runtime_cfg.get("write_mode", "overwrite")).parquet(paths["curated_parquet"])

        sample_fraction = float(runtime_cfg.get("sample_fraction", 1.0))
        seed = int(runtime_cfg.get("seed", 42))
        train_source_df = full_curated_df
        if 0.0 < sample_fraction < 1.0:
            LOGGER.info("Sampling curated data with fraction=%s", sample_fraction)
            train_source_df = full_curated_df.sample(
                withReplacement=False,
                fraction=sample_fraction,
                seed=seed,
            )

        train_df, _, test_df = train_source_df.randomSplit([0.70, 0.15, 0.15], seed=seed)
        train_df = add_class_weights(train_df)

        feature_stages = build_feature_stages(
            vocab_size=int(preprocessing_cfg["vocab_size"]),
            min_df=int(preprocessing_cfg["min_df"]),
        )

        lr = LogisticRegression(
            featuresCol="features",
            labelCol="label",
            weightCol="class_weight",
            maxIter=int(model_cfg["max_iter"]),
        )

        pipeline = Pipeline(stages=[*feature_stages, lr])

        reg_params: List[float] = [float(v) for v in model_cfg.get("reg_params", [0.1])]
        param_grid = ParamGridBuilder().addGrid(lr.regParam, reg_params).build()

        evaluator = MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="f1",
        )

        cv = CrossValidator(
            estimator=pipeline,
            estimatorParamMaps=param_grid,
            evaluator=evaluator,
            numFolds=int(model_cfg.get("cv_folds", 3)),
            seed=seed,
            parallelism=int(model_cfg.get("cv_parallelism", 2)),
        )

        cv_model = cv.fit(train_df)
        best_model = cv_model.bestModel
        predictions = best_model.transform(test_df)

        metrics = compute_metrics(predictions)
        metrics["selected_reg_param"] = float(best_model.stages[-1]._java_obj.getRegParam())
        metrics["train_rows"] = int(train_df.count())
        metrics["test_rows"] = int(test_df.count())

        best_model.write().overwrite().save(paths["model_dir"])
        write_json_payload(spark, paths["metrics_json"], metrics)

        LOGGER.info("Training completed. Model saved at: %s", paths["model_dir"])
        LOGGER.info("Metrics saved at: %s", paths["metrics_json"])
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
