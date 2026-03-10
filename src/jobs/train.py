from __future__ import annotations

import argparse
from itertools import product
from typing import Dict, List

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.pipeline.processor import build_feature_stages_with_options, clean_and_label
from src.utils.config import load_config
from src.utils.io_utils import write_json_payload
from src.utils.logging import get_logger
from src.utils.spark_utils import create_spark_session

LOGGER = get_logger(__name__)
LABELS = [0.0, 1.0, 2.0]


def add_class_weights(df: DataFrame, multipliers: Dict[int, float] | None = None) -> DataFrame:
    """Add a class_weight column to reduce imbalance impact."""
    counts = {int(row["label"]): row["count"] for row in df.groupBy("label").count().collect()}
    class_count = len(counts)
    total = sum(counts.values())

    if class_count == 0 or total == 0:
        return df.withColumn("class_weight", F.lit(1.0))

    weights = {label: float(total) / (class_count * count) for label, count in counts.items()}

    # Optional per-class multipliers let us bias learning toward hard classes.
    if multipliers:
        for label, multiplier in multipliers.items():
            if label in weights:
                weights[label] = weights[label] * float(multiplier)

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

    macro_f1 = float(sum(metrics.fMeasure(label) for label in LABELS) / len(LABELS))
    macro_precision = float(sum(metrics.precision(label) for label in LABELS) / len(LABELS))
    macro_recall = float(sum(metrics.recall(label) for label in LABELS) / len(LABELS))

    per_class_metrics = {
        str(int(label)): {
            "precision": float(metrics.precision(label)),
            "recall": float(metrics.recall(label)),
            "f1": float(metrics.fMeasure(label)),
        }
        for label in LABELS
    }

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "weighted_f1": weighted_f1,
        "macro_f1": macro_f1,
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": metrics.confusionMatrix().toArray().tolist(),
    }


def build_error_analysis(
    predictions: DataFrame,
    confusion_matrix: List[List[float]],
    top_k: int = 3,
    sample_misclassified: int = 12,
) -> Dict[str, object]:
    """Build lightweight error analysis without expensive extra shuffles.

    Top confusion pairs are derived from the confusion matrix already computed
    during metric evaluation. We only sample a small number of misclassified
    rows for qualitative inspection.
    """
    pair_counts: List[Dict[str, object]] = []
    total_misclassified = 0
    for i, row in enumerate(confusion_matrix):
        for j, value in enumerate(row):
            count = int(value)
            if i == j:
                continue
            total_misclassified += count
            pair_counts.append(
                {
                    "actual_label": i,
                    "predicted_label": j,
                    "count": count,
                }
            )

    top_confusions = sorted(pair_counts, key=lambda item: int(item["count"]), reverse=True)[:top_k]

    sample_rows = (
        predictions.select(
            F.col("review_id"),
            F.col("text_clean"),
            F.col("label").cast("int").alias("actual_label"),
            F.col("prediction").cast("int").alias("predicted_label"),
        )
        .filter(F.col("actual_label") != F.col("predicted_label"))
        .limit(sample_misclassified)
        .collect()
    )
    sample_errors = [
        {
            "review_id": str(row["review_id"]) if row["review_id"] is not None else "",
            "actual_label": int(row["actual_label"]),
            "predicted_label": int(row["predicted_label"]),
            "text_preview": str(row["text_clean"])[:220],
        }
        for row in sample_rows
    ]

    return {
        "total_misclassified": int(total_misclassified),
        "top_confusions": top_confusions,
        "sample_errors": sample_errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train sentiment model with Spark ML")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config")
    return parser.parse_args()


def build_hyperparam_grid(model_cfg: Dict[str, object]) -> List[tuple[float, float]]:
    """Build (regParam, elasticNetParam) pairs from config.

    We tune with a validation split using macro-F1 as the selection criterion.
    """
    reg_params: List[float] = [float(v) for v in model_cfg.get("reg_params", [0.1])]
    elastic_net_params: List[float] = [float(v) for v in model_cfg.get("elastic_net_params", [0.0])]
    grid = [(reg, enet) for reg, enet in product(reg_params, elastic_net_params)]
    if not grid:
        raise ValueError("Model hyperparameter grid is empty. Check reg_params and elastic_net_params.")
    return grid


def ensure_non_empty_split(df: DataFrame, split_name: str) -> None:
    """Fail fast when a split is empty to avoid confusing Spark ML errors."""
    if df.limit(1).count() == 0:
        raise ValueError(
            f"{split_name} split is empty. Increase sample_fraction or reduce filtering constraints."
        )


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

        train_df, val_df, test_df = train_source_df.randomSplit([0.70, 0.15, 0.15], seed=seed)
        ensure_non_empty_split(train_df, "train")
        ensure_non_empty_split(val_df, "validation")
        ensure_non_empty_split(test_df, "test")

        class_weight_multipliers_cfg = model_cfg.get("class_weight_multipliers", {})
        class_weight_multipliers: Dict[int, float] = {
            int(label): float(multiplier)
            for label, multiplier in class_weight_multipliers_cfg.items()
        }
        feature_stages = build_feature_stages_with_options(preprocessing_cfg)

        weighted_train_df = add_class_weights(train_df, multipliers=class_weight_multipliers)
        param_grid = build_hyperparam_grid(model_cfg)

        best_val_macro_f1 = float("-inf")
        best_params = (0.1, 0.0)

        for reg_param, elastic_net_param in param_grid:
            lr = LogisticRegression(
                featuresCol="features",
                labelCol="label",
                weightCol="class_weight",
                maxIter=int(model_cfg["max_iter"]),
                regParam=float(reg_param),
                elasticNetParam=float(elastic_net_param),
            )
            candidate_pipeline = Pipeline(stages=[*feature_stages, lr])
            candidate_model = candidate_pipeline.fit(weighted_train_df)
            val_predictions = candidate_model.transform(val_df)
            val_macro_f1 = float(compute_metrics(val_predictions)["macro_f1"])

            LOGGER.info(
                "Tuning candidate regParam=%s elasticNetParam=%s validation_macro_f1=%.6f",
                reg_param,
                elastic_net_param,
                val_macro_f1,
            )

            if val_macro_f1 > best_val_macro_f1:
                best_val_macro_f1 = val_macro_f1
                best_params = (float(reg_param), float(elastic_net_param))

        full_train_df = train_df.unionByName(val_df)
        weighted_full_train_df = add_class_weights(full_train_df, multipliers=class_weight_multipliers)

        final_lr = LogisticRegression(
            featuresCol="features",
            labelCol="label",
            weightCol="class_weight",
            maxIter=int(model_cfg["max_iter"]),
            regParam=best_params[0],
            elasticNetParam=best_params[1],
        )
        final_pipeline = Pipeline(stages=[*feature_stages, final_lr])
        best_model = final_pipeline.fit(weighted_full_train_df)
        predictions = best_model.transform(test_df)

        metrics = compute_metrics(predictions)
        metrics["selection_strategy"] = "holdout_validation_macro_f1"
        metrics["selected_reg_param"] = float(best_params[0])
        metrics["selected_elastic_net_param"] = float(best_params[1])
        metrics["validation_macro_f1"] = float(best_val_macro_f1)
        metrics["train_rows"] = int(full_train_df.count())
        metrics["test_rows"] = int(test_df.count())
        try:
            metrics["error_analysis"] = build_error_analysis(
                predictions,
                confusion_matrix=metrics["confusion_matrix"],
            )
        except Exception as exc:  # pragma: no cover - defensive fallback for low-disk runs
            LOGGER.warning("Skipping error_analysis due to runtime issue: %s", exc)
            metrics["error_analysis"] = {
                "status": "skipped",
                "reason": str(exc),
            }

        best_model.write().overwrite().save(paths["model_dir"])
        write_json_payload(spark, paths["metrics_json"], metrics)

        LOGGER.info("Training completed. Model saved at: %s", paths["model_dir"])
        LOGGER.info("Metrics saved at: %s", paths["metrics_json"])
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
