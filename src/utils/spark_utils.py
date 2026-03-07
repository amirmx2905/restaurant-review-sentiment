from __future__ import annotations

import os
import sys
from typing import Any, Dict

from pyspark.sql import SparkSession


def create_spark_session(config: Dict[str, Any]) -> SparkSession:
    """Create and return a SparkSession based on config values."""
    # Keep worker and driver Python versions aligned across local/cluster runs.
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    spark_cfg = config.get("spark", {})
    builder = (
        SparkSession.builder.appName(spark_cfg.get("app_name", "restaurant-review-sentiment"))
        .master(spark_cfg.get("master", "local[*]"))
        .config("spark.sql.shuffle.partitions", str(spark_cfg.get("shuffle_partitions", 200)))
        .config("spark.pyspark.python", sys.executable)
        .config("spark.pyspark.driver.python", sys.executable)
    )

    optional_confs = {
        "spark.driver.memory": spark_cfg.get("driver_memory"),
        "spark.executor.memory": spark_cfg.get("executor_memory"),
        "spark.driver.maxResultSize": spark_cfg.get("driver_max_result_size"),
        "spark.default.parallelism": spark_cfg.get("default_parallelism"),
        "spark.sql.adaptive.enabled": spark_cfg.get("adaptive_enabled"),
    }
    for key, value in optional_confs.items():
        if value is not None:
            builder = builder.config(key, str(value))

    return builder.getOrCreate()
