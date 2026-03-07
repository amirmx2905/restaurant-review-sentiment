from __future__ import annotations

import json
from typing import Any, Dict

from pyspark.sql import SparkSession


def delete_path_if_exists(spark: SparkSession, path: str) -> None:
    """Delete a local/HDFS path if it exists."""
    jvm = spark.sparkContext._jvm
    jsc = spark.sparkContext._jsc
    hadoop_conf = jsc.hadoopConfiguration()
    fs = jvm.org.apache.hadoop.fs.FileSystem.get(hadoop_conf)
    target = jvm.org.apache.hadoop.fs.Path(path)

    if fs.exists(target):
        fs.delete(target, True)


def write_json_payload(spark: SparkSession, path: str, payload: Dict[str, Any]) -> None:
    """Write a single JSON object as text to a local/HDFS path."""
    delete_path_if_exists(spark, path)
    spark.sparkContext.parallelize([json.dumps(payload, sort_keys=True)], 1).saveAsTextFile(path)
