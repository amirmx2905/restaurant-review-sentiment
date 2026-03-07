from __future__ import annotations

import re
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from pyspark.sql import DataFrame


_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
_NON_ALPHA_PATTERN = re.compile(r"[^a-z\s]")
_SPACE_PATTERN = re.compile(r"\s+")


def stars_to_label(stars: float) -> int | None:
    """Map star ratings to sentiment classes: 0=neg, 1=neutral, 2=pos."""
    if stars is None:
        return None
    if 1.0 <= stars <= 2.0:
        return 0
    if stars == 3.0:
        return 1
    if 4.0 <= stars <= 5.0:
        return 2
    return None


def normalize_text(text: str | None) -> str:
    """Normalize text using the same rules as Spark SQL cleaning."""
    if not text:
        return ""
    output = text.lower()
    output = _URL_PATTERN.sub(" ", output)
    output = _NON_ALPHA_PATTERN.sub(" ", output)
    output = _SPACE_PATTERN.sub(" ", output)
    return output.strip()


def clean_and_label(df: "DataFrame", min_words: int) -> "DataFrame":
    """Clean review text, add labels, filter invalid rows, and deduplicate."""
    from pyspark.sql import functions as F

    cleaned = (
        df.withColumn("text_clean", F.lower(F.col("text")))
        .withColumn("text_clean", F.regexp_replace(F.col("text_clean"), r"https?://\S+|www\.\S+", " "))
        .withColumn("text_clean", F.regexp_replace(F.col("text_clean"), r"[^a-z\s]", " "))
        .withColumn("text_clean", F.regexp_replace(F.col("text_clean"), r"\s+", " "))
        .withColumn("text_clean", F.trim(F.col("text_clean")))
        .withColumn(
            "label",
            F.when((F.col("stars") >= 1.0) & (F.col("stars") <= 2.0), F.lit(0))
            .when(F.col("stars") == 3.0, F.lit(1))
            .when((F.col("stars") >= 4.0) & (F.col("stars") <= 5.0), F.lit(2))
            .otherwise(F.lit(None)),
        )
        .withColumn("word_count", F.size(F.split(F.col("text_clean"), r"\s+")))
        .withColumn("text_hash", F.sha2(F.col("text_clean"), 256))
    )

    filtered = cleaned.filter(
        (F.col("text").isNotNull())
        & (F.col("stars").isNotNull())
        & (F.length(F.col("text_clean")) > 0)
        & (F.col("word_count") >= F.lit(min_words))
        & (F.col("label").isNotNull())
    )

    deduped = filtered.dropDuplicates(["text_hash"]).drop("text_hash", "word_count")
    return deduped


def build_feature_stages(vocab_size: int, min_df: int) -> List[object]:
    """Build Spark ML feature stages for tokenization and TF-IDF."""
    from pyspark.ml.feature import CountVectorizer, IDF, RegexTokenizer, StopWordsRemover

    tokenizer = RegexTokenizer(inputCol="text_clean", outputCol="tokens", pattern=r"\W+")
    stopword_remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens")
    vectorizer = CountVectorizer(
        inputCol="filtered_tokens",
        outputCol="tf",
        vocabSize=vocab_size,
        minDF=min_df,
    )
    idf = IDF(inputCol="tf", outputCol="features")
    return [tokenizer, stopword_remover, vectorizer, idf]
