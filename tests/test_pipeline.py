from src.pipeline.processor import normalize_text, stars_to_label


def test_stars_to_label_mapping() -> None:
    assert stars_to_label(1.0) == 0
    assert stars_to_label(2.0) == 0
    assert stars_to_label(3.0) == 1
    assert stars_to_label(4.0) == 2
    assert stars_to_label(5.0) == 2


def test_stars_to_label_invalid() -> None:
    assert stars_to_label(0.0) is None
    assert stars_to_label(5.5) is None
    assert stars_to_label(None) is None


def test_normalize_text() -> None:
    text = "Amazing FOOD!!! Visit https://example.com now"
    assert normalize_text(text) == "amazing food visit now"
