import pytest

from house_prices.settings import RAW_DATA_DIR

# Tests marked with this are skipped until the Kaggle data is downloaded (see data/README.md)
needs_data = pytest.mark.skipif(
    not (RAW_DATA_DIR / "train.csv").exists() or not (RAW_DATA_DIR / "test.csv").exists(),
    reason="Kaggle data not found in data/raw/",
)
