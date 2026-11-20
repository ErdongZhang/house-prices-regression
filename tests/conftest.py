import pytest

from house_prices.settings import RAW_DATA_DIR


@pytest.fixture
def raw_data_dir():
    """Folder with the Kaggle CSV files; tests using it are skipped until the data is downloaded."""
    if not ((RAW_DATA_DIR / "train.csv").exists() and (RAW_DATA_DIR / "test.csv").exists()):
        pytest.skip("Kaggle data not found in data/raw/ (see data/README.md)")
    return RAW_DATA_DIR
