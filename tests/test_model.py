"""End-to-end check: a small version of the final model trains and writes a valid submission."""

from house_prices.data import clean_house_data, is_outlier, load_house_data
from house_prices.features import make_features
from house_prices.models import build_model, load_config, unusual_sale_weights
from house_prices.predict import make_submission
from house_prices.settings import CONFIG_DIR


def test_config_builds_the_documented_blend():
    model = build_model(load_config(CONFIG_DIR / "final_model.toml"))
    assert [name for name, _ in model.members] == ["Ridge", "Gradient Boosting"]
    assert model.weights.tolist() == [1.0, 1.0]


def test_small_model_writes_valid_submission(raw_data_dir):
    config = load_config(CONFIG_DIR / "final_model.toml")
    config["members"][1]["params"]["n_estimators"] = 20  # keep the test fast

    train = clean_house_data(load_house_data(raw_data_dir / "train.csv"))
    train = train.loc[~is_outlier(train)]
    model = build_model(config).fit(
        make_features(train), train["SalePrice"],
        sample_weight=unusual_sale_weights(train, config["unusual_sale_weight"]),
    )

    test_raw = load_house_data(raw_data_dir / "test.csv")
    submission = make_submission(model, test_raw)
    assert len(submission) == len(test_raw)
    assert submission["SalePrice"].between(10_000, 2_000_000).all()
