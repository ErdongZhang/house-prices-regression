"""The fixed cleaning rules from notebook section 3."""

import io

import pandas as pd

from conftest import needs_data

from house_prices.data import clean_house_data, is_outlier, load_house_data
from house_prices.settings import RAW_DATA_DIR


def test_load_keeps_none_as_a_category():
    csv = io.StringIO("Id,MasVnrType\n1,None\n2,NA\n3,\n")
    column = load_house_data(csv)["MasVnrType"]
    assert column.iloc[0] == "None"
    assert column.iloc[1:].isna().all()


@needs_data
def test_cleaning_rules_on_real_rows():
    train = load_house_data(RAW_DATA_DIR / "train.csv")
    clean = clean_house_data(train)

    no_garage = train["GarageArea"].eq(0)
    assert clean.loc[no_garage, "GarageType"].eq("NoGarage").all()
    assert clean.loc[no_garage, "HasGarage"].eq(0).all()
    # Houses without a garage take the house's own year
    assert clean.loc[no_garage, "GarageYrBlt"].equals(clean.loc[no_garage, "YearBuilt"].astype(float))

    # Absence categories never stay missing
    for column in ["Alley", "Fence", "MiscFeature", "PoolQC", "FireplaceQu", "BsmtQual"]:
        assert clean[column].notna().all(), column

    # Cleaning learns nothing, so it never changes the number or order of rows
    assert clean["Id"].equals(train["Id"])


def test_outlier_rule():
    houses = pd.DataFrame({"GrLivArea": [5000, 5000, 1500], "SalePrice": [150_000, 600_000, 150_000]})
    assert is_outlier(houses).tolist() == [True, False, False]
