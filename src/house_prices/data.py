"""Reading and rule-based cleaning of the competition CSV files (notebook sections 2-4)."""

import pandas as pd

BASEMENT_NUMBERS = ["BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath"]
BASEMENT_CATEGORIES = ["BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"]


def load_house_data(file_path):
    """Read a competition CSV while keeping "None" as a real category."""
    return pd.read_csv(file_path, keep_default_na=False, na_values=["", "NA"])


def clean_house_data(df):
    """Apply fixed, row-by-row cleaning rules from the data dictionary.

    No statistics are learned here, so the same function can be applied
    to the training data and to unseen test data without leakage.
    """
    result = df.copy()

    # Features whose missing values mean absence according to the dictionary
    absence_labels = {"Alley": "NoAlley", "Fence": "NoFence", "MiscFeature": "NoMiscFeature"}
    for column, label in absence_labels.items():
        result[column] = result[column].fillna(label)

    # Confirm absence using related numerical features
    # A missing measurement is not treated as confirmed absence
    no_pool = result["PoolArea"].eq(0)
    no_fireplace = result["Fireplaces"].eq(0)
    no_garage = result["GarageArea"].eq(0) & result["GarageCars"].eq(0)
    no_basement = (
        result["TotalBsmtSF"].eq(0)
        | result[BASEMENT_NUMBERS + BASEMENT_CATEGORIES].isna().all(axis=1)
    )

    facility_groups = [
        (["PoolQC"], no_pool, "NoPool"),
        (["FireplaceQu"], no_fireplace, "NoFireplace"),
        (["GarageType", "GarageFinish", "GarageQual", "GarageCond"], no_garage, "NoGarage"),
        (BASEMENT_CATEGORIES, no_basement, "NoBasement"),
    ]
    for columns, absence_mask, label in facility_groups:
        for column in columns:
            result.loc[absence_mask & result[column].isna(), column] = label
            # Remaining missing values mean information is unknown
            result[column] = result[column].fillna("Unknown")

    # No basement means zero basement area and no basement bathrooms
    for column in BASEMENT_NUMBERS:
        result.loc[no_basement & result[column].isna(), column] = 0

    # Garage year: a typo (more than a year after the sale) or no garage
    # means the house's own year is used
    result["HasGarage"] = (~no_garage).astype(int)
    use_house_year = no_garage | result["GarageYrBlt"].gt(result["YrSold"] + 1)
    result.loc[use_house_year, "GarageYrBlt"] = result.loc[use_house_year, "YearBuilt"]

    # Flag contradictory masonry records in both directions
    area_without_type = result["MasVnrType"].eq("None") & result["MasVnrArea"].gt(0)
    type_without_area = (
        result["MasVnrType"].notna() & result["MasVnrType"].ne("None") & result["MasVnrArea"].eq(0)
    )
    result["VeneerConflict"] = (area_without_type | type_without_area).astype(int)
    result.loc[area_without_type, "MasVnrType"] = "Unknown"
    result["MasVnrType"] = result["MasVnrType"].replace({"None": "NoVeneer"}).fillna("Unknown")

    # Explicit "no veneer" means zero area; other unknown areas are imputed later
    result["MasVnrArea"] = result["MasVnrArea"].fillna(result["MasVnrType"].map({"NoVeneer": 0.0}))

    return result


def is_outlier(df):
    """Very large houses sold at unusually low prices (documented by the dataset author)."""
    return df["GrLivArea"].gt(4000) & df["SalePrice"].lt(300000)
