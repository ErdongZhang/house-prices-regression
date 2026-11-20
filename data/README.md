# Data

The competition data is not included in this repository (Kaggle's rules do not allow redistributing it).

Download it from the [competition page](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data) (a free Kaggle account is required) and put the files here:

```
data/raw/
├── train.csv              # 1,460 houses with SalePrice
├── test.csv               # 1,459 houses to predict
└── data_description.txt   # meaning of every column and code
```

With the [Kaggle CLI](https://github.com/Kaggle/kaggle-api) configured:

```bash
kaggle competitions download -c house-prices-advanced-regression-techniques -p data/raw
unzip data/raw/house-prices-advanced-regression-techniques.zip -d data/raw
```

The data is based on the Ames Housing dataset compiled by Dean De Cock (*Journal of Statistics Education* 19(3), 2011).
