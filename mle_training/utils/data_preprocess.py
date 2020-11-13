import logging
import os
import pickle

import numpy as np
import pandas as pd

logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)
HOUSING_PATH = "../data/raw"
PKL_IMPUTER = "../pickles/imputers/Pickle_Imputer.pkl"


def get_data(housing_path: str = HOUSING_PATH):
    """
    Load the raw housing dataset

    Args:
        housing_path (str, optional): location to "housing.csv" file. Defaults to \
            HOUSING_PATH.

    Returns:
        pd.Dataframe: raw housing dataset
    """
    logging.info("Importing the Housing Price Dataset")
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def data_random_split(data, test_size: float = 0.2, random_state: int = 42):
    """
    Randomly split data into train/test

    Args:
        data (pandas dataframe): housing dataset

        test_size (float, optional): test set size ratio. Defaults to 0.2.

        random_state (int, optional): random seed. Defaults to 42.

    Returns:
        pd.Dataframe, pd.Dataframe: train and test set
    """
    from sklearn.model_selection import train_test_split

    logging.info("Randomly Splitting the Housing Data")
    train_set, test_set = train_test_split(
        data, test_size=test_size, random_state=random_state
    )
    return train_set, test_set


def data_strat_split(data, test_size: float = 0.2, random_state: int = 42):
    """Split data into train/test based on income category

    Args:
        data (pandas dataframe): housing dataset

        test_size (float, optional): test set size ratio. Defaults to 0.2.

        random_state (int, optional): random seed. Defaults to 42.

    Returns:
        pd.Dataframe, pd.Dataframe: train and test set
    """
    data["income_cat"] = pd.cut(
        data["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    logging.info("Splitting the data into train/test")

    from sklearn.model_selection import StratifiedShuffleSplit

    split = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    for train_index, test_index in split.split(data, data["income_cat"]):
        strat_train_set = data.loc[train_index]
        strat_test_set = data.loc[test_index]

    logging.info("dropping income cat")
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    return strat_train_set, strat_test_set


def fit(train_data):
    """
    Fit the missing value imputer on train data

    Args:
        train_data (pd.DataFrame): training data
    """
    from sklearn.impute import SimpleImputer

    logging.info("Fitting imputer for missing values")
    data = train_data.drop(["median_house_value", "ocean_proximity"], axis=1).copy()
    imputer = SimpleImputer(strategy="median").fit(data)
    logging.info("Creating pickle file for Imputer")
    os.makedirs(os.path.dirname(PKL_IMPUTER), exist_ok=True)
    with open(PKL_IMPUTER, "wb") as file:
        pickle.dump(imputer, file)


def transform(data):
    """
    Perform Data Preprocessing, Cleaning and feature engineering

    Args:
        data (pd.DataFrame): data to transform

    Returns:
        pd.DataFrame, pd.DataFrame: cleaned and transformed independent and dependent \
            variables
    """
    logging.info("Extracting Dependent and Independent variables.")
    housing_label = data["median_house_value"]
    housing_data = data.drop(
        ["median_house_value"], axis=1
    )  # drop labels for training set

    housing_num = housing_data.drop("ocean_proximity", axis=1)

    logging.info("Dealing with missing values")
    logging.info("Importing the Imputer pickle file")

    try:
        with open(PKL_IMPUTER, "rb") as file:
            imputer = pickle.load(file)
    except FileNotFoundError:
        logging.error("The missing value imputer has to be fit on training data first")
        raise

    X = imputer.transform(housing_num)

    housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_data.index)

    logging.info("Creating New Features")
    housing_tr["rooms_per_household"] = (
        housing_tr["total_rooms"] / housing_tr["households"]
    )
    housing_tr["bedrooms_per_room"] = (
        housing_tr["total_bedrooms"] / housing_tr["total_rooms"]
    )
    housing_tr["population_per_household"] = (
        housing_tr["population"] / housing_tr["households"]
    )

    logging.info("Encoding Categorical Variables")
    housing_cat = housing_data[["ocean_proximity"]]
    housing_prepared = housing_tr.join(pd.get_dummies(housing_cat, drop_first=True))

    return housing_prepared, housing_label
