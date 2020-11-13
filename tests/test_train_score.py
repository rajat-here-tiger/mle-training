import pickle

import pandas as pd
import pytest

from mle_training import train_score  # train and score module


@pytest.fixture()
def data():
    df = pd.read_csv("data/small_sample_data/housing.csv")
    return df


@pytest.fixture()
def test():
    X_test = [
        [
            -1.18210000e02,
            3.39100000e01,
            2.40000000e01,
            1.54500000e03,
            3.91000000e02,
            1.80700000e03,
            3.88000000e02,
            2.64290000e00,
            3.98195876e00,
            2.53074434e-01,
            4.65721649e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
        ]
    ]
    return X_test


def test_linear_reg_model(data, test):
    import numpy as np

    from mle_training.utils import data_preprocess as preprocess

    housing = data

    # Fit missing value imputer on train data
    preprocess.fit(train_data=housing)

    # Transform train and test data
    X_train, y_train = preprocess.transform(data=housing)

    # Fit model and score on training set
    lin_model = train_score.linear_reg_model(X=X_train, y=y_train)

    with open("pickles/models/Pickle_Linear_Model.pkl", "rb") as file:
        save_model = pickle.load(file)

    y_pred_model = lin_model.predict(test)[0]
    y_pred_save_model = save_model.predict(test)[0]

    assert (np.round(y_pred_model, 3) == 136237.050) & (
        np.round(y_pred_save_model, 3) == 136237.050
    )


def test_tree_reg_model(data, test):
    import numpy as np

    from mle_training.utils import data_preprocess as preprocess

    housing = data

    # Fit missing value imputer on train data
    preprocess.fit(train_data=housing)

    # Transform train and test data
    X_train, y_train = preprocess.transform(data=housing)

    # Fit model and score on training set
    tree_model = train_score.tree_reg_model(X=X_train, y=y_train)

    with open("pickles/models/Pickle_Tree_Model.pkl", "rb") as file:
        save_model = pickle.load(file)

    y_pred_model = tree_model.predict(test)[0]
    y_pred_save_model = save_model.predict(test)[0]

    assert (np.round(y_pred_model, 3) == 105300.0) & (
        np.round(y_pred_save_model, 3) == 105300.0
    )


def test_forest_reg_model(data, test):
    import numpy as np

    from mle_training.utils import data_preprocess as preprocess

    housing = data

    # Fit missing value imputer on train data
    preprocess.fit(train_data=housing)

    # Transform train and test data
    X_train, y_train = preprocess.transform(data=housing)

    # Fit model and score on training set
    forest_model = train_score.forest_reg_model(X=X_train, y=y_train)

    with open("pickles/models/Pickle_Forest_Model.pkl", "rb") as file:
        save_model = pickle.load(file)

    y_pred_model = forest_model.predict(test)[0]
    y_pred_save_model = save_model.predict(test)[0]

    assert (np.round(y_pred_model, 3) == 134796.667) & (
        np.round(y_pred_save_model, 3) == 134796.667
    )
