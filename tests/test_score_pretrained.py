import pandas as pd
import pytest


@pytest.fixture(scope="package")
def data():
    df = pd.read_csv("data/small_sample_data/housing.csv")
    return df


def test_model_score(data):
    import numpy as np

    from mle_training.utils import data_preprocess as preprocess

    housing = data
    # Stratified split based on income category
    train_data, test_data = preprocess.data_strat_split(
        data=housing, test_size=0.2, random_state=42
    )

    # Fit missing value imputer on train data
    preprocess.fit(train_data=train_data)

    # Transform train and test data
    X_train, y_train = preprocess.transform(data=train_data)
    X_test, y_test = preprocess.transform(data=test_data)

    from mle_training import score_pretrained  # module to score pretrained model
    from mle_training import train_score  # train and score module

    # Fit model and score on training set
    lin_model = train_score.linear_reg_model(X=X_train, y=y_train)

    # Score trained model on test set (can be a model stored in a pickle file)
    lin_rmse = score_pretrained.model_score(model=lin_model, X=X_test, y=y_test)
    assert np.round(lin_rmse, 3) == 67796.575
