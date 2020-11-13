import pandas as pd
import pytest

from mle_training import score_pretrained  # module to score pretrained model


@pytest.fixture()
def data():
    df = pd.read_csv("data/small_sample_data/housing.csv")
    return df


def test_model_score(data):
    import numpy as np

    from mle_training.utils import data_preprocess as preprocess

    housing = data

    # Fit missing value imputer on train data
    preprocess.fit(train_data=housing)

    # Transform train and test data
    X_train, y_train = preprocess.transform(data=housing)

    from mle_training import train_score  # train and score module

    # Fit model and score on training set
    lin_model = train_score.linear_reg_model(X=X_train, y=y_train)

    # Score trained model on test set (can be a model stored in a pickle file)
    lin_rmse = score_pretrained.model_score(
        model=lin_model, X=[X_train.iloc[0, :].values], y=[y_train[0]]
    )
    assert np.round(lin_rmse, 3) == 30937.050
