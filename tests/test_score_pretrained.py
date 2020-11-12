import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from mle_training import score_pretrained


@pytest.fixture(scope="package")
def X_data():
    df = pd.DataFrame.from_dict(
        {
            "longitude": {5241: -118.39, 10970: -117.86, 20351: -119.05},
            "latitude": {5241: 34.12, 10970: 33.77, 20351: 34.21},
            "housing_median_age": {5241: 29.0, 10970: 39.0, 20351: 27.0},
            "total_rooms": {5241: 6447.0, 10970: 4159.0, 20351: 4357.0},
            "total_bedrooms": {5241: 1012.0, 10970: 655.0, 20351: 926.0},
            "population": {5241: 2184.0, 10970: 1669.0, 20351: 2110.0},
            "households": {5241: 960.0, 10970: 651.0, 20351: 876.0},
            "median_income": {5241: 8.2816, 10970: 4.6111, 20351: 3.0119},
            "rooms_per_household": {
                5241: 6.715625,
                10970: 6.38863287250384,
                20351: 4.973744292237443,
            },
            "bedrooms_per_room": {
                5241: 0.1569722351481309,
                10970: 0.15748978119740323,
                20351: 0.2125315584117512,
            },
            "population_per_household": {
                5241: 2.275,
                10970: 2.563748079877112,
                20351: 2.4086757990867578,
            },
            "ocean_proximity_INLAND": {5241: 0, 10970: 0, 20351: 0},
            "ocean_proximity_ISLAND": {5241: 0, 10970: 0, 20351: 0},
            "ocean_proximity_NEAR BAY": {5241: 0, 10970: 0, 20351: 0},
            "ocean_proximity_NEAR OCEAN": {5241: 0, 10970: 0, 20351: 0},
        }
    )
    return df


@pytest.fixture(scope="package")
def y_data():
    df = pd.Series({5241: 500001.0, 10970: 240300.0, 20351: 218200.0})
    return df


@pytest.mark.parametrize(
    "model", [LinearRegression(), DecisionTreeRegressor(), RandomForestRegressor()]
)
def test_model_score(model, X_data, y_data):
    assert isinstance(
        score_pretrained.model_score(model.fit(X_data, y_data), X_data, y_data), float
    )
