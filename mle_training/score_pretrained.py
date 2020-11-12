import logging

import numpy as np


def model_score(model, X, y):
    """
    Score a pretrained model on given data

    Args:
        model (scikit-learn model-like): Scikit learn pretrained model

        X (Pandas dataframe or matrix-like): independent variables

        y (Pandas series or array-like): dependent variables

    Returns:
        float: model RMSE score on given data
    """
    logging.info("Scoring Model Performance on Test Data")
    from sklearn.metrics import mean_squared_error

    y_pred = model.predict(X)
    model_mse = mean_squared_error(y, y_pred)
    model_rmse = np.sqrt(model_mse)
    logging.info("Model RMSE Score {:.3f}".format(model_rmse))
    return model_rmse
