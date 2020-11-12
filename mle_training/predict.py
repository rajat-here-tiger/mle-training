import logging


def model_predict(model, X):
    """
    Make predictions from a trained model

    Args:
        model (scikit learn model-like): a pretrained model

        X (Pandas dataframe or matrix-like): independent variables
    """
    logging.info("Predicting Model Outcome")
    from sklearn.exceptions import NotFittedError

    try:
        return model.predict(X)
    except NotFittedError:
        logging.error("Train the mode on training set first.")
        raise
