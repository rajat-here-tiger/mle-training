import logging
import os
import pickle

import numpy as np

logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)
PKL_LINEAR_MODEL = "../pickles/models/Pickle_Linear_Model.pkl"
PKL_TREE_MODEL = "../pickles/models/Pickle_Tree_Model.pkl"
PKL_FOREST_MODEL = "../pickles/models/Pickle_Forest_Model.pkl"


def linear_reg_model(X, y, param_grid=None, method: str = "grid_search"):
    """
    Fit a linear regression model to given data and score performance

    Args:
        X (Pandas dataframe or matrix-like): independent variables

        y (Pandas series or array-like): dependent variables

        param_grid (dict or list of dict, optional): parameter grid to use for tuning.\
            Defaults to None.

        method (str, optional): hypertuning method to use \
            (grid_search or random_search). Defaults to "grid_search".

    Returns:
        LinearRegression: linear regression model trained on given data
    """
    from sklearn.linear_model import LinearRegression

    logging.info("Training Linear Regression Model")
    if param_grid is None:
        model = LinearRegression().fit(X, y)
    else:
        model = LinearRegression()
        if method == "grid_search":
            logging.info(
                "Performing Grid Search for Optimal Hyperparameters in Linear\
                Regression"
            )
            model = best_grid_search(X, y, model, param_grid)

        elif method == "random_search":
            logging.info(
                "Performing Randomized Search for Optimal Hyperparameters in Linear\
                Regression"
            )
            model = best_random_search(X, y, model, param_grid)

    logging.info("Calculating Model Score on Training Data")
    model_score(model, X, y)

    logging.info("Exporting Model into " + PKL_LINEAR_MODEL)
    os.makedirs(os.path.dirname(PKL_LINEAR_MODEL), exist_ok=True)
    with open(PKL_LINEAR_MODEL, "wb") as file:
        pickle.dump(model, file)
    return model


def tree_reg_model(
    X, y, param_grid=None, method: str = "grid_search", random_state: int = 42
):
    """
    Fit a decision tree regression model to given data and score performance

    Args:
        X (Pandas dataframe or matrix-like): independent variables

        y (Pandas series or array-like): dependent variables

        param_grid (dict or list of dict, optional): parameter grid to use for tuning.\
            Defaults to None.

        method (str, optional): hypertuning method to use\
            (grid_search or random_search). Defaults to "grid_search".

        random_state (int, optional): random seed. Defaults to 42.

    Returns:
        DecisionTreeReggressor: decision tree regression model trained on given data
    """
    from sklearn.tree import DecisionTreeRegressor

    logging.info("Training Decision Tree Regression Model")
    if param_grid is None:
        model = DecisionTreeRegressor(random_state=random_state).fit(X, y)
    else:
        model = DecisionTreeRegressor(random_state=random_state)
        if method == "grid_search":
            logging.info(
                "Performing Grid Search for Optimal Hyperparameters in Decision Tree\
                Regressor"
            )
            model = best_grid_search(X, y, model, param_grid)

        elif method == "random_search":
            logging.info(
                "Performing Randomized Search for Optimal Hyperparameters in Decision Tree\
                Regressor"
            )
            model = best_random_search(X, y, model, param_grid)

    logging.info("Calculating Model Score on Training Data")
    model_score(model, X, y)

    logging.info("Exporting Model into " + PKL_TREE_MODEL)
    os.makedirs(os.path.dirname(PKL_TREE_MODEL), exist_ok=True)
    with open(PKL_TREE_MODEL, "wb") as file:
        pickle.dump(model, file)
    return model


def forest_reg_model(
    X, y, param_grid=None, method: str = "grid_search", random_state: int = 42
):
    """
    Fit a random forest regression model to given data and score performance

    Args:
        X (Pandas dataframe or matrix-like): independent variables

        y (Pandas series or array-like): dependent variables

        param_grid (dict or list of dict, optional): parameter grid to use for tuning. \
            Defaults to None.

        method (str, optional): hypertuning method to use \
            (grid_search or random_search). Defaults to "grid_search".

        random_state (int, optional): random seed. Defaults to 42.

    Returns:
        RandomForestRegressor: random forest regression model trained on given data
    """
    from sklearn.ensemble import RandomForestRegressor

    logging.info("Training Random Forest Regression Model")

    if param_grid is None:
        model = RandomForestRegressor(
            max_features=6, n_estimators=30, random_state=random_state
        ).fit(X, y)
    else:
        model = RandomForestRegressor(random_state=random_state)

        if method == "grid_search":
            logging.info(
                "Performing Grid Search for Optimal Hyperparameters in Random Forrest\
                Regressor"
            )
            model = best_grid_search(X, y, model, param_grid)

        elif method == "random_search":
            logging.info(
                "Performing Randomized Search for Optimal Hyperparameters in Random Forrest\
                Regressor"
            )
            model = best_random_search(X, y, model, param_grid)

    logging.info("Evaluating Feature Importance in Random Forrest Regressor")
    feature_importances = model.feature_importances_
    print(
        "Feature Importance:\n",
        sorted(zip(feature_importances, X.columns), reverse=True),
    )

    logging.info("Calculating Model Score on Training Data")
    model_score(model, X, y)

    logging.info("Exporting Model into " + PKL_FOREST_MODEL)
    os.makedirs(os.path.dirname(PKL_FOREST_MODEL), exist_ok=True)
    with open(PKL_FOREST_MODEL, "wb") as file:
        pickle.dump(model, file)
    return model


def best_grid_search(X, y, model, param_grid):
    """
    Perform grid search for optimal parameters on a given model

    Args:
        X (Pandas dataframe or matrix-like): independent variables

        y (Pandas series or array-like): dependent variables

        model (scikit-learn model-like): Scikit learn model

        param_grid (dict or list of dict, optional): parameter grid to use for tuning. \
            Defaults to None.

    Returns:
        scikit-learn model: model trained with optimal hyperparameter parameters
    """
    from sklearn.model_selection import GridSearchCV

    # train across 5 folds, that's a total of (12+6)*5=90 rounds of training
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        return_train_score=True,
    )
    grid_search.fit(X, y)

    return grid_search.best_estimator_


def best_random_search(X, y, model, param_grid):
    """
    Perform randomized search for optimal parameters on a given model

    Args:
        X (Pandas dataframe or matrix-like): independent variables

        y (Pandas series or array-like): dependent variables

        model (scikit-learn model-like): Scikit learn model

        param_grid (dict or list of dict, optional): parameter grid to use for tuning. \
            Defaults to None.


    Returns:
        scikit-learn model: model trained with optimal hyperparameter parameters
    """
    from sklearn.model_selection import RandomizedSearchCV

    rnd_search = RandomizedSearchCV(
        model,
        param_distributions=param_grid,
        n_iter=10,
        cv=5,
        scoring="neg_mean_squared_error",
        random_state=42,
    )
    rnd_search.fit(X, y)
    return rnd_search.best_estimator_


def model_score(model, X, y):
    """
    Perform 5 fold cross validation and score mean RMSE of trained model

    Args:
        model (scikit-learn model-like): Scikit learn model

        X (Pandas dataframe or matrix-like): independent variables

        y (Pandas series or array-like): dependent variables
    """
    from sklearn.model_selection import cross_val_score

    scores = cross_val_score(model, X, y, cv=5, scoring="neg_mean_squared_error")
    logging.info("Model RMSE Score {:.3f}".format(np.sqrt(-np.mean(scores))))
    return
