import logging

import numpy as np
import pandas as pd

logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)


def lattitude_vs_longitude_plot(housing_data: pd.DataFrame):
    """
    This function created plots of lattitude vs longitude and saves it in the plots \
        directory.

    Args:
        housing_data (pd.DataFrame): housing dataset
    """
    logging.info("Generating Plots for Exploratory Analysis")
    ax = housing_data.plot(kind="scatter", x="longitude", y="latitude")
    fig = ax.get_figure()
    fig.savefig("../plots/lattitude_vs_longitude.png")
    ax = housing_data.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
    fig = ax.get_figure()
    fig.savefig("../plots/lattitude_vs_longitude_alpha.png")
    return


def corr_independent_dependent(housing_data: pd.DataFrame):
    """
    Find the correlation between the dependent and independent variables

    Args:
        housing_data (pd.DataFrame): housing dataset

    Returns:
        pd.Series: correlation between dependent and independent variables
    """
    logging.info("Calculating correlation between independent and dependent variables")
    corr_matrix = housing_data.corr()
    return corr_matrix["median_house_value"].sort_values(ascending=False)


def compare_props(
    housing_data: pd.DataFrame,
    strat_test_set: pd.DataFrame,
    random_test_set: pd.DataFrame,
):
    """
    Compares the proportion of people from different income categories in test data \
        generated from different splitting methods.

    Args:
        housing_data (pd.DataFrame): raw housing dataset

        strat_test_set (pd.DataFrame): test set generated using stratified split based \
             on income category

        random_test_set (pd.DataFrame): randomly generated test set

    Returns:
        pd.DataFrame : Dataframe comparing the proportions of income categories
    """
    housing_data["income_cat"] = pd.cut(
        housing_data["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    strat_test_set["income_cat"] = pd.cut(
        strat_test_set["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    random_test_set["income_cat"] = pd.cut(
        random_test_set["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    logging.info("Calculating Proportion of Income category across all samples")
    compare_props = pd.DataFrame(
        {
            "Overall": housing_data["income_cat"].value_counts() / len(housing_data),
            "Stratified": strat_test_set["income_cat"].value_counts()
            / len(strat_test_set),
            "Random": random_test_set["income_cat"].value_counts()
            / len(random_test_set),
        }
    ).sort_index()
    compare_props["Rand. %error"] = (
        100 * compare_props["Random"] / compare_props["Overall"] - 100
    )
    compare_props["Strat. %error"] = (
        100 * compare_props["Stratified"] / compare_props["Overall"] - 100
    )
    return compare_props
