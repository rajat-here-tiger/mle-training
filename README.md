# Median housing value prediction

The housing data can be downloaded from https://raw.githubusercontent.com/ageron/handson-ml/master/. The script has codes to download the data. We have modelled the median house value on given housing data. 

The following techniques have been used: 

 - Linear regression
 - Decision Tree
 - Random Forest

## Steps performed
 - We prepare and clean the data. We check and impute for missing values.
 - Features are generated and the variables are checked for correlation.
 - Multiple sampling techinuqies are evaluated. The data set is split into train and test.
 - All the above said modelling techniques are tried and evaluated. The final metric used to evaluate is root mean squared error.

## To set up the project
1. Clone the repository in a local directory.
```
git clone https://github.com/rajat-here-tiger/mle-training
cd mle-training
```
2. Create and activate virtual python development environment. On terminal, run
```
conda env create -f env.yaml
source activate devenv
```
3. Download the housing data. On terminal, run
```
bash download_data.sh 
```
4. Install the package in development mode. On terminal, run
```
pip install -e .
```
5. Verify correct installation
```
pytest
```
6. Generate Sphinx Documentation
```
cd docs
make html
```

# Structure of project
```
.
├── README.md
├── data
│   ├── raw
│   │   └── housing.csv
│   └── small_sample_data
│       └── housing.csv
├── docs/
├── download_data.sh
├── env.yaml
├── mle_training
│   ├── __init__.py
│   ├── predict.py
│   ├── score_pretrained.py
│   ├── train_score.py
│   └── utils
│       ├── __init__.py
│       ├── data_exploration.py
│       └── data_preprocess.py
├── notebooks
│   └── example_notebook.ipynb
├── pickles/
│   ├── imputers/
│   └── models/
├── plots/
├── setup.py
└── tests/
```

## Using the package

### 1. Preprocess the raw housing data for modeling

```python
from mle_training.utils import data_preprocess as preprocess

# Get data
housing = preprocess.get_data()

# Stratified split based on income category
train_data, test_data = preprocess.data_strat_split(data=housing,test_size=0.2,random_state=42)

# Uncomment for Random split
# train_data, test_data = preprocess.data_random_split(data=housing, test_size=0.2, random_state=42)

# Fit missing value imputer on train data
preprocess.fit(train_data=train_data)

# Transform train and test data
X_train, y_train = preprocess.transform(data=train_data)
X_test, y_test = preprocess.transform(data=test_data)
```

### 2. Modeling

```python
from mle_training import train_score  # train and score module
from mle_training import score_pretrained  # module to score pretrained model
from mle_training import predict  # module to make predictions
```
#### 1. Linear Regression Model

```python
# Fit model and score on training set
lin_model = train_score.linear_reg_model(X=X_train, y=y_train)

# Additional optional arguments
# param_grid = [{"fit_intercept": [True, False]}]  # Example hyperparameters to tune
# method = "grid_search"  # "grid_search" (default) or "random_search", method to perform hyperparameter tuning
# lin_model = train_score.lin_reg_model(X=X_train, y=y_train, param_grid=param_grid, method="random_search")

# Make predictions from pretrained model (can be a model stored in a pickle file)
y_pred = predict.model_predict(model=lin_model, X=X_test)

# Score trained model on test set (can be a model stored in a pickle file)
lin_rmse = score_pretrained.model_score(model=lin_model, X=X_test, y=y_test)
```

#### 2. Decision Tree Regression Model

```python
# Fit model and score on training set
tree_model = train_score.tree_reg_model(X=X_train, y=y_train)

# Additional optional arguments
# param_grid = [{'criterion': ['mse','mae'], 'max_depth': [2, 4]}]  # Example hyperparameters to tune
# method = "grid_search"  # "grid_search" (default) or "random_search", method to perform hyperparameter tuning
# tree_model = train_score.tree_reg_model(X=X_train, y=y_train, param_grid=param_grid, method="random_search")

# Make predictions from pretrained model (can be a model stored in a pickle file)
y_pred = predict.model_predict(model=tree_model, X=X_test)

# Score trained model on test set (can be a model stored in a pickle file)
tree_rmse = score_pretrained.model_score(model=tree_model, X=X_test, y=y_test)
```

#### 3. Random Forest Regression Model

```python
# Fit model and score on training set
forest_model = train_score.forest_reg_model(X=X_train, y=y_train)

# Additional optional arguments
# param_grid = [
#     # try 12 (3×4) combinations of hyperparameters
#     {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
#     # then try 6 (2×3) combinations with bootstrap set as False
#     {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
# ] # Example hyperparameters to tune
# method = "grid_search"  # "grid_search" (default) or "random_search", method to perform hyperparameter tuning
# forest_model = train_score.tree_reg_model(X=X_train, y=y_train, param_grid=param_grid, method="grid_search")

# Make predictions from pretrained model (can be a model stored in a pickle file)
y_pred = predict.model_predict(model=forest_model, X=X_test)

# Score trained model on test set (can be a model stored in a pickle file)
forest_rmse = score_pretrained.model_score(model=forest_model, X=X_test, y=y_test)
```

### 3. Data Exploration functions

These functions can be used with raw housing data to:
1. Generate lattitude vs longitude plots  
2. Finad correlation between independent and dependent variables  
3. Generate comparison between random and stratified splitting techniques  

```python
from mle_training.utils import data_exploration as explore

# Generate lattitude vs longitude plots
explore.lattitude_vs_longitude_plot(housing_data=housing)

# Correlation between independent and dependent variables
explore.corr_independent_dependent(housing_data=housing)

# Compare split proportion
# Stratified split based on income category
_, strat_test_data = preprocess.data_strat_split(data=housing, test_size=0.2, random_state=42)

# Random split
_, random_test_data = preprocess.data_random_split(data=housing, test_size=0.2, random_state=42)

compare_matrix = explore.compare_props(
    housing_data=housing,
    strat_test_set=strat_test_data,
    random_test_set=random_test_data,
)
```

### 4. Running tests
On terminal, run
```
pytest
```