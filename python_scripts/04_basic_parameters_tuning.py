# ---
# jupyter:
#   jupytext:
#     formats: python_scripts//py:percent,notebooks//ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Introduction to scikit-learn: basic model hyper-parameters tuning
#
# The process of learning a predictive model is driven by a set of internal
# parameters and a set of training data. These internal parameters are called
# hyper-parameters and are specific for each family of models. In addition,
# a specific set of parameters are optimal for a specific dataset and thus they need
# to be optimized.
#
# This notebook shows:
# * the influence of changing model parameters;
# * how to tune these hyper-parameters;
# * how to evaluate the model performance together with hyper-parameter
#   tuning.

# %%
import pandas as pd

df = pd.read_csv(
    "https://www.openml.org/data/get_csv/1595261/adult-census.csv")
# Or use the local copy:
# df = pd.read_csv('../datasets/adult-census.csv')

# %%
target_name = "class"
target = df[target_name].to_numpy()
target

# %%
data = df.drop(columns=[target_name, "fnlwgt"])
data.head()

# %% [markdown]
# Once the dataset is loaded, we split it into a training and testing sets.

# %%
from sklearn.model_selection import train_test_split

df_train, df_test, target_train, target_test = train_test_split(
    data, target, random_state=42)

# %% [markdown]
# Then, we define the preprocessing pipeline to transform differently
# the numerical and categorical data.

# %%
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder

categorical_columns = [
    'workclass', 'education', 'marital-status', 'occupation',
    'relationship', 'race', 'native-country', 'sex']

categories = [data[column].unique()
              for column in data[categorical_columns]]

categorical_preprocessor = OrdinalEncoder(categories=categories)

preprocessor = ColumnTransformer(
    [('cat-preprocessor', categorical_preprocessor, categorical_columns)],
    remainder='passthrough', sparse_threshold=0)

# %% [markdown]
# Finally, we use a tree-based classifier (i.e. histogram gradient-boosting) to
# predict whether or not a person earns more than 50,000 dollars a year.

# %%
# for the moment this line is required to import HistGradientBoostingClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline

model = make_pipeline(
    preprocessor, HistGradientBoostingClassifier(random_state=42))
model.fit(df_train, target_train)
print(f"The accuracy score using a {model.__class__.__name__} is "
      f"{model.score(df_test, target_test):.2f}")

# %% [markdown]
# ## The issue of finding the best model parameters
#
# In the previous example, we created an histogram gradient-boosting classifier
# using the default parameters by omitting to explicitely set these parameters.
#
# However, there is no reasons that this set of parameters are optimal for our
# dataset. For instance, fine-tuning the histogram gradient-boosting can be
# achieved by finding the best combination of the following parameters: (i)
# `learning_rate`, (ii) `min_samples_leaf`, and (iii) `max_leaf_nodes`.
# Nevertheless, finding this combination manually will be tedious. Indeed,
# there are relationship between these parameters which are difficult to find
# manually: increasing the depth of trees (increasing `max_samples_leaf`)
# should be associated with a lower learning-rate.
#
# Scikit-learn provides tools to explore and evaluate the parameters
# space.
# %% [markdown]
# ## Finding the best model hyper-parameters via exhaustive parameters search
#
# Our goal is to find the best combination of the parameters stated above.
#
# In short, we will set these parameters with some defined values, train our
# model on some data, and evaluate the model performance on some left out data.
# Ideally, we will select the parameters leading to the optimal performance on
# the testing set. Scikit-learn provides a `GridSearchCV` estimator which will
# handle the cross-validation and hyper-parameter search for us.

# %% [markdown]
# The first step is to find the name of the parameters to be set. We use the
# method `get_params()` to get this information. For instance, for a single
# model like the `HistGradientBoostingClassifier`, we can get the list such as:

print(
    "The hyper-parameters are for a histogram GBDT model are:")
for param_name in HistGradientBoostingClassifier().get_params().keys():
    print(param_name)

# %% [markdown]
# When the model of interest is a `Pipeline`, i.e. a serie of transformers and
# a predictor, the name of the estimator will be added at the front of the
# parameter name with a double underscore ("dunder") in-between (e.g.
# `estimator__parameters`).
print("The hyper-parameters are for the full-pipeline are:")
for param_name in model.get_params().keys():
    print(param_name)

# %% [markdown]
# The parameters that we want to set are:
# - `'histgradientboostingclassifier__learning_rate'`;
# - `'histgradientboostingclassifier__max_leaf_nodes'`.
# Let see how to use the `GridSearchCV` estimator for doing such search.
# Since the grid-search will be costly, we will only explore the combination
# learning-rate and the maximum number of nodes.

# %%
import numpy as np
from sklearn.model_selection import GridSearchCV

param_grid = {
    'histgradientboostingclassifier__learning_rate': (0.001, 0.1),
    'histgradientboostingclassifier__max_leaf_nodes': (5, 63),
}
model_grid_search = GridSearchCV(model, param_grid=param_grid,
                                 n_jobs=4, cv=5)
model_grid_search.fit(df_train, target_train)
print(
    f"The accuracy score using a {model_grid_search.__class__.__name__} is "
    f"{model_grid_search.score(df_test, target_test):.2f}")

# %% [markdown]
# The `GridSearchCV` estimator takes a `param_grid` parameter which defines
# all hyper-parameters and their associated values. The grid-search will be in
# charge of creating all possible combinations and test them.
#
# The number of combinations will be equal to the product of the number of
# values to explore for each parameter (e.g. in our example 2x2 combinations).
# Thus, adding a new parameters with associated values to explore become
# rapidly computationally expensive.
#
# Once the grid-search fitted, it can be used as any other predictor by calling
# `predict` and `predict_proba`. Internally, it will use the model with the
# best parameters found during `fit`.

# Get predictions for the 5 first samples using the estimator with the best
# parameters.
model_grid_search.predict(df_test.iloc[0:5])

# %% [markdown]
# You can know about these parameters by looking at the `best_params_`
# attribute.

print(
    f"The best set of parameters is: {model_grid_search.best_params_}"
)

# %% [markdown]
# With the `GridSearchCV` estimator, the parameters need to be specified
# explicitely. We mentioned that exploring a large number of values for
# different parameters will be quickly untractable.
#
# Instead, we can randomly generate the parameter candidates. The
# `RandomSearchCV` allows for such stochastic search. It is used similarly to
# the `GridSearchCV` but the sampling distributions need to be specified
# instead of the parameter values.

# %%
from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV

class reciprocal_int:
    def __init__(self, a, b):
        self._distribution = reciprocal(a, b)
    def rvs(self, *args, **kwargs):
        return self._distribution.rvs(*args, **kwargs).astype(int)


param_distributions = {
    'histgradientboostingclassifier__max_iter': reciprocal_int(10, 100),
    'histgradientboostingclassifier__learning_rate': reciprocal(0.001, 0.1),
    'histgradientboostingclassifier__max_leaf_nodes': reciprocal_int(5, 63),
    'histgradientboostingclassifier__min_samples_leaf': reciprocal_int(3, 40),
}
model_grid_search = RandomizedSearchCV(
    model, param_distributions=param_distributions, n_iter=10,
    n_jobs=4, cv=5)
model_grid_search.fit(df_train, target_train)
print(
    f"The accuracy score using a {model_grid_search.__class__.__name__} is "
    f"{model_grid_search.score(df_test, target_test):.2f}")
print(
    f"The best set of parameters is: {model_grid_search.best_params_}"
)

# %% [markdown]
# ## Notes on search efficiency
#
# Be aware that sometimes, scikit-learn provides `EstimatorCV` classes
# which will internally perform the cross-validation in such way that it will
# be more computationally efficient. We can give the example of the
# `LogisticRegressionCV` which can be used to find the best `C` in a more
# efficient way than what we previously did with the `GridSearchCV`.

# %%
from sklearn.linear_model import LogisticRegressionCV

# define the different Cs to try out
param_grid = {"C": (0.1, 1.0, 10.0)}

model = make_pipeline(
    preprocessor,
    LogisticRegressionCV(Cs=param_grid['C'], max_iter=1000,
                         solver='lbfgs', n_jobs=4, cv=5))
start = time.time()
model.fit(df_train, target_train)
elapsed_time = time.time() - start
print(f"Time elapsed to train LogisticRegressionCV: "
      f"{elapsed_time:.3f} seconds")

# %% [markdown]
# The `fit` time for the `CV` version of `LogisticRegression` gives a speed-up
# x2. This speed-up is provided by re-using the values of coefficients to
# warm-start the estimator for the different `C` values.

# %% [markdown]
# ## Exercises:
#
# - Build a machine learning pipeline:
#       * preprocess the categorical columns using an `OrdinalEncoder` and let
#         the numerical columns as they are.
#       * use an `HistGradientBoostingClassifier` as a predictive model.
# - Make an hyper-parameters search using `RandomizedSearchCV` and tuning the
#   parameters:
#       * `learning_rate` with values ranging from 0.001 to 0.5. You can use
#         an exponential distribution to sample the possible values.
#       * `l2_regularization` with values ranging from 0 to 0.5. You can use
#         a uniform distribution.
#       * `max_leaf_nodes` with values ranging from 5 to 30. The values should
#         be integer following a uniform distribution.
#       * `min_samples_leaf` with values ranging from 5 to 30. The values
#         should be integer following a uniform distribution.
#
# In case you have issues of with unknown categories, try to precompute the
# list of possible categories ahead of time and pass it explicitly to the
# constructor of the encoder:
#
# ```python
# categories = [data[column].unique()
#               for column in data[categorical_columns]]
# OrdinalEncoder(categories=categories)
# ```

# %% [markdown]
# ## Combining evaluation and hyper-parameters search
#
# Cross-validation was used for searching for the best model parameters. We
# previously evaluated model performance through cross-validation as well. If we
# would like to combine both aspects, we need to perform a "nested"
# cross-validation. The "outer" cross-validation is applied to assess the
# model while the "inner" cross-validation sets the hyper-parameters of the
# model on the data set provided by the "outer" cross-validation. In practice,
# it is equivalent to including, `GridSearchCV`, `RandomSearchCV`, or any
# `EstimatorCV` in a `cross_val_score` or `cross_validate` function call.

# %%
from sklearn.model_selection import cross_val_score

model = make_pipeline(
    preprocessor,
    LogisticRegressionCV(max_iter=1000, solver='lbfgs', cv=5))
score = cross_val_score(model, data, target, n_jobs=4, cv=5)
print(
    f"The accuracy score is: {score.mean():.2f} +- {score.std():.2f}"
)
print(f"The different scores obtained are: \n{score}")

# %% [markdown]
# Be aware that such training might involve a variation of the hyper-parameters
# of the model. When analyzing such model, you should not only look at the
# overall model performance but look at the hyper-parameters variations as
# well.
