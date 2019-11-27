# ---
# jupyter:
#   jupytext:
#     formats: python_scripts//py:percent,notebooks//ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.0
# ---

# %% [markdown]
# In this notebook, we will look at the necessary steps required before any machine learning takes place.
# * load the data
# * look at the variables in the dataset, in particular, differentiate
#   between numerical and categorical variables, which need different
#   preprocessing in most machine learning workflows
# * visualize the distribution of the variables to gain some insights into the dataset

# %%
# Inline plots
# %matplotlib inline

# plotting style
import seaborn as sns

# %% [markdown]
# ## Loading the adult census dataset

# %% [markdown]
# We will use data from the "Current Population adult_census" from 1994 that we
# downloaded from [OpenML](http://openml.org/).

# %%
import pandas as pd

adult_census = pd.read_csv(
    "https://www.openml.org/data/get_csv/1595261/adult-census.csv")

# Or use the local copy:
# adult_census = pd.read_csv('../datasets/adult-census.csv')

# %% [markdown]
# We can look at the OpenML webpage to know more about this dataset.

# %%
from IPython.display import IFrame
IFrame('https://www.openml.org/d/1590', width=1200, height=600)

# %% [markdown]
# ## Look at the variables in the dataset
# The data are stored in a pandas dataframe.

# %%
adult_census.head()

# %% [markdown]
# The column named **class** is our target variable (i.e., the variable which
# we want to predict). The two possible classes are `<= 50K` (low-revenue) and
# `> 50K` (high-revenue).

# %%
target_column = 'class'
adult_census[target_column].value_counts()

# %% [markdown]
# Note: classes are slightly imbalanced. Class imbalance happens often in
# practice and may need special techniques for machine learning. For example in
# a medical setting, if we are trying to predict whether patients will develop
# a rare disease, there will be a lot more healthy patients than ill patients in
# the dataset.

# %% [markdown]
# The dataset contains both numerical and categorical data. Numerical values
# can take continuous values for example `age`. Categorical values can have a
# finite number of values, for example `native-country`.

# %%
numerical_columns = [
    'age', 'education-num', 'capital-gain', 'capital-loss',
    'hours-per-week']
categorical_columns = [
    'workclass', 'education', 'marital-status', 'occupation',
    'relationship', 'race', 'sex', 'native-country']
all_columns = numerical_columns + categorical_columns + [
    target_column]

adult_census = adult_census[all_columns]

# %% [markdown]
# Note that for simplicity, we have ignored the "fnlwgt" (final weight) column
# that was crafted by the creators of the dataset when sampling the dataset to
# be representative of the full census database.

# %% [markdown]
# ## Inspect the data
# Before building a machine learning model, it is a good idea to look at the
# data:
# * maybe the task you are trying to achieve can be solved without machine
#   learning
# * you need to check that the data you need for your task is indeed present in
# the dataset
# * inspecting the data is a good way to find peculiarities. These can can
#   arise during data collection (for example, malfunctioning sensor or missing
#   values), or from the way the data is processed afterwards (for example capped
#   values).

# %% [markdown]
# Let's look at the distribution of individual variables, to get some insights
# about the data. We can start by plotting histograms, note that this only
# works for numerical variables:

# %%
_ = adult_census.hist(figsize=(20, 10))

# %% [markdown]
# We can already make a few comments about some of the variables:
# * age: there are not that many points for 'age > 70'. The dataset description
# does indicate that retired people have been filtered out (`hours-per-week > 0`).
# * education-num: peak at 10 and 13, hard to tell what it corresponds to
# without looking much further. We'll do that later in this notebook.
# * hours per week peaks at 40, this was very likely the standard number of
# working hours at the time of the data collection
# * most values of capital-gain and capital-loss are close to zero

# %% [markdown]
# For categorical variables, we can look at the distribution of values:


# %%
adult_census['sex'].value_counts()

# %%
adult_census['education'].value_counts()

# %% [markdown]
# `pandas_profiling` is a nice tool for inspecting the data (both numerical and
# categorical variables).

# %%
import pandas_profiling
adult_census.profile_report()

# %% [markdown]
# As noted above, `education-num` distribution has two clear peaks around 10
# and 13. It would be reasonable to expect that `education-num` is the number of
# years of education. Let's look at the relationship between `education` and
# `education-num`.
# %%
pd.crosstab(index=adult_census['education'],
            columns=adult_census['education-num'])

# %% [markdown]
# This shows that education and education-num gives you the same information.
# For example, `education-num=2` is equivalent to `education='1st-4th'`. In
# practice that means we can remove `education-num` without losing information.
# Note that having redundant (or highly correlated) columns can be a problem
# for machine learning algorithms.

# %% [markdown]
# Another way to inspect the data is to do a pairplot and show how each variable
# differs according to our target, `class`. Plots along the diagonal show the
# distribution of individual variables for each `class`. The plots on the
# off-diagonal can reveal interesting interactions between variables.

# %%
n_samples_to_plot = 5000
columns = ['age', 'education-num', 'hours-per-week']
_ = sns.pairplot(data=adult_census[:n_samples_to_plot], vars=columns,
                 hue=target_column, plot_kws={'alpha': 0.2},
                 height=4, diag_kind='hist')

# %%
_ = sns.pairplot(data=adult_census[:n_samples_to_plot], x_vars='age',
                 y_vars='h# ---
# jupyter:
#   jupytext:
#     formats: python_scripts//py:percent,notebooks//ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.0
# ---

# %% [markdown]
# In this notebook, we will look at the necessary steps required before any machine learning takes place.
# * load the data
# * look at the variables in the dataset, in particular, differentiate
#   between numerical and categorical variables, which need different
#   preprocessing in most machine learning workflows
# * visualize the distribution of the variables to gain some insights into the dataset

# %%
# Inline plots
# %matplotlib inline

# plotting style
import seaborn as sns

# %% [markdown]
# ## Loading the adult census dataset

# %% [markdown]
# We will use data from the "Current Population adult_census" from 1994 that we
# downloaded from [OpenML](http://openml.org/).

# %%
import pandas as pd

adult_census = pd.read_csv(
    "https://www.openml.org/data/get_csv/1595261/adult-census.csv")

# Or use the local copy:
# adult_census = pd.read_csv('../datasets/adult-census.csv')

# %% [markdown]
# We can look at the OpenML webpage to know more about this dataset.

# %%
from IPython.display import IFrame
IFrame('https://www.openml.org/d/1590', width=1200, height=600)

# %% [markdown]
# ## Look at the variables in the dataset
# The data are stored in a pandas dataframe.

# %%
adult_census.head()

# %% [markdown]
# The column named **class** is our target variable (i.e., the variable which
# we want to predict). The two possible classes are `<= 50K` (low-revenue) and
# `> 50K` (high-revenue).

# %%
target_column = 'class'
adult_census[target_column].value_counts()

# %% [markdown]
# Note: classes are slightly imbalanced. Class imbalance happens often in
# practice and may need special techniques for machine learning. For example in
# a medical setting, if we are trying to predict whether patients will develop
# a rare disease, there will be a lot more healthy patients than ill patients in
# the dataset.

# %% [markdown]
# The dataset contains both numerical and categorical data. Numerical values
# can take continuous values for example `age`. Categorical values can have a
# finite number of values, for example `native-country`.

# %%
numerical_columns = [
    'age', 'education-num', 'capital-gain', 'capital-loss',
    'hours-per-week']
categorical_columns = [
    'workclass', 'education', 'marital-status', 'occupation',
    'relationship', 'race', 'sex', 'native-country']
all_columns = numerical_columns + categorical_columns + [
    target_column]

adult_census = adult_census[all_columns]

# %% [markdown]
# Note that for simplicity, we have ignored the "fnlwgt" (final weight) column
# that was crafted by the creators of the dataset when sampling the dataset to
# be representative of the full census database.

# %% [markdown]
# ## Inspect the data
# Before building a machine learning model, it is a good idea to look at the
# data:
# * maybe the task you are trying to achieve can be solved without machine
#   learning
# * you need to check that the data you need for your task is indeed present in
# the dataset
# * inspecting the data is a good way to find peculiarities. These can can
#   arise during data collection (for example, malfunctioning sensor or missing
#   values), or from the way the data is processed afterwards (for example capped
#   values).

# %% [markdown]
# Let's look at the distribution of individual variables, to get some insights
# about the data. We can start by plotting histograms, note that this only
# works for numerical variables:

# %%
_ = adult_census.hist(figsize=(20, 10))

# %% [markdown]
# We can already make a few comments about some of the variables:
# * age: there are not that many points for 'age > 70'. The dataset description
# does indicate that retired people have been filtered out (`hours-per-week > 0`).
# * education-num: peak at 10 and 13, hard to tell what it corresponds to
# without looking much further. We'll do that later in this notebook.
# * hours per week peaks at 40, this was very likely the standard number of
# working hours at the time of the data collection
# * most values of capital-gain and capital-loss are close to zero

# %% [markdown]
# For categorical variables, we can look at the distribution of values:


# %%
adult_census['sex'].value_counts()

# %%
adult_census['education'].value_counts()

# %% [markdown]
# `pandas_profiling` is a nice tool for inspecting the data (both numerical and
# categorical variables).

# %%
import pandas_profiling
adult_census.profile_report()

# %% [markdown]
# As noted above, `education-num` distribution has two clear peaks around 10
# and 13. It would be reasonable to expect that `education-num` is the number of
# years of education. Let's look at the relationship between `education` and
# `education-num`.
# %%
pd.crosstab(index=adult_census['education'],
            columns=adult_census['education-num'])

# %% [markdown]
# This shows that education and education-num gives you the same information.
# For example, `education-num=2` is equivalent to `education='1st-4th'`. In
# practice that means we can remove `education-num` without losing information.
# Note that having redundant (or highly correlated) columns can be a problem
# for machine learning algorithms.

# %% [markdown]
# Another way to inspect the data is to do a pairplot and show how each variable
# differs according to our target, `class`. Plots along the diagonal show the
# distribution of individual variables for each `class`. The plots on the
# off-diagonal can reveal interesting interactions between variables.

# %%
n_samples_to_plot = 5000
columns = ['age', 'education-num', 'hours-per-week']
_ = sns.pairplot(data=adult_census[:n_samples_to_plot], vars=columns,
                 hue=target_column, plot_kws={'alpha': 0.2},
                 height=4, diag_kind='hist')

# %%
_ = sns.pairplot(data=adult_census[:n_samples_to_plot], x_vars='age',
                 y_vars='hours-per-week', hue=target_column,
                 markers=['o',
                          'v'], plot_kws={'alpha': 0.2}, height=12)

# %% [markdown]
#
# By looking at the data you could infer some hand-written rules to predict the
# class:
# * if you are young (less than 25 year-old roughly), you are in the `<= 50K` class.
# * if you are old (more than 70 year-old roughly), you are in the `<= 50K` class.
# * if you work part-time (less than 40 hours roughly) you are in the `<= 50K` class.
#
# These hand-written rules could work reasonably well without the need for any
# machine learning. Note however that it is not very easy to create rules for
# the region `40 < hours-per-week < 60` and `30 < age < 70`. We can hope that
# machine learning can help in this region. Also note that visualization can
# help creating hand-written rules but is limited to 2 dimensions (maybe 3
# dimensions), whereas machine learning models can build models in
# high-dimensional spaces.
#
# Another thing worth mentioning in this plot: if you are young (less than 25
# year-old roughly) or old (more than 70 year-old roughly) you tend to work
# less. This is a non-linear relationship between age and hours
# per week. Some machine learning models can only capture linear interactions so
# this may be a factor when deciding which model to chose.

# %% [markdown]
# In a machine-learning setting, we will use an algorithm to automatically
# decide what the "rules" should be in order to make predictions on new data.
# The following plot shows the rules a simple algorithm (called decision tree)
# creates:
#
# <img src="../figures/simple-decision-tree-adult-census.png" width=100%/>

# %% [markdown]
# * In the region `age < 28.5` (left region) the prediction is `low-income`. The blue color
#   indicates that the prediction is `low-income`. What is plotted is actually
#   the probability of the class `low-income` as predicted by the algorithm. So the dark
#   blue color indicates that the algorithm is quite sure about this prediction.
# * In the region `age > 28.5 AND hours-per-week < 40.5` (bottom-right region),
#   the prediction is `low-income`. Note that the blue is a bit lighter that for
#   the left region which means that the algorithm is not as certain in this
#   region.
# * In the region `age > 28.5 AND hours-per-week > 40.5` (top-right region),
#   the prediction is `low-income`. The probability of the class `low-income` is
#   very close to 0.5. In practice that means that the model is kind of saying "I
#   don't really know".

# %% [markdown]
# Comments: the decision tree was limited to have three regions. Also a model
# will be better in higher dimensions.

# %% [markdown]
#
# In this notebook we have:
# * loaded the data from a CSV file using `pandas`
# * looked at the kind of variables in the dataset, and differentiated
#   between categorical and numerical variables
# * inspected the data with `pandas`, `seaborn` and `pandas_profiling`. Data inspection
#   can allow you to decide whether using machine learning is appropriate for
#   your data and to highlight potential peculiarities in your data
#
# Key ideas discussed:
# * if your target variable is imbalanced (e.g., you have more samples from one
#   target category than another), you may need special techniques for machine
#   learning
# * having redundant (or highly correlated) columns can be a problem for
#   machine learning algorithms
# * some machine learning models can only capture linear interaction so be
#   aware of non-linear relationships in your dataours-per-week', hue=target_column,
                 markers=[# ---
# jupyter:
#   jupytext:
#     formats: python_scripts//py:percent,notebooks//ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.0
# ---

# %% [markdown]
# In this notebook, we will look at the necessary steps required before any machine learning takes place.
# * load the data
# * look at the variables in the dataset, in particular, differentiate
#   between numerical and categorical variables, which need different
#   preprocessing in most machine learning workflows
# * visualize the distribution of the variables to gain some insights into the dataset

# %%
# Inline plots
# %matplotlib inline

# plotting style
import seaborn as sns

# %% [markdown]
# ## Loading the adult census dataset

# %% [markdown]
# We will use data from the "Current Population adult_census" from 1994 that we
# downloaded from [OpenML](http://openml.org/).

# %%
import pandas as pd

adult_census = pd.read_csv(
    "https://www.openml.org/data/get_csv/1595261/adult-census.csv")

# Or use the local copy:
# adult_census = pd.read_csv('../datasets/adult-census.csv')

# %% [markdown]
# We can look at the OpenML webpage to know more about this dataset.

# %%
from IPython.display import IFrame
IFrame('https://www.openml.org/d/1590', width=1200, height=600)

# %% [markdown]
# ## Look at the variables in the dataset
# The data are stored in a pandas dataframe.

# %%
adult_census.head()

# %% [markdown]
# The column named **class** is our target variable (i.e., the variable which
# we want to predict). The two possible classes are `<= 50K` (low-revenue) and
# `> 50K` (high-revenue).

# %%
target_column = 'class'
adult_census[target_column].value_counts()

# %% [markdown]
# Note: classes are slightly imbalanced. Class imbalance happens often in
# practice and may need special techniques for machine learning. For example in
# a medical setting, if we are trying to predict whether patients will develop
# a rare disease, there will be a lot more healthy patients than ill patients in
# the dataset.

# %% [markdown]
# The dataset contains both numerical and categorical data. Numerical values
# can take continuous values for example `age`. Categorical values can have a
# finite number of values, for example `native-country`.

# %%
numerical_columns = [
    'age', 'education-num', 'capital-gain', 'capital-loss',
    'hours-per-week']
categorical_columns = [
    'workclass', 'education', 'marital-status', 'occupation',
    'relationship', 'race', 'sex', 'native-country']
all_columns = numerical_columns + categorical_columns + [
    target_column]

adult_census = adult_census[all_columns]

# %% [markdown]
# Note that for simplicity, we have ignored the "fnlwgt" (final weight) column
# that was crafted by the creators of the dataset when sampling the dataset to
# be representative of the full census database.

# %% [markdown]
# ## Inspect the data
# Before building a machine learning model, it is a good idea to look at the
# data:
# * maybe the task you are trying to achieve can be solved without machine
#   learning
# * you need to check that the data you need for your task is indeed present in
# the dataset
# * inspecting the data is a good way to find peculiarities. These can can
#   arise during data collection (for example, malfunctioning sensor or missing
#   values), or from the way the data is processed afterwards (for example capped
#   values).

# %% [markdown]
# Let's look at the distribution of individual variables, to get some insights
# about the data. We can start by plotting histograms, note that this only
# works for numerical variables:

# %%
_ = adult_census.hist(figsize=(20, 10))

# %% [markdown]
# We can already make a few comments about some of the variables:
# * age: there are not that many points for 'age > 70'. The dataset description
# does indicate that retired people have been filtered out (`hours-per-week > 0`).
# * education-num: peak at 10 and 13, hard to tell what it corresponds to
# without looking much further. We'll do that later in this notebook.
# * hours per week peaks at 40, this was very likely the standard number of
# working hours at the time of the data collection
# * most values of capital-gain and capital-loss are close to zero

# %% [markdown]
# For categorical variables, we can look at the distribution of values:


# %%
adult_census['sex'].value_counts()

# %%
adult_census['education'].value_counts()

# %% [markdown]
# `pandas_profiling` is a nice tool for inspecting the data (both numerical and
# categorical variables).

# %%
import pandas_profiling
adult_census.profile_report()

# %% [markdown]
# As noted above, `education-num` distribution has two clear peaks around 10
# and 13. It would be reasonable to expect that `education-num` is the number of
# years of education. Let's look at the relationship between `education` and
# `education-num`.
# %%
pd.crosstab(index=adult_census['education'],
            columns=adult_census['education-num'])

# %% [markdown]
# This shows that education and education-num gives you the same information.
# For example, `education-num=2` is equivalent to `education='1st-4th'`. In
# practice that means we can remove `education-num` without losing information.
# Note that having redundant (or highly correlated) columns can be a problem
# for machine learning algorithms.

# %% [markdown]
# Another way to inspect the data is to do a pairplot and show how each variable
# differs according to our target, `class`. Plots along the diagonal show the
# distribution of individual variables for each `class`. The plots on the
# off-diagonal can reveal interesting interactions between variables.

# %%
n_samples_to_plot = 5000
columns = ['age', 'education-num', 'hours-per-week']
_ = sns.pairplot(data=adult_census[:n_samples_to_plot], vars=columns,
                 hue=target_column, plot_kws={'alpha': 0.2},
                 height=4, diag_kind='hist')

# %%
_ = sns.pairplot(data=adult_census[:n_samples_to_plot], x_vars='age',
                 y_vars='hours-per-week', hue=target_column,
                 markers=['o',
                          'v'], plot_kws={'alpha': 0.2}, height=12)

# %% [markdown]
#
# By looking at the data you could infer some hand-written rules to predict the
# class:
# * if you are young (less than 25 year-old roughly), you are in the `<= 50K` class.
# * if you are old (more than 70 year-old roughly), you are in the `<= 50K` class.
# * if you work part-time (less than 40 hours roughly) you are in the `<= 50K` class.
#
# These hand-written rules could work reasonably well without the need for any
# machine learning. Note however that it is not very easy to create rules for
# the region `40 < hours-per-week < 60` and `30 < age < 70`. We can hope that
# machine learning can help in this region. Also note that visualization can
# help creating hand-written rules but is limited to 2 dimensions (maybe 3
# dimensions), whereas machine learning models can build models in
# high-dimensional spaces.
#
# Another thing worth mentioning in this plot: if you are young (less than 25
# year-old roughly) or old (more than 70 year-old roughly) you tend to work
# less. This is a non-linear relationship between age and hours
# per week. Some machine learning models can only capture linear interactions so
# this may be a factor when deciding which model to chose.

# %% [markdown]
# In a machine-learning setting, we will use an algorithm to automatically
# decide what the "rules" should be in order to make predictions on new data.
# The following plot shows the rules a simple algorithm (called decision tree)
# creates:
#
# <img src="../figures/simple-decision-tree-adult-census.png" width=100%/>

# %% [markdown]
# * In the region `age < 28.5` (left region) the prediction is `low-income`. The blue color
#   indicates that the prediction is `low-income`. What is plotted is actually
#   the probability of the class `low-income` as predicted by the algorithm. So the dark
#   blue color indicates that the algorithm is quite sure about this prediction.
# * In the region `age > 28.5 AND hours-per-week < 40.5` (bottom-right region),
#   the prediction is `low-income`. Note that the blue is a bit lighter that for
#   the left region which means that the algorithm is not as certain in this
#   region.
# * In the region `age > 28.5 AND hours-per-week > 40.5` (top-right region),
#   the prediction is `low-income`. The probability of the class `low-income` is
#   very close to 0.5. In practice that means that the model is kind of saying "I
#   don't really know".

# %% [markdown]
# Comments: the decision tree was limited to have three regions. Also a model
# will be better in higher dimensions.

# %% [markdown]
#
# In this notebook we have:
# * loaded the data from a CSV file using `pandas`
# * looked at the kind of variables in the dataset, and differentiated
#   between categorical and numerical variables
# * inspected the data with `pandas`, `seaborn` and `pandas_profiling`. Data inspection
#   can allow you to decide whether using machine learning is appropriate for
#   your data and to highlight potential peculiarities in your data
#
# Key ideas discussed:
# * if your target variable is imbalanced (e.g., you have more samples from one
#   target category than another), you may need special techniques for machine
#   learning
# * having redundant (or highly correlated) columns can be a problem for
#   machine learning algorithms
# * some machine learning models can only capture linear interaction so be
#   aware of non-linear relationships in your data'o',
                          # ---
# jupyter:
#   jupytext:
#     formats: python_scripts//py:percent,notebooks//ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.0
# ---

# %% [markdown]
# In this notebook, we will look at the necessary steps required before any machine learning takes place.
# * load the data
# * look at the variables in the dataset, in particular, differentiate
#   between numerical and categorical variables, which need different
#   preprocessing in most machine learning workflows
# * visualize the distribution of the variables to gain some insights into the dataset

# %%
# Inline plots
# %matplotlib inline

# plotting style
import seaborn as sns

# %% [markdown]
# ## Loading the adult census dataset

# %% [markdown]
# We will use data from the "Current Population adult_census" from 1994 that we
# downloaded from [OpenML](http://openml.org/).

# %%
import pandas as pd

adult_census = pd.read_csv(
    "https://www.openml.org/data/get_csv/1595261/adult-census.csv")

# Or use the local copy:
# adult_census = pd.read_csv('../datasets/adult-census.csv')

# %% [markdown]
# We can look at the OpenML webpage to know more about this dataset.

# %%
from IPython.display import IFrame
IFrame('https://www.openml.org/d/1590', width=1200, height=600)

# %% [markdown]
# ## Look at the variables in the dataset
# The data are stored in a pandas dataframe.

# %%
adult_census.head()

# %% [markdown]
# The column named **class** is our target variable (i.e., the variable which
# we want to predict). The two possible classes are `<= 50K` (low-revenue) and
# `> 50K` (high-revenue).

# %%
target_column = 'class'
adult_census[target_column].value_counts()

# %% [markdown]
# Note: classes are slightly imbalanced. Class imbalance happens often in
# practice and may need special techniques for machine learning. For example in
# a medical setting, if we are trying to predict whether patients will develop
# a rare disease, there will be a lot more healthy patients than ill patients in
# the dataset.

# %% [markdown]
# The dataset contains both numerical and categorical data. Numerical values
# can take continuous values for example `age`. Categorical values can have a
# finite number of values, for example `native-country`.

# %%
numerical_columns = [
    'age', 'education-num', 'capital-gain', 'capital-loss',
    'hours-per-week']
categorical_columns = [
    'workclass', 'education', 'marital-status', 'occupation',
    'relationship', 'race', 'sex', 'native-country']
all_columns = numerical_columns + categorical_columns + [
    target_column]

adult_census = adult_census[all_columns]

# %% [markdown]
# Note that for simplicity, we have ignored the "fnlwgt" (final weight) column
# that was crafted by the creators of the dataset when sampling the dataset to
# be representative of the full census database.

# %% [markdown]
# ## Inspect the data
# Before building a machine learning model, it is a good idea to look at the
# data:
# * maybe the task you are trying to achieve can be solved without machine
#   learning
# * you need to check that the data you need for your task is indeed present in
# the dataset
# * inspecting the data is a good way to find peculiarities. These can can
#   arise during data collection (for example, malfunctioning sensor or missing
#   values), or from the way the data is processed afterwards (for example capped
#   values).

# %% [markdown]
# Let's look at the distribution of individual variables, to get some insights
# about the data. We can start by plotting histograms, note that this only
# works for numerical variables:

# %%
_ = adult_census.hist(figsize=(20, 10))

# %% [markdown]
# We can already make a few comments about some of the variables:
# * age: there are not that many points for 'age > 70'. The dataset description
# does indicate that retired people have been filtered out (`hours-per-week > 0`).
# * education-num: peak at 10 and 13, hard to tell what it corresponds to
# without looking much further. We'll do that later in this notebook.
# * hours per week peaks at 40, this was very likely the standard number of
# working hours at the time of the data collection
# * most values of capital-gain and capital-loss are close to zero

# %% [markdown]
# For categorical variables, we can look at the distribution of values:


# %%
adult_census['sex'].value_counts()

# %%
adult_census['education'].value_counts()

# %% [markdown]
# `pandas_profiling` is a nice tool for inspecting the data (both numerical and
# categorical variables).

# %%
import pandas_profiling
adult_census.profile_report()

# %% [markdown]
# As noted above, `education-num` distribution has two clear peaks around 10
# and 13. It would be reasonable to expect that `education-num` is the number of
# years of education. Let's look at the relationship between `education` and
# `education-num`.
# %%
pd.crosstab(index=adult_census['education'],
            columns=adult_census['education-num'])

# %% [markdown]
# This shows that education and education-num gives you the same information.
# For example, `education-num=2` is equivalent to `education='1st-4th'`. In
# practice that means we can remove `education-num` without losing information.
# Note that having redundant (or highly correlated) columns can be a problem
# for machine learning algorithms.

# %% [markdown]
# Another way to inspect the data is to do a pairplot and show how each variable
# differs according to our target, `class`. Plots along the diagonal show the
# distribution of individual variables for each `class`. The plots on the
# off-diagonal can reveal interesting interactions between variables.

# %%
n_samples_to_plot = 5000
columns = ['age', 'education-num', 'hours-per-week']
_ = sns.pairplot(data=adult_census[:n_samples_to_plot], vars=columns,
                 hue=target_column, plot_kws={'alpha': 0.2},
                 height=4, diag_kind='hist')

# %%
_ = sns.pairplot(data=adult_census[:n_samples_to_plot], x_vars='age',
                 y_vars='hours-per-week', hue=target_column,
                 markers=['o',
                          'v'], plot_kws={'alpha': 0.2}, height=12)

# %% [markdown]
#
# By looking at the data you could infer some hand-written rules to predict the
# class:
# * if you are young (less than 25 year-old roughly), you are in the `<= 50K` class.
# * if you are old (more than 70 year-old roughly), you are in the `<= 50K` class.
# * if you work part-time (less than 40 hours roughly) you are in the `<= 50K` class.
#
# These hand-written rules could work reasonably well without the need for any
# machine learning. Note however that it is not very easy to create rules for
# the region `40 < hours-per-week < 60` and `30 < age < 70`. We can hope that
# machine learning can help in this region. Also note that visualization can
# help creating hand-written rules but is limited to 2 dimensions (maybe 3
# dimensions), whereas machine learning models can build models in
# high-dimensional spaces.
#
# Another thing worth mentioning in this plot: if you are young (less than 25
# year-old roughly) or old (more than 70 year-old roughly) you tend to work
# less. This is a non-linear relationship between age and hours
# per week. Some machine learning models can only capture linear interactions so
# this may be a factor when deciding which model to chose.

# %% [markdown]
# In a machine-learning setting, we will use an algorithm to automatically
# decide what the "rules" should be in order to make predictions on new data.
# The following plot shows the rules a simple algorithm (called decision tree)
# creates:
#
# <img src="../figures/simple-decision-tree-adult-census.png" width=100%/>

# %% [markdown]
# * In the region `age < 28.5` (left region) the prediction is `low-income`. The blue color
#   indicates that the prediction is `low-income`. What is plotted is actually
#   the probability of the class `low-income` as predicted by the algorithm. So the dark
#   blue color indicates that the algorithm is quite sure about this prediction.
# * In the region `age > 28.5 AND hours-per-week < 40.5` (bottom-right region),
#   the prediction is `low-income`. Note that the blue is a bit lighter that for
#   the left region which means that the algorithm is not as certain in this
#   region.
# * In the region `age > 28.5 AND hours-per-week > 40.5` (top-right region),
#   the prediction is `low-income`. The probability of the class `low-income` is
#   very close to 0.5. In practice that means that the model is kind of saying "I
#   don't really know".

# %% [markdown]
# Comments: the decision tree was limited to have three regions. Also a model
# will be better in higher dimensions.

# %% [markdown]
#
# In this notebook we have:
# * loaded the data from a CSV file using `pandas`
# * looked at the kind of variables in the dataset, and differentiated
#   between categorical and numerical variables
# * inspected the data with `pandas`, `seaborn` and `pandas_profiling`. Data inspection
#   can allow you to decide whether using machine learning is appropriate for
#   your data and to highlight potential peculiarities in your data
#
# Key ideas discussed:
# * if your target variable is imbalanced (e.g., you have more samples from one
#   target category than another), you may need special techniques for machine
#   learning
# * having redundant (or highly correlated) columns can be a problem for
#   machine learning algorithms
# * some machine learning models can only capture linear interaction so be
#   aware of non-linear relationships in your data'v'], plot_kws={'alpha': 0.2}, height=12)
# ---
# jupyter:
#   jupytext:
#     formats: python_scripts//py:percent,notebooks//ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.0
# ---

# %% [markdown]
# In this notebook, we will look at the necessary steps required before any machine learning takes place.
# * load the data
# * look at the variables in the dataset, in particular, differentiate
#   between numerical and categorical variables, which need different
#   preprocessing in most machine learning workflows
# * visualize the distribution of the variables to gain some insights into the dataset

# %%
# Inline plots
# %matplotlib inline

# plotting style
import seaborn as sns

# %% [markdown]
# ## Loading the adult census dataset

# %% [markdown]
# We will use data from the "Current Population adult_census" from 1994 that we
# downloaded from [OpenML](http://openml.org/).

# %%
import pandas as pd

adult_census = pd.read_csv(
    "https://www.openml.org/data/get_csv/1595261/adult-census.csv")

# Or use the local copy:
# adult_census = pd.read_csv('../datasets/adult-census.csv')

# %% [markdown]
# We can look at the OpenML webpage to know more about this dataset.

# %%
from IPython.display import IFrame
IFrame('https://www.openml.org/d/1590', width=1200, height=600)

# %% [markdown]
# ## Look at the variables in the dataset
# The data are stored in a pandas dataframe.

# %%
adult_census.head()

# %% [markdown]
# The column named **class** is our target variable (i.e., the variable which
# we want to predict). The two possible classes are `<= 50K` (low-revenue) and
# `> 50K` (high-revenue).

# %%
target_column = 'class'
adult_census[target_column].value_counts()

# %% [markdown]
# Note: classes are slightly imbalanced. Class imbalance happens often in
# practice and may need special techniques for machine learning. For example in
# a medical setting, if we are trying to predict whether patients will develop
# a rare disease, there will be a lot more healthy patients than ill patients in
# the dataset.

# %% [markdown]
# The dataset contains both numerical and categorical data. Numerical values
# can take continuous values for example `age`. Categorical values can have a
# finite number of values, for example `native-country`.

# %%
numerical_columns = [
    'age', 'education-num', 'capital-gain', 'capital-loss',
    'hours-per-week']
categorical_columns = [
    'workclass', 'education', 'marital-status', 'occupation',
    'relationship', 'race', 'sex', 'native-country']
all_columns = numerical_columns + categorical_columns + [
    target_column]

adult_census = adult_census[all_columns]

# %% [markdown]
# Note that for simplicity, we have ignored the "fnlwgt" (final weight) column
# that was crafted by the creators of the dataset when sampling the dataset to
# be representative of the full census database.

# %% [markdown]
# ## Inspect the data
# Before building a machine learning model, it is a good idea to look at the
# data:
# * maybe the task you are trying to achieve can be solved without machine
#   learning
# * you need to check that the data you need for your task is indeed present in
# the dataset
# * inspecting the data is a good way to find peculiarities. These can can
#   arise during data collection (for example, malfunctioning sensor or missing
#   values), or from the way the data is processed afterwards (for example capped
#   values).

# %% [markdown]
# Let's look at the distribution of individual variables, to get some insights
# about the data. We can start by plotting histograms, note that this only
# works for numerical variables:

# %%
_ = adult_census.hist(figsize=(20, 10))

# %% [markdown]
# We can already make a few comments about some of the variables:
# * age: there are not that many points for 'age > 70'. The dataset description
# does indicate that retired people have been filtered out (`hours-per-week > 0`).
# * education-num: peak at 10 and 13, hard to tell what it corresponds to
# without looking much further. We'll do that later in this notebook.
# * hours per week peaks at 40, this was very likely the standard number of
# working hours at the time of the data collection
# * most values of capital-gain and capital-loss are close to zero

# %% [markdown]
# For categorical variables, we can look at the distribution of values:


# %%
adult_census['sex'].value_counts()

# %%
adult_census['education'].value_counts()

# %% [markdown]
# `pandas_profiling` is a nice tool for inspecting the data (both numerical and
# categorical variables).

# %%
import pandas_profiling
adult_census.profile_report()

# %% [markdown]
# As noted above, `education-num` distribution has two clear peaks around 10
# and 13. It would be reasonable to expect that `education-num` is the number of
# years of education. Let's look at the relationship between `education` and
# `education-num`.
# %%
pd.crosstab(index=adult_census['education'],
            columns=adult_census['education-num'])

# %% [markdown]
# This shows that education and education-num gives you the same information.
# For example, `education-num=2` is equivalent to `education='1st-4th'`. In
# practice that means we can remove `education-num` without losing information.
# Note that having redundant (or highly correlated) columns can be a problem
# for machine learning algorithms.

# %% [markdown]
# Another way to inspect the data is to do a pairplot and show how each variable
# differs according to our target, `class`. Plots along the diagonal show the
# distribution of individual variables for each `class`. The plots on the
# off-diagonal can reveal interesting interactions between variables.

# %%
n_samples_to_plot = 5000
columns = ['age', 'education-num', 'hours-per-week']
_ = sns.pairplot(data=adult_census[:n_samples_to_plot], vars=columns,
                 hue=target_column, plot_kws={'alpha': 0.2},
                 height=4, diag_kind='hist')

# %%
_ = sns.pairplot(data=adult_census[:n_samples_to_plot], x_vars='age',
                 y_vars='hours-per-week', hue=target_column,
                 markers=['o',
                          'v'], plot_kws={'alpha': 0.2}, height=12)

# %% [markdown]
#
# By looking at the data you could infer some hand-written rules to predict the
# class:
# * if you are young (less than 25 year-old roughly), you are in the `<= 50K` class.
# * if you are old (more than 70 year-old roughly), you are in the `<= 50K` class.
# * if you work part-time (less than 40 hours roughly) you are in the `<= 50K` class.
#
# These hand-written rules could work reasonably well without the need for any
# machine learning. Note however that it is not very easy to create rules for
# the region `40 < hours-per-week < 60` and `30 < age < 70`. We can hope that
# machine learning can help in this region. Also note that visualization can
# help creating hand-written rules but is limited to 2 dimensions (maybe 3
# dimensions), whereas machine learning models can build models in
# high-dimensional spaces.
#
# Another thing worth mentioning in this plot: if you are young (less than 25
# year-old roughly) or old (more than 70 year-old roughly) you tend to work
# less. This is a non-linear relationship between age and hours
# per week. Some machine learning models can only capture linear interactions so
# this may be a factor when deciding which model to chose.

# %% [markdown]
# In a machine-learning setting, we will use an algorithm to automatically
# decide what the "rules" should be in order to make predictions on new data.
# The following plot shows the rules a simple algorithm (called decision tree)
# creates:
#
# <img src="../figures/simple-decision-tree-adult-census.png" width=100%/>

# %% [markdown]
# * In the region `age < 28.5` (left region) the prediction is `low-income`. The blue color
#   indicates that the prediction is `low-income`. What is plotted is actually
#   the probability of the class `low-income` as predicted by the algorithm. So the dark
#   blue color indicates that the algorithm is quite sure about this prediction.
# * In the region `age > 28.5 AND hours-per-week < 40.5` (bottom-right region),
#   the prediction is `low-income`. Note that the blue is a bit lighter that for
#   the left region which means that the algorithm is not as certain in this
#   region.
# * In the region `age > 28.5 AND hours-per-week > 40.5` (top-right region),
#   the prediction is `low-income`. The probability of the class `low-income` is
#   very close to 0.5. In practice that means that the model is kind of saying "I
#   don't really know".

# %% [markdown]
# Comments: the decision tree was limited to have three regions. Also a model
# will be better in higher dimensions.

# %% [markdown]
#
# In this notebook we have:
# * loaded the data from a CSV file using `pandas`
# * looked at the kind of variables in the dataset, and differentiated
#   between categorical and numerical variables
# * inspected the data with `pandas`, `seaborn` and `pandas_profiling`. Data inspection
#   can allow you to decide whether using machine learning is appropriate for
#   your data and to highlight potential peculiarities in your data
#
# Key ideas discussed:
# * if your target variable is imbalanced (e.g., you have more samples from one
#   target category than another), you may need special techniques for machine
#   learning
# * having redundant (or highly correlated) columns can be a problem for
#   machine learning algorithms
# * some machine learning models can only capture linear interaction so be
#   aware of non-linear relationships in your data
# %% [markdown]# ---
# jupyter:
#   jupytext:
#     formats: python_scripts//py:percent,notebooks//ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.0
# ---

# %% [markdown]
# In this notebook, we will look at the necessary steps required before any machine learning takes place.
# * load the data
# * look at the variables in the dataset, in particular, differentiate
#   between numerical and categorical variables, which need different
#   preprocessing in most machine learning workflows
# * visualize the distribution of the variables to gain some insights into the dataset

# %%
# Inline plots
# %matplotlib inline

# plotting style
import seaborn as sns

# %% [markdown]
# ## Loading the adult census dataset

# %% [markdown]
# We will use data from the "Current Population adult_census" from 1994 that we
# downloaded from [OpenML](http://openml.org/).

# %%
import pandas as pd

adult_census = pd.read_csv(
    "https://www.openml.org/data/get_csv/1595261/adult-census.csv")

# Or use the local copy:
# adult_census = pd.read_csv('../datasets/adult-census.csv')

# %% [markdown]
# We can look at the OpenML webpage to know more about this dataset.

# %%
from IPython.display import IFrame
IFrame('https://www.openml.org/d/1590', width=1200, height=600)

# %% [markdown]
# ## Look at the variables in the dataset
# The data are stored in a pandas dataframe.

# %%
adult_census.head()

# %% [markdown]
# The column named **class** is our target variable (i.e., the variable which
# we want to predict). The two possible classes are `<= 50K` (low-revenue) and
# `> 50K` (high-revenue).

# %%
target_column = 'class'
adult_census[target_column].value_counts()

# %% [markdown]
# Note: classes are slightly imbalanced. Class imbalance happens often in
# practice and may need special techniques for machine learning. For example in
# a medical setting, if we are trying to predict whether patients will develop
# a rare disease, there will be a lot more healthy patients than ill patients in
# the dataset.

# %% [markdown]
# The dataset contains both numerical and categorical data. Numerical values
# can take continuous values for example `age`. Categorical values can have a
# finite number of values, for example `native-country`.

# %%
numerical_columns = [
    'age', 'education-num', 'capital-gain', 'capital-loss',
    'hours-per-week']
categorical_columns = [
    'workclass', 'education', 'marital-status', 'occupation',
    'relationship', 'race', 'sex', 'native-country']
all_columns = numerical_columns + categorical_columns + [
    target_column]

adult_census = adult_census[all_columns]

# %% [markdown]
# Note that for simplicity, we have ignored the "fnlwgt" (final weight) column
# that was crafted by the creators of the dataset when sampling the dataset to
# be representative of the full census database.

# %% [markdown]
# ## Inspect the data
# Before building a machine learning model, it is a good idea to look at the
# data:
# * maybe the task you are trying to achieve can be solved without machine
#   learning
# * you need to check that the data you need for your task is indeed present in
# the dataset
# * inspecting the data is a good way to find peculiarities. These can can
#   arise during data collection (for example, malfunctioning sensor or missing
#   values), or from the way the data is processed afterwards (for example capped
#   values).

# %% [markdown]
# Let's look at the distribution of individual variables, to get some insights
# about the data. We can start by plotting histograms, note that this only
# works for numerical variables:

# %%
_ = adult_census.hist(figsize=(20, 10))

# %% [markdown]
# We can already make a few comments about some of the variables:
# * age: there are not that many points for 'age > 70'. The dataset description
# does indicate that retired people have been filtered out (`hours-per-week > 0`).
# * education-num: peak at 10 and 13, hard to tell what it corresponds to
# without looking much further. We'll do that later in this notebook.
# * hours per week peaks at 40, this was very likely the standard number of
# working hours at the time of the data collection
# * most values of capital-gain and capital-loss are close to zero

# %% [markdown]
# For categorical variables, we can look at the distribution of values:


# %%
adult_census['sex'].value_counts()

# %%
adult_census['education'].value_counts()

# %% [markdown]
# `pandas_profiling` is a nice tool for inspecting the data (both numerical and
# categorical variables).

# %%
import pandas_profiling
adult_census.profile_report()

# %% [markdown]
# As noted above, `education-num` distribution has two clear peaks around 10
# and 13. It would be reasonable to expect that `education-num` is the number of
# years of education. Let's look at the relationship between `education` and
# `education-num`.
# %%
pd.crosstab(index=adult_census['education'],
            columns=adult_census['education-num'])

# %% [markdown]
# This shows that education and education-num gives you the same information.
# For example, `education-num=2` is equivalent to `education='1st-4th'`. In
# practice that means we can remove `education-num` without losing information.
# Note that having redundant (or highly correlated) columns can be a problem
# for machine learning algorithms.

# %% [markdown]
# Another way to inspect the data is to do a pairplot and show how each variable
# differs according to our target, `class`. Plots along the diagonal show the
# distribution of individual variables for each `class`. The plots on the
# off-diagonal can reveal interesting interactions between variables.

# %%
n_samples_to_plot = 5000
columns = ['age', 'education-num', 'hours-per-week']
_ = sns.pairplot(data=adult_census[:n_samples_to_plot], vars=columns,
                 hue=target_column, plot_kws={'alpha': 0.2},
                 height=4, diag_kind='hist')

# %%
_ = sns.pairplot(data=adult_census[:n_samples_to_plot], x_vars='age',
                 y_vars='hours-per-week', hue=target_column,
                 markers=['o',
                          'v'], plot_kws={'alpha': 0.2}, height=12)

# %% [markdown]
#
# By looking at the data you could infer some hand-written rules to predict the
# class:
# * if you are young (less than 25 year-old roughly), you are in the `<= 50K` class.
# * if you are old (more than 70 year-old roughly), you are in the `<= 50K` class.
# * if you work part-time (less than 40 hours roughly) you are in the `<= 50K` class.
#
# These hand-written rules could work reasonably well without the need for any
# machine learning. Note however that it is not very easy to create rules for
# the region `40 < hours-per-week < 60` and `30 < age < 70`. We can hope that
# machine learning can help in this region. Also note that visualization can
# help creating hand-written rules but is limited to 2 dimensions (maybe 3
# dimensions), whereas machine learning models can build models in
# high-dimensional spaces.
#
# Another thing worth mentioning in this plot: if you are young (less than 25
# year-old roughly) or old (more than 70 year-old roughly) you tend to work
# less. This is a non-linear relationship between age and hours
# per week. Some machine learning models can only capture linear interactions so
# this may be a factor when deciding which model to chose.

# %% [markdown]
# In a machine-learning setting, we will use an algorithm to automatically
# decide what the "rules" should be in order to make predictions on new data.
# The following plot shows the rules a simple algorithm (called decision tree)
# creates:
#
# <img src="../figures/simple-decision-tree-adult-census.png" width=100%/>

# %% [markdown]
# * In the region `age < 28.5` (left region) the prediction is `low-income`. The blue color
#   indicates that the prediction is `low-income`. What is plotted is actually
#   the probability of the class `low-income` as predicted by the algorithm. So the dark
#   blue color indicates that the algorithm is quite sure about this prediction.
# * In the region `age > 28.5 AND hours-per-week < 40.5` (bottom-right region),
#   the prediction is `low-income`. Note that the blue is a bit lighter that for
#   the left region which means that the algorithm is not as certain in this
#   region.
# * In the region `age > 28.5 AND hours-per-week > 40.5` (top-right region),
#   the prediction is `low-income`. The probability of the class `low-income` is
#   very close to 0.5. In practice that means that the model is kind of saying "I
#   don't really know".

# %% [markdown]
# Comments: the decision tree was limited to have three regions. Also a model
# will be better in higher dimensions.

# %% [markdown]
#
# In this notebook we have:
# * loaded the data from a CSV file using `pandas`
# * looked at the kind of variables in the dataset, and differentiated
#   between categorical and numerical variables
# * inspected the data with `pandas`, `seaborn` and `pandas_profiling`. Data inspection
#   can allow you to decide whether using machine learning is appropriate for
#   your data and to highlight potential peculiarities in your data
#
# Key ideas discussed:
# * if your target variable is imbalanced (e.g., you have more samples from one
#   target category than another), you may need special techniques for machine
#   learning
# * having redundant (or highly correlated) columns can be a problem for
#   machine learning algorithms
# * some machine learning models can only capture linear interaction so be
#   aware of non-linear relationships in your data
## ---
# jupyter:
#   jupytext:
#     formats: python_scripts//py:percent,notebooks//ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.0
# ---

# %% [markdown]
# In this notebook, we will look at the necessary steps required before any machine learning takes place.
# * load the data
# * look at the variables in the dataset, in particular, differentiate
#   between numerical and categorical variables, which need different
#   preprocessing in most machine learning workflows
# * visualize the distribution of the variables to gain some insights into the dataset

# %%
# Inline plots
# %matplotlib inline

# plotting style
import seaborn as sns

# %% [markdown]
# ## Loading the adult census dataset

# %% [markdown]
# We will use data from the "Current Population adult_census" from 1994 that we
# downloaded from [OpenML](http://openml.org/).

# %%
import pandas as pd

adult_census = pd.read_csv(
    "https://www.openml.org/data/get_csv/1595261/adult-census.csv")

# Or use the local copy:
# adult_census = pd.read_csv('../datasets/adult-census.csv')

# %% [markdown]
# We can look at the OpenML webpage to know more about this dataset.

# %%
from IPython.display import IFrame
IFrame('https://www.openml.org/d/1590', width=1200, height=600)

# %% [markdown]
# ## Look at the variables in the dataset
# The data are stored in a pandas dataframe.

# %%
adult_census.head()

# %% [markdown]
# The column named **class** is our target variable (i.e., the variable which
# we want to predict). The two possible classes are `<= 50K` (low-revenue) and
# `> 50K` (high-revenue).

# %%
target_column = 'class'
adult_census[target_column].value_counts()

# %% [markdown]
# Note: classes are slightly imbalanced. Class imbalance happens often in
# practice and may need special techniques for machine learning. For example in
# a medical setting, if we are trying to predict whether patients will develop
# a rare disease, there will be a lot more healthy patients than ill patients in
# the dataset.

# %% [markdown]
# The dataset contains both numerical and categorical data. Numerical values
# can take continuous values for example `age`. Categorical values can have a
# finite number of values, for example `native-country`.

# %%
numerical_columns = [
    'age', 'education-num', 'capital-gain', 'capital-loss',
    'hours-per-week']
categorical_columns = [
    'workclass', 'education', 'marital-status', 'occupation',
    'relationship', 'race', 'sex', 'native-country']
all_columns = numerical_columns + categorical_columns + [
    target_column]

adult_census = adult_census[all_columns]

# %% [markdown]
# Note that for simplicity, we have ignored the "fnlwgt" (final weight) column
# that was crafted by the creators of the dataset when sampling the dataset to
# be representative of the full census database.

# %% [markdown]
# ## Inspect the data
# Before building a machine learning model, it is a good idea to look at the
# data:
# * maybe the task you are trying to achieve can be solved without machine
#   learning
# * you need to check that the data you need for your task is indeed present in
# the dataset
# * inspecting the data is a good way to find peculiarities. These can can
#   arise during data collection (for example, malfunctioning sensor or missing
#   values), or from the way the data is processed afterwards (for example capped
#   values).

# %% [markdown]
# Let's look at the distribution of individual variables, to get some insights
# about the data. We can start by plotting histograms, note that this only
# works for numerical variables:

# %%
_ = adult_census.hist(figsize=(20, 10))

# %% [markdown]
# We can already make a few comments about some of the variables:
# * age: there are not that many points for 'age > 70'. The dataset description
# does indicate that retired people have been filtered out (`hours-per-week > 0`).
# * education-num: peak at 10 and 13, hard to tell what it corresponds to
# without looking much further. We'll do that later in this notebook.
# * hours per week peaks at 40, this was very likely the standard number of
# working hours at the time of the data collection
# * most values of capital-gain and capital-loss are close to zero

# %% [markdown]
# For categorical variables, we can look at the distribution of values:


# %%
adult_census['sex'].value_counts()

# %%
adult_census['education'].value_counts()

# %% [markdown]
# `pandas_profiling` is a nice tool for inspecting the data (both numerical and
# categorical variables).

# %%
import pandas_profiling
adult_census.profile_report()

# %% [markdown]
# As noted above, `education-num` distribution has two clear peaks around 10
# and 13. It would be reasonable to expect that `education-num` is the number of
# years of education. Let's look at the relationship between `education` and
# `education-num`.
# %%
pd.crosstab(index=adult_census['education'],
            columns=adult_census['education-num'])

# %% [markdown]
# This shows that education and education-num gives you the same information.
# For example, `education-num=2` is equivalent to `education='1st-4th'`. In
# practice that means we can remove `education-num` without losing information.
# Note that having redundant (or highly correlated) columns can be a problem
# for machine learning algorithms.

# %% [markdown]
# Another way to inspect the data is to do a pairplot and show how each variable
# differs according to our target, `class`. Plots along the diagonal show the
# distribution of individual variables for each `class`. The plots on the
# off-diagonal can reveal interesting interactions between variables.

# %%
n_samples_to_plot = 5000
columns = ['age', 'education-num', 'hours-per-week']
_ = sns.pairplot(data=adult_census[:n_samples_to_plot], vars=columns,
                 hue=target_column, plot_kws={'alpha': 0.2},
                 height=4, diag_kind='hist')

# %%
_ = sns.pairplot(data=adult_census[:n_samples_to_plot], x_vars='age',
                 y_vars='hours-per-week', hue=target_column,
                 markers=['o',
                          'v'], plot_kws={'alpha': 0.2}, height=12)

# %% [markdown]
#
# By looking at the data you could infer some hand-written rules to predict the
# class:
# * if you are young (less than 25 year-old roughly), you are in the `<= 50K` class.
# * if you are old (more than 70 year-old roughly), you are in the `<= 50K` class.
# * if you work part-time (less than 40 hours roughly) you are in the `<= 50K` class.
#
# These hand-written rules could work reasonably well without the need for any
# machine learning. Note however that it is not very easy to create rules for
# the region `40 < hours-per-week < 60` and `30 < age < 70`. We can hope that
# machine learning can help in this region. Also note that visualization can
# help creating hand-written rules but is limited to 2 dimensions (maybe 3
# dimensions), whereas machine learning models can build models in
# high-dimensional spaces.
#
# Another thing worth mentioning in this plot: if you are young (less than 25
# year-old roughly) or old (more than 70 year-old roughly) you tend to work
# less. This is a non-linear relationship between age and hours
# per week. Some machine learning models can only capture linear interactions so
# this may be a factor when deciding which model to chose.

# %% [markdown]
# In a machine-learning setting, we will use an algorithm to automatically
# decide what the "rules" should be in order to make predictions on new data.
# The following plot shows the rules a simple algorithm (called decision tree)
# creates:
#
# <img src="../figures/simple-decision-tree-adult-census.png" width=100%/>

# %% [markdown]
# * In the region `age < 28.5` (left region) the prediction is `low-income`. The blue color
#   indicates that the prediction is `low-income`. What is plotted is actually
#   the probability of the class `low-income` as predicted by the algorithm. So the dark
#   blue color indicates that the algorithm is quite sure about this prediction.
# * In the region `age > 28.5 AND hours-per-week < 40.5` (bottom-right region),
#   the prediction is `low-income`. Note that the blue is a bit lighter that for
#   the left region which means that the algorithm is not as certain in this
#   region.
# * In the region `age > 28.5 AND hours-per-week > 40.5` (top-right region),
#   the prediction is `low-income`. The probability of the class `low-income` is
#   very close to 0.5. In practice that means that the model is kind of saying "I
#   don't really know".

# %% [markdown]
# Comments: the decision tree was limited to have three regions. Also a model
# will be better in higher dimensions.

# %% [markdown]
#
# In this notebook we have:
# * loaded the data from a CSV file using `pandas`
# * looked at the kind of variables in the dataset, and differentiated
#   between categorical and numerical variables
# * inspected the data with `pandas`, `seaborn` and `pandas_profiling`. Data inspection
#   can allow you to decide whether using machine learning is appropriate for
#   your data and to highlight potential peculiarities in your data
#
# Key ideas discussed:
# * if your target variable is imbalanced (e.g., you have more samples from one
#   target category than another), you may need special techniques for machine
#   learning
# * having redundant (or highly correlated) columns can be a problem for
#   machine learning algorithms
# * some machine learning models can only capture linear interaction so be
#   aware of non-linear relationships in your data
# By looking at the data y# ---
# jupyter:
#   jupytext:
#     formats: python_scripts//py:percent,notebooks//ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.0
# ---

# %% [markdown]
# In this notebook, we will look at the necessary steps required before any machine learning takes place.
# * load the data
# * look at the variables in the dataset, in particular, differentiate
#   between numerical and categorical variables, which need different
#   preprocessing in most machine learning workflows
# * visualize the distribution of the variables to gain some insights into the dataset

# %%
# Inline plots
# %matplotlib inline

# plotting style
import seaborn as sns

# %% [markdown]
# ## Loading the adult census dataset

# %% [markdown]
# We will use data from the "Current Population adult_census" from 1994 that we
# downloaded from [OpenML](http://openml.org/).

# %%
import pandas as pd

adult_census = pd.read_csv(
    "https://www.openml.org/data/get_csv/1595261/adult-census.csv")

# Or use the local copy:
# adult_census = pd.read_csv('../datasets/adult-census.csv')

# %% [markdown]
# We can look at the OpenML webpage to know more about this dataset.

# %%
from IPython.display import IFrame
IFrame('https://www.openml.org/d/1590', width=1200, height=600)

# %% [markdown]
# ## Look at the variables in the dataset
# The data are stored in a pandas dataframe.

# %%
adult_census.head()

# %% [markdown]
# The column named **class** is our target variable (i.e., the variable which
# we want to predict). The two possible classes are `<= 50K` (low-revenue) and
# `> 50K` (high-revenue).

# %%
target_column = 'class'
adult_census[target_column].value_counts()

# %% [markdown]
# Note: classes are slightly imbalanced. Class imbalance happens often in
# practice and may need special techniques for machine learning. For example in
# a medical setting, if we are trying to predict whether patients will develop
# a rare disease, there will be a lot more healthy patients than ill patients in
# the dataset.

# %% [markdown]
# The dataset contains both numerical and categorical data. Numerical values
# can take continuous values for example `age`. Categorical values can have a
# finite number of values, for example `native-country`.

# %%
numerical_columns = [
    'age', 'education-num', 'capital-gain', 'capital-loss',
    'hours-per-week']
categorical_columns = [
    'workclass', 'education', 'marital-status', 'occupation',
    'relationship', 'race', 'sex', 'native-country']
all_columns = numerical_columns + categorical_columns + [
    target_column]

adult_census = adult_census[all_columns]

# %% [markdown]
# Note that for simplicity, we have ignored the "fnlwgt" (final weight) column
# that was crafted by the creators of the dataset when sampling the dataset to
# be representative of the full census database.

# %% [markdown]
# ## Inspect the data
# Before building a machine learning model, it is a good idea to look at the
# data:
# * maybe the task you are trying to achieve can be solved without machine
#   learning
# * you need to check that the data you need for your task is indeed present in
# the dataset
# * inspecting the data is a good way to find peculiarities. These can can
#   arise during data collection (for example, malfunctioning sensor or missing
#   values), or from the way the data is processed afterwards (for example capped
#   values).

# %% [markdown]
# Let's look at the distribution of individual variables, to get some insights
# about the data. We can start by plotting histograms, note that this only
# works for numerical variables:

# %%
_ = adult_census.hist(figsize=(20, 10))

# %% [markdown]
# We can already make a few comments about some of the variables:
# * age: there are not that many points for 'age > 70'. The dataset description
# does indicate that retired people have been filtered out (`hours-per-week > 0`).
# * education-num: peak at 10 and 13, hard to tell what it corresponds to
# without looking much further. We'll do that later in this notebook.
# * hours per week peaks at 40, this was very likely the standard number of
# working hours at the time of the data collection
# * most values of capital-gain and capital-loss are close to zero

# %% [markdown]
# For categorical variables, we can look at the distribution of values:


# %%
adult_census['sex'].value_counts()

# %%
adult_census['education'].value_counts()

# %% [markdown]
# `pandas_profiling` is a nice tool for inspecting the data (both numerical and
# categorical variables).

# %%
import pandas_profiling
adult_census.profile_report()

# %% [markdown]
# As noted above, `education-num` distribution has two clear peaks around 10
# and 13. It would be reasonable to expect that `education-num` is the number of
# years of education. Let's look at the relationship between `education` and
# `education-num`.
# %%
pd.crosstab(index=adult_census['education'],
            columns=adult_census['education-num'])

# %% [markdown]
# This shows that education and education-num gives you the same information.
# For example, `education-num=2` is equivalent to `education='1st-4th'`. In
# practice that means we can remove `education-num` without losing information.
# Note that having redundant (or highly correlated) columns can be a problem
# for machine learning algorithms.

# %% [markdown]
# Another way to inspect the data is to do a pairplot and show how each variable
# differs according to our target, `class`. Plots along the diagonal show the
# distribution of individual variables for each `class`. The plots on the
# off-diagonal can reveal interesting interactions between variables.

# %%
n_samples_to_plot = 5000
columns = ['age', 'education-num', 'hours-per-week']
_ = sns.pairplot(data=adult_census[:n_samples_to_plot], vars=columns,
                 hue=target_column, plot_kws={'alpha': 0.2},
                 height=4, diag_kind='hist')

# %%
_ = sns.pairplot(data=adult_census[:n_samples_to_plot], x_vars='age',
                 y_vars='hours-per-week', hue=target_column,
                 markers=['o',
                          'v'], plot_kws={'alpha': 0.2}, height=12)

# %% [markdown]
#
# By looking at the data you could infer some hand-written rules to predict the
# class:
# * if you are young (less than 25 year-old roughly), you are in the `<= 50K` class.
# * if you are old (more than 70 year-old roughly), you are in the `<= 50K` class.
# * if you work part-time (less than 40 hours roughly) you are in the `<= 50K` class.
#
# These hand-written rules could work reasonably well without the need for any
# machine learning. Note however that it is not very easy to create rules for
# the region `40 < hours-per-week < 60` and `30 < age < 70`. We can hope that
# machine learning can help in this region. Also note that visualization can
# help creating hand-written rules but is limited to 2 dimensions (maybe 3
# dimensions), whereas machine learning models can build models in
# high-dimensional spaces.
#
# Another thing worth mentioning in this plot: if you are young (less than 25
# year-old roughly) or old (more than 70 year-old roughly) you tend to work
# less. This is a non-linear relationship between age and hours
# per week. Some machine learning models can only capture linear interactions so
# this may be a factor when deciding which model to chose.

# %% [markdown]
# In a machine-learning setting, we will use an algorithm to automatically
# decide what the "rules" should be in order to make predictions on new data.
# The following plot shows the rules a simple algorithm (called decision tree)
# creates:
#
# <img src="../figures/simple-decision-tree-adult-census.png" width=100%/>

# %% [markdown]
# * In the region `age < 28.5` (left region) the prediction is `low-income`. The blue color
#   indicates that the prediction is `low-income`. What is plotted is actually
#   the probability of the class `low-income` as predicted by the algorithm. So the dark
#   blue color indicates that the algorithm is quite sure about this prediction.
# * In the region `age > 28.5 AND hours-per-week < 40.5` (bottom-right region),
#   the prediction is `low-income`. Note that the blue is a bit lighter that for
#   the left region which means that the algorithm is not as certain in this
#   region.
# * In the region `age > 28.5 AND hours-per-week > 40.5` (top-right region),
#   the prediction is `low-income`. The probability of the class `low-income` is
#   very close to 0.5. In practice that means that the model is kind of saying "I
#   don't really know".

# %% [markdown]
# Comments: the decision tree was limited to have three regions. Also a model
# will be better in higher dimensions.

# %% [markdown]
#
# In this notebook we have:
# * loaded the data from a CSV file using `pandas`
# * looked at the kind of variables in the dataset, and differentiated
#   between categorical and numerical variables
# * inspected the data with `pandas`, `seaborn` and `pandas_profiling`. Data inspection
#   can allow you to decide whether using machine learning is appropriate for
#   your data and to highlight potential peculiarities in your data
#
# Key ideas discussed:
# * if your target variable is imbalanced (e.g., you have more samples from one
#   target category than another), you may need special techniques for machine
#   learning
# * having redundant (or highly correlated) columns can be a problem for
#   machine learning algorithms
# * some machine learning models can only capture linear interaction so be
#   aware of non-linear relationships in your dataou could infer some hand-written rules to predict the
# class:# ---
# jupyter:
#   jupytext:
#     formats: python_scripts//py:percent,notebooks//ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.0
# ---

# %% [markdown]
# In this notebook, we will look at the necessary steps required before any machine learning takes place.
# * load the data
# * look at the variables in the dataset, in particular, differentiate
#   between numerical and categorical variables, which need different
#   preprocessing in most machine learning workflows
# * visualize the distribution of the variables to gain some insights into the dataset

# %%
# Inline plots
# %matplotlib inline

# plotting style
import seaborn as sns

# %% [markdown]
# ## Loading the adult census dataset

# %% [markdown]
# We will use data from the "Current Population adult_census" from 1994 that we
# downloaded from [OpenML](http://openml.org/).

# %%
import pandas as pd

adult_census = pd.read_csv(
    "https://www.openml.org/data/get_csv/1595261/adult-census.csv")

# Or use the local copy:
# adult_census = pd.read_csv('../datasets/adult-census.csv')

# %% [markdown]
# We can look at the OpenML webpage to know more about this dataset.

# %%
from IPython.display import IFrame
IFrame('https://www.openml.org/d/1590', width=1200, height=600)

# %% [markdown]
# ## Look at the variables in the dataset
# The data are stored in a pandas dataframe.

# %%
adult_census.head()

# %% [markdown]
# The column named **class** is our target variable (i.e., the variable which
# we want to predict). The two possible classes are `<= 50K` (low-revenue) and
# `> 50K` (high-revenue).

# %%
target_column = 'class'
adult_census[target_column].value_counts()

# %% [markdown]
# Note: classes are slightly imbalanced. Class imbalance happens often in
# practice and may need special techniques for machine learning. For example in
# a medical setting, if we are trying to predict whether patients will develop
# a rare disease, there will be a lot more healthy patients than ill patients in
# the dataset.

# %% [markdown]
# The dataset contains both numerical and categorical data. Numerical values
# can take continuous values for example `age`. Categorical values can have a
# finite number of values, for example `native-country`.

# %%
numerical_columns = [
    'age', 'education-num', 'capital-gain', 'capital-loss',
    'hours-per-week']
categorical_columns = [
    'workclass', 'education', 'marital-status', 'occupation',
    'relationship', 'race', 'sex', 'native-country']
all_columns = numerical_columns + categorical_columns + [
    target_column]

adult_census = adult_census[all_columns]

# %% [markdown]
# Note that for simplicity, we have ignored the "fnlwgt" (final weight) column
# that was crafted by the creators of the dataset when sampling the dataset to
# be representative of the full census database.

# %% [markdown]
# ## Inspect the data
# Before building a machine learning model, it is a good idea to look at the
# data:
# * maybe the task you are trying to achieve can be solved without machine
#   learning
# * you need to check that the data you need for your task is indeed present in
# the dataset
# * inspecting the data is a good way to find peculiarities. These can can
#   arise during data collection (for example, malfunctioning sensor or missing
#   values), or from the way the data is processed afterwards (for example capped
#   values).

# %% [markdown]
# Let's look at the distribution of individual variables, to get some insights
# about the data. We can start by plotting histograms, note that this only
# works for numerical variables:

# %%
_ = adult_census.hist(figsize=(20, 10))

# %% [markdown]
# We can already make a few comments about some of the variables:
# * age: there are not that many points for 'age > 70'. The dataset description
# does indicate that retired people have been filtered out (`hours-per-week > 0`).
# * education-num: peak at 10 and 13, hard to tell what it corresponds to
# without looking much further. We'll do that later in this notebook.
# * hours per week peaks at 40, this was very likely the standard number of
# working hours at the time of the data collection
# * most values of capital-gain and capital-loss are close to zero

# %% [markdown]
# For categorical variables, we can look at the distribution of values:


# %%
adult_census['sex'].value_counts()

# %%
adult_census['education'].value_counts()

# %% [markdown]
# `pandas_profiling` is a nice tool for inspecting the data (both numerical and
# categorical variables).

# %%
import pandas_profiling
adult_census.profile_report()

# %% [markdown]
# As noted above, `education-num` distribution has two clear peaks around 10
# and 13. It would be reasonable to expect that `education-num` is the number of
# years of education. Let's look at the relationship between `education` and
# `education-num`.
# %%
pd.crosstab(index=adult_census['education'],
            columns=adult_census['education-num'])

# %% [markdown]
# This shows that education and education-num gives you the same information.
# For example, `education-num=2` is equivalent to `education='1st-4th'`. In
# practice that means we can remove `education-num` without losing information.
# Note that having redundant (or highly correlated) columns can be a problem
# for machine learning algorithms.

# %% [markdown]
# Another way to inspect the data is to do a pairplot and show how each variable
# differs according to our target, `class`. Plots along the diagonal show the
# distribution of individual variables for each `class`. The plots on the
# off-diagonal can reveal interesting interactions between variables.

# %%
n_samples_to_plot = 5000
columns = ['age', 'education-num', 'hours-per-week']
_ = sns.pairplot(data=adult_census[:n_samples_to_plot], vars=columns,
                 hue=target_column, plot_kws={'alpha': 0.2},
                 height=4, diag_kind='hist')

# %%
_ = sns.pairplot(data=adult_census[:n_samples_to_plot], x_vars='age',
                 y_vars='hours-per-week', hue=target_column,
                 markers=['o',
                          'v'], plot_kws={'alpha': 0.2}, height=12)

# %% [markdown]
#
# By looking at the data you could infer some hand-written rules to predict the
# class:
# * if you are young (less than 25 year-old roughly), you are in the `<= 50K` class.
# * if you are old (more than 70 year-old roughly), you are in the `<= 50K` class.
# * if you work part-time (less than 40 hours roughly) you are in the `<= 50K` class.
#
# These hand-written rules could work reasonably well without the need for any
# machine learning. Note however that it is not very easy to create rules for
# the region `40 < hours-per-week < 60` and `30 < age < 70`. We can hope that
# machine learning can help in this region. Also note that visualization can
# help creating hand-written rules but is limited to 2 dimensions (maybe 3
# dimensions), whereas machine learning models can build models in
# high-dimensional spaces.
#
# Another thing worth mentioning in this plot: if you are young (less than 25
# year-old roughly) or old (more than 70 year-old roughly) you tend to work
# less. This is a non-linear relationship between age and hours
# per week. Some machine learning models can only capture linear interactions so
# this may be a factor when deciding which model to chose.

# %% [markdown]
# In a machine-learning setting, we will use an algorithm to automatically
# decide what the "rules" should be in order to make predictions on new data.
# The following plot shows the rules a simple algorithm (called decision tree)
# creates:
#
# <img src="../figures/simple-decision-tree-adult-census.png" width=100%/>

# %% [markdown]
# * In the region `age < 28.5` (left region) the prediction is `low-income`. The blue color
#   indicates that the prediction is `low-income`. What is plotted is actually
#   the probability of the class `low-income` as predicted by the algorithm. So the dark
#   blue color indicates that the algorithm is quite sure about this prediction.
# * In the region `age > 28.5 AND hours-per-week < 40.5` (bottom-right region),
#   the prediction is `low-income`. Note that the blue is a bit lighter that for
#   the left region which means that the algorithm is not as certain in this
#   region.
# * In the region `age > 28.5 AND hours-per-week > 40.5` (top-right region),
#   the prediction is `low-income`. The probability of the class `low-income` is
#   very close to 0.5. In practice that means that the model is kind of saying "I
#   don't really know".

# %% [markdown]
# Comments: the decision tree was limited to have three regions. Also a model
# will be better in higher dimensions.

# %% [markdown]
#
# In this notebook we have:
# * loaded the data from a CSV file using `pandas`
# * looked at the kind of variables in the dataset, and differentiated
#   between categorical and numerical variables
# * inspected the data with `pandas`, `seaborn` and `pandas_profiling`. Data inspection
#   can allow you to decide whether using machine learning is appropriate for
#   your data and to highlight potential peculiarities in your data
#
# Key ideas discussed:
# * if your target variable is imbalanced (e.g., you have more samples from one
#   target category than another), you may need special techniques for machine
#   learning
# * having redundant (or highly correlated) columns can be a problem for
#   machine learning algorithms
# * some machine learning models can only capture linear interaction so be
#   aware of non-linear relationships in your data
# * if you are young (less# ---
# jupyter:
#   jupytext:
#     formats: python_scripts//py:percent,notebooks//ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.0
# ---

# %% [markdown]
# In this notebook, we will look at the necessary steps required before any machine learning takes place.
# * load the data
# * look at the variables in the dataset, in particular, differentiate
#   between numerical and categorical variables, which need different
#   preprocessing in most machine learning workflows
# * visualize the distribution of the variables to gain some insights into the dataset

# %%
# Inline plots
# %matplotlib inline

# plotting style
import seaborn as sns

# %% [markdown]
# ## Loading the adult census dataset

# %% [markdown]
# We will use data from the "Current Population adult_census" from 1994 that we
# downloaded from [OpenML](http://openml.org/).

# %%
import pandas as pd

adult_census = pd.read_csv(
    "https://www.openml.org/data/get_csv/1595261/adult-census.csv")

# Or use the local copy:
# adult_census = pd.read_csv('../datasets/adult-census.csv')

# %% [markdown]
# We can look at the OpenML webpage to know more about this dataset.

# %%
from IPython.display import IFrame
IFrame('https://www.openml.org/d/1590', width=1200, height=600)

# %% [markdown]
# ## Look at the variables in the dataset
# The data are stored in a pandas dataframe.

# %%
adult_census.head()

# %% [markdown]
# The column named **class** is our target variable (i.e., the variable which
# we want to predict). The two possible classes are `<= 50K` (low-revenue) and
# `> 50K` (high-revenue).

# %%
target_column = 'class'
adult_census[target_column].value_counts()

# %% [markdown]
# Note: classes are slightly imbalanced. Class imbalance happens often in
# practice and may need special techniques for machine learning. For example in
# a medical setting, if we are trying to predict whether patients will develop
# a rare disease, there will be a lot more healthy patients than ill patients in
# the dataset.

# %% [markdown]
# The dataset contains both numerical and categorical data. Numerical values
# can take continuous values for example `age`. Categorical values can have a
# finite number of values, for example `native-country`.

# %%
numerical_columns = [
    'age', 'education-num', 'capital-gain', 'capital-loss',
    'hours-per-week']
categorical_columns = [
    'workclass', 'education', 'marital-status', 'occupation',
    'relationship', 'race', 'sex', 'native-country']
all_columns = numerical_columns + categorical_columns + [
    target_column]

adult_census = adult_census[all_columns]

# %% [markdown]
# Note that for simplicity, we have ignored the "fnlwgt" (final weight) column
# that was crafted by the creators of the dataset when sampling the dataset to
# be representative of the full census database.

# %% [markdown]
# ## Inspect the data
# Before building a machine learning model, it is a good idea to look at the
# data:
# * maybe the task you are trying to achieve can be solved without machine
#   learning
# * you need to check that the data you need for your task is indeed present in
# the dataset
# * inspecting the data is a good way to find peculiarities. These can can
#   arise during data collection (for example, malfunctioning sensor or missing
#   values), or from the way the data is processed afterwards (for example capped
#   values).

# %% [markdown]
# Let's look at the distribution of individual variables, to get some insights
# about the data. We can start by plotting histograms, note that this only
# works for numerical variables:

# %%
_ = adult_census.hist(figsize=(20, 10))

# %% [markdown]
# We can already make a few comments about some of the variables:
# * age: there are not that many points for 'age > 70'. The dataset description
# does indicate that retired people have been filtered out (`hours-per-week > 0`).
# * education-num: peak at 10 and 13, hard to tell what it corresponds to
# without looking much further. We'll do that later in this notebook.
# * hours per week peaks at 40, this was very likely the standard number of
# working hours at the time of the data collection
# * most values of capital-gain and capital-loss are close to zero

# %% [markdown]
# For categorical variables, we can look at the distribution of values:


# %%
adult_census['sex'].value_counts()

# %%
adult_census['education'].value_counts()

# %% [markdown]
# `pandas_profiling` is a nice tool for inspecting the data (both numerical and
# categorical variables).

# %%
import pandas_profiling
adult_census.profile_report()

# %% [markdown]
# As noted above, `education-num` distribution has two clear peaks around 10
# and 13. It would be reasonable to expect that `education-num` is the number of
# years of education. Let's look at the relationship between `education` and
# `education-num`.
# %%
pd.crosstab(index=adult_census['education'],
            columns=adult_census['education-num'])

# %% [markdown]
# This shows that education and education-num gives you the same information.
# For example, `education-num=2` is equivalent to `education='1st-4th'`. In
# practice that means we can remove `education-num` without losing information.
# Note that having redundant (or highly correlated) columns can be a problem
# for machine learning algorithms.

# %% [markdown]
# Another way to inspect the data is to do a pairplot and show how each variable
# differs according to our target, `class`. Plots along the diagonal show the
# distribution of individual variables for each `class`. The plots on the
# off-diagonal can reveal interesting interactions between variables.

# %%
n_samples_to_plot = 5000
columns = ['age', 'education-num', 'hours-per-week']
_ = sns.pairplot(data=adult_census[:n_samples_to_plot], vars=columns,
                 hue=target_column, plot_kws={'alpha': 0.2},
                 height=4, diag_kind='hist')

# %%
_ = sns.pairplot(data=adult_census[:n_samples_to_plot], x_vars='age',
                 y_vars='hours-per-week', hue=target_column,
                 markers=['o',
                          'v'], plot_kws={'alpha': 0.2}, height=12)

# %% [markdown]
#
# By looking at the data you could infer some hand-written rules to predict the
# class:
# * if you are young (less than 25 year-old roughly), you are in the `<= 50K` class.
# * if you are old (more than 70 year-old roughly), you are in the `<= 50K` class.
# * if you work part-time (less than 40 hours roughly) you are in the `<= 50K` class.
#
# These hand-written rules could work reasonably well without the need for any
# machine learning. Note however that it is not very easy to create rules for
# the region `40 < hours-per-week < 60` and `30 < age < 70`. We can hope that
# machine learning can help in this region. Also note that visualization can
# help creating hand-written rules but is limited to 2 dimensions (maybe 3
# dimensions), whereas machine learning models can build models in
# high-dimensional spaces.
#
# Another thing worth mentioning in this plot: if you are young (less than 25
# year-old roughly) or old (more than 70 year-old roughly) you tend to work
# less. This is a non-linear relationship between age and hours
# per week. Some machine learning models can only capture linear interactions so
# this may be a factor when deciding which model to chose.

# %% [markdown]
# In a machine-learning setting, we will use an algorithm to automatically
# decide what the "rules" should be in order to make predictions on new data.
# The following plot shows the rules a simple algorithm (called decision tree)
# creates:
#
# <img src="../figures/simple-decision-tree-adult-census.png" width=100%/>

# %% [markdown]
# * In the region `age < 28.5` (left region) the prediction is `low-income`. The blue color
#   indicates that the prediction is `low-income`. What is plotted is actually
#   the probability of the class `low-income` as predicted by the algorithm. So the dark
#   blue color indicates that the algorithm is quite sure about this prediction.
# * In the region `age > 28.5 AND hours-per-week < 40.5` (bottom-right region),
#   the prediction is `low-income`. Note that the blue is a bit lighter that for
#   the left region which means that the algorithm is not as certain in this
#   region.
# * In the region `age > 28.5 AND hours-per-week > 40.5` (top-right region),
#   the prediction is `low-income`. The probability of the class `low-income` is
#   very close to 0.5. In practice that means that the model is kind of saying "I
#   don't really know".

# %% [markdown]
# Comments: the decision tree was limited to have three regions. Also a model
# will be better in higher dimensions.

# %% [markdown]
#
# In this notebook we have:
# * loaded the data from a CSV file using `pandas`
# * looked at the kind of variables in the dataset, and differentiated
#   between categorical and numerical variables
# * inspected the data with `pandas`, `seaborn` and `pandas_profiling`. Data inspection
#   can allow you to decide whether using machine learning is appropriate for
#   your data and to highlight potential peculiarities in your data
#
# Key ideas discussed:
# * if your target variable is imbalanced (e.g., you have more samples from one
#   target category than another), you may need special techniques for machine
#   learning
# * having redundant (or highly correlated) columns can be a problem for
#   machine learning algorithms
# * some machine learning models can only capture linear interaction so be
#   aware of non-linear relationships in your data than 25 year-old roughly), you are in the `<= 50K` class.
# * if you are old (more t# ---
# jupyter:
#   jupytext:
#     formats: python_scripts//py:percent,notebooks//ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.0
# ---

# %% [markdown]
# In this notebook, we will look at the necessary steps required before any machine learning takes place.
# * load the data
# * look at the variables in the dataset, in particular, differentiate
#   between numerical and categorical variables, which need different
#   preprocessing in most machine learning workflows
# * visualize the distribution of the variables to gain some insights into the dataset

# %%
# Inline plots
# %matplotlib inline

# plotting style
import seaborn as sns

# %% [markdown]
# ## Loading the adult census dataset

# %% [markdown]
# We will use data from the "Current Population adult_census" from 1994 that we
# downloaded from [OpenML](http://openml.org/).

# %%
import pandas as pd

adult_census = pd.read_csv(
    "https://www.openml.org/data/get_csv/1595261/adult-census.csv")

# Or use the local copy:
# adult_census = pd.read_csv('../datasets/adult-census.csv')

# %% [markdown]
# We can look at the OpenML webpage to know more about this dataset.

# %%
from IPython.display import IFrame
IFrame('https://www.openml.org/d/1590', width=1200, height=600)

# %% [markdown]
# ## Look at the variables in the dataset
# The data are stored in a pandas dataframe.

# %%
adult_census.head()

# %% [markdown]
# The column named **class** is our target variable (i.e., the variable which
# we want to predict). The two possible classes are `<= 50K` (low-revenue) and
# `> 50K` (high-revenue).

# %%
target_column = 'class'
adult_census[target_column].value_counts()

# %% [markdown]
# Note: classes are slightly imbalanced. Class imbalance happens often in
# practice and may need special techniques for machine learning. For example in
# a medical setting, if we are trying to predict whether patients will develop
# a rare disease, there will be a lot more healthy patients than ill patients in
# the dataset.

# %% [markdown]
# The dataset contains both numerical and categorical data. Numerical values
# can take continuous values for example `age`. Categorical values can have a
# finite number of values, for example `native-country`.

# %%
numerical_columns = [
    'age', 'education-num', 'capital-gain', 'capital-loss',
    'hours-per-week']
categorical_columns = [
    'workclass', 'education', 'marital-status', 'occupation',
    'relationship', 'race', 'sex', 'native-country']
all_columns = numerical_columns + categorical_columns + [
    target_column]

adult_census = adult_census[all_columns]

# %% [markdown]
# Note that for simplicity, we have ignored the "fnlwgt" (final weight) column
# that was crafted by the creators of the dataset when sampling the dataset to
# be representative of the full census database.

# %% [markdown]
# ## Inspect the data
# Before building a machine learning model, it is a good idea to look at the
# data:
# * maybe the task you are trying to achieve can be solved without machine
#   learning
# * you need to check that the data you need for your task is indeed present in
# the dataset
# * inspecting the data is a good way to find peculiarities. These can can
#   arise during data collection (for example, malfunctioning sensor or missing
#   values), or from the way the data is processed afterwards (for example capped
#   values).

# %% [markdown]
# Let's look at the distribution of individual variables, to get some insights
# about the data. We can start by plotting histograms, note that this only
# works for numerical variables:

# %%
_ = adult_census.hist(figsize=(20, 10))

# %% [markdown]
# We can already make a few comments about some of the variables:
# * age: there are not that many points for 'age > 70'. The dataset description
# does indicate that retired people have been filtered out (`hours-per-week > 0`).
# * education-num: peak at 10 and 13, hard to tell what it corresponds to
# without looking much further. We'll do that later in this notebook.
# * hours per week peaks at 40, this was very likely the standard number of
# working hours at the time of the data collection
# * most values of capital-gain and capital-loss are close to zero

# %% [markdown]
# For categorical variables, we can look at the distribution of values:


# %%
adult_census['sex'].value_counts()

# %%
adult_census['education'].value_counts()

# %% [markdown]
# `pandas_profiling` is a nice tool for inspecting the data (both numerical and
# categorical variables).

# %%
import pandas_profiling
adult_census.profile_report()

# %% [markdown]
# As noted above, `education-num` distribution has two clear peaks around 10
# and 13. It would be reasonable to expect that `education-num` is the number of
# years of education. Let's look at the relationship between `education` and
# `education-num`.
# %%
pd.crosstab(index=adult_census['education'],
            columns=adult_census['education-num'])

# %% [markdown]
# This shows that education and education-num gives you the same information.
# For example, `education-num=2` is equivalent to `education='1st-4th'`. In
# practice that means we can remove `education-num` without losing information.
# Note that having redundant (or highly correlated) columns can be a problem
# for machine learning algorithms.

# %% [markdown]
# Another way to inspect the data is to do a pairplot and show how each variable
# differs according to our target, `class`. Plots along the diagonal show the
# distribution of individual variables for each `class`. The plots on the
# off-diagonal can reveal interesting interactions between variables.

# %%
n_samples_to_plot = 5000
columns = ['age', 'education-num', 'hours-per-week']
_ = sns.pairplot(data=adult_census[:n_samples_to_plot], vars=columns,
                 hue=target_column, plot_kws={'alpha': 0.2},
                 height=4, diag_kind='hist')

# %%
_ = sns.pairplot(data=adult_census[:n_samples_to_plot], x_vars='age',
                 y_vars='hours-per-week', hue=target_column,
                 markers=['o',
                          'v'], plot_kws={'alpha': 0.2}, height=12)

# %% [markdown]
#
# By looking at the data you could infer some hand-written rules to predict the
# class:
# * if you are young (less than 25 year-old roughly), you are in the `<= 50K` class.
# * if you are old (more than 70 year-old roughly), you are in the `<= 50K` class.
# * if you work part-time (less than 40 hours roughly) you are in the `<= 50K` class.
#
# These hand-written rules could work reasonably well without the need for any
# machine learning. Note however that it is not very easy to create rules for
# the region `40 < hours-per-week < 60` and `30 < age < 70`. We can hope that
# machine learning can help in this region. Also note that visualization can
# help creating hand-written rules but is limited to 2 dimensions (maybe 3
# dimensions), whereas machine learning models can build models in
# high-dimensional spaces.
#
# Another thing worth mentioning in this plot: if you are young (less than 25
# year-old roughly) or old (more than 70 year-old roughly) you tend to work
# less. This is a non-linear relationship between age and hours
# per week. Some machine learning models can only capture linear interactions so
# this may be a factor when deciding which model to chose.

# %% [markdown]
# In a machine-learning setting, we will use an algorithm to automatically
# decide what the "rules" should be in order to make predictions on new data.
# The following plot shows the rules a simple algorithm (called decision tree)
# creates:
#
# <img src="../figures/simple-decision-tree-adult-census.png" width=100%/>

# %% [markdown]
# * In the region `age < 28.5` (left region) the prediction is `low-income`. The blue color
#   indicates that the prediction is `low-income`. What is plotted is actually
#   the probability of the class `low-income` as predicted by the algorithm. So the dark
#   blue color indicates that the algorithm is quite sure about this prediction.
# * In the region `age > 28.5 AND hours-per-week < 40.5` (bottom-right region),
#   the prediction is `low-income`. Note that the blue is a bit lighter that for
#   the left region which means that the algorithm is not as certain in this
#   region.
# * In the region `age > 28.5 AND hours-per-week > 40.5` (top-right region),
#   the prediction is `low-income`. The probability of the class `low-income` is
#   very close to 0.5. In practice that means that the model is kind of saying "I
#   don't really know".

# %% [markdown]
# Comments: the decision tree was limited to have three regions. Also a model
# will be better in higher dimensions.

# %% [markdown]
#
# In this notebook we have:
# * loaded the data from a CSV file using `pandas`
# * looked at the kind of variables in the dataset, and differentiated
#   between categorical and numerical variables
# * inspected the data with `pandas`, `seaborn` and `pandas_profiling`. Data inspection
#   can allow you to decide whether using machine learning is appropriate for
#   your data and to highlight potential peculiarities in your data
#
# Key ideas discussed:
# * if your target variable is imbalanced (e.g., you have more samples from one
#   target category than another), you may need special techniques for machine
#   learning
# * having redundant (or highly correlated) columns can be a problem for
#   machine learning algorithms
# * some machine learning models can only capture linear interaction so be
#   aware of non-linear relationships in your datahan 70 year-old roughly), you are in the `<= 50K` class.
# * if you work part-time # ---
# jupyter:
#   jupytext:
#     formats: python_scripts//py:percent,notebooks//ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.0
# ---

# %% [markdown]
# In this notebook, we will look at the necessary steps required before any machine learning takes place.
# * load the data
# * look at the variables in the dataset, in particular, differentiate
#   between numerical and categorical variables, which need different
#   preprocessing in most machine learning workflows
# * visualize the distribution of the variables to gain some insights into the dataset

# %%
# Inline plots
# %matplotlib inline

# plotting style
import seaborn as sns

# %% [markdown]
# ## Loading the adult census dataset

# %% [markdown]
# We will use data from the "Current Population adult_census" from 1994 that we
# downloaded from [OpenML](http://openml.org/).

# %%
import pandas as pd

adult_census = pd.read_csv(
    "https://www.openml.org/data/get_csv/1595261/adult-census.csv")

# Or use the local copy:
# adult_census = pd.read_csv('../datasets/adult-census.csv')

# %% [markdown]
# We can look at the OpenML webpage to know more about this dataset.

# %%
from IPython.display import IFrame
IFrame('https://www.openml.org/d/1590', width=1200, height=600)

# %% [markdown]
# ## Look at the variables in the dataset
# The data are stored in a pandas dataframe.

# %%
adult_census.head()

# %% [markdown]
# The column named **class** is our target variable (i.e., the variable which
# we want to predict). The two possible classes are `<= 50K` (low-revenue) and
# `> 50K` (high-revenue).

# %%
target_column = 'class'
adult_census[target_column].value_counts()

# %% [markdown]
# Note: classes are slightly imbalanced. Class imbalance happens often in
# practice and may need special techniques for machine learning. For example in
# a medical setting, if we are trying to predict whether patients will develop
# a rare disease, there will be a lot more healthy patients than ill patients in
# the dataset.

# %% [markdown]
# The dataset contains both numerical and categorical data. Numerical values
# can take continuous values for example `age`. Categorical values can have a
# finite number of values, for example `native-country`.

# %%
numerical_columns = [
    'age', 'education-num', 'capital-gain', 'capital-loss',
    'hours-per-week']
categorical_columns = [
    'workclass', 'education', 'marital-status', 'occupation',
    'relationship', 'race', 'sex', 'native-country']
all_columns = numerical_columns + categorical_columns + [
    target_column]

adult_census = adult_census[all_columns]

# %% [markdown]
# Note that for simplicity, we have ignored the "fnlwgt" (final weight) column
# that was crafted by the creators of the dataset when sampling the dataset to
# be representative of the full census database.

# %% [markdown]
# ## Inspect the data
# Before building a machine learning model, it is a good idea to look at the
# data:
# * maybe the task you are trying to achieve can be solved without machine
#   learning
# * you need to check that the data you need for your task is indeed present in
# the dataset
# * inspecting the data is a good way to find peculiarities. These can can
#   arise during data collection (for example, malfunctioning sensor or missing
#   values), or from the way the data is processed afterwards (for example capped
#   values).

# %% [markdown]
# Let's look at the distribution of individual variables, to get some insights
# about the data. We can start by plotting histograms, note that this only
# works for numerical variables:

# %%
_ = adult_census.hist(figsize=(20, 10))

# %% [markdown]
# We can already make a few comments about some of the variables:
# * age: there are not that many points for 'age > 70'. The dataset description
# does indicate that retired people have been filtered out (`hours-per-week > 0`).
# * education-num: peak at 10 and 13, hard to tell what it corresponds to
# without looking much further. We'll do that later in this notebook.
# * hours per week peaks at 40, this was very likely the standard number of
# working hours at the time of the data collection
# * most values of capital-gain and capital-loss are close to zero

# %% [markdown]
# For categorical variables, we can look at the distribution of values:


# %%
adult_census['sex'].value_counts()

# %%
adult_census['education'].value_counts()

# %% [markdown]
# `pandas_profiling` is a nice tool for inspecting the data (both numerical and
# categorical variables).

# %%
import pandas_profiling
adult_census.profile_report()

# %% [markdown]
# As noted above, `education-num` distribution has two clear peaks around 10
# and 13. It would be reasonable to expect that `education-num` is the number of
# years of education. Let's look at the relationship between `education` and
# `education-num`.
# %%
pd.crosstab(index=adult_census['education'],
            columns=adult_census['education-num'])

# %% [markdown]
# This shows that education and education-num gives you the same information.
# For example, `education-num=2` is equivalent to `education='1st-4th'`. In
# practice that means we can remove `education-num` without losing information.
# Note that having redundant (or highly correlated) columns can be a problem
# for machine learning algorithms.

# %% [markdown]
# Another way to inspect the data is to do a pairplot and show how each variable
# differs according to our target, `class`. Plots along the diagonal show the
# distribution of individual variables for each `class`. The plots on the
# off-diagonal can reveal interesting interactions between variables.

# %%
n_samples_to_plot = 5000
columns = ['age', 'education-num', 'hours-per-week']
_ = sns.pairplot(data=adult_census[:n_samples_to_plot], vars=columns,
                 hue=target_column, plot_kws={'alpha': 0.2},
                 height=4, diag_kind='hist')

# %%
_ = sns.pairplot(data=adult_census[:n_samples_to_plot], x_vars='age',
                 y_vars='hours-per-week', hue=target_column,
                 markers=['o',
                          'v'], plot_kws={'alpha': 0.2}, height=12)

# %% [markdown]
#
# By looking at the data you could infer some hand-written rules to predict the
# class:
# * if you are young (less than 25 year-old roughly), you are in the `<= 50K` class.
# * if you are old (more than 70 year-old roughly), you are in the `<= 50K` class.
# * if you work part-time (less than 40 hours roughly) you are in the `<= 50K` class.
#
# These hand-written rules could work reasonably well without the need for any
# machine learning. Note however that it is not very easy to create rules for
# the region `40 < hours-per-week < 60` and `30 < age < 70`. We can hope that
# machine learning can help in this region. Also note that visualization can
# help creating hand-written rules but is limited to 2 dimensions (maybe 3
# dimensions), whereas machine learning models can build models in
# high-dimensional spaces.
#
# Another thing worth mentioning in this plot: if you are young (less than 25
# year-old roughly) or old (more than 70 year-old roughly) you tend to work
# less. This is a non-linear relationship between age and hours
# per week. Some machine learning models can only capture linear interactions so
# this may be a factor when deciding which model to chose.

# %% [markdown]
# In a machine-learning setting, we will use an algorithm to automatically
# decide what the "rules" should be in order to make predictions on new data.
# The following plot shows the rules a simple algorithm (called decision tree)
# creates:
#
# <img src="../figures/simple-decision-tree-adult-census.png" width=100%/>

# %% [markdown]
# * In the region `age < 28.5` (left region) the prediction is `low-income`. The blue color
#   indicates that the prediction is `low-income`. What is plotted is actually
#   the probability of the class `low-income` as predicted by the algorithm. So the dark
#   blue color indicates that the algorithm is quite sure about this prediction.
# * In the region `age > 28.5 AND hours-per-week < 40.5` (bottom-right region),
#   the prediction is `low-income`. Note that the blue is a bit lighter that for
#   the left region which means that the algorithm is not as certain in this
#   region.
# * In the region `age > 28.5 AND hours-per-week > 40.5` (top-right region),
#   the prediction is `low-income`. The probability of the class `low-income` is
#   very close to 0.5. In practice that means that the model is kind of saying "I
#   don't really know".

# %% [markdown]
# Comments: the decision tree was limited to have three regions. Also a model
# will be better in higher dimensions.

# %% [markdown]
#
# In this notebook we have:
# * loaded the data from a CSV file using `pandas`
# * looked at the kind of variables in the dataset, and differentiated
#   between categorical and numerical variables
# * inspected the data with `pandas`, `seaborn` and `pandas_profiling`. Data inspection
#   can allow you to decide whether using machine learning is appropriate for
#   your data and to highlight potential peculiarities in your data
#
# Key ideas discussed:
# * if your target variable is imbalanced (e.g., you have more samples from one
#   target category than another), you may need special techniques for machine
#   learning
# * having redundant (or highly correlated) columns can be a problem for
#   machine learning algorithms
# * some machine learning models can only capture linear interaction so be
#   aware of non-linear relationships in your data(less than 40 hours roughly) you are in the `<= 50K` class.
## ---
# jupyter:
#   jupytext:
#     formats: python_scripts//py:percent,notebooks//ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.0
# ---

# %% [markdown]
# In this notebook, we will look at the necessary steps required before any machine learning takes place.
# * load the data
# * look at the variables in the dataset, in particular, differentiate
#   between numerical and categorical variables, which need different
#   preprocessing in most machine learning workflows
# * visualize the distribution of the variables to gain some insights into the dataset

# %%
# Inline plots
# %matplotlib inline

# plotting style
import seaborn as sns

# %% [markdown]
# ## Loading the adult census dataset

# %% [markdown]
# We will use data from the "Current Population adult_census" from 1994 that we
# downloaded from [OpenML](http://openml.org/).

# %%
import pandas as pd

adult_census = pd.read_csv(
    "https://www.openml.org/data/get_csv/1595261/adult-census.csv")

# Or use the local copy:
# adult_census = pd.read_csv('../datasets/adult-census.csv')

# %% [markdown]
# We can look at the OpenML webpage to know more about this dataset.

# %%
from IPython.display import IFrame
IFrame('https://www.openml.org/d/1590', width=1200, height=600)

# %% [markdown]
# ## Look at the variables in the dataset
# The data are stored in a pandas dataframe.

# %%
adult_census.head()

# %% [markdown]
# The column named **class** is our target variable (i.e., the variable which
# we want to predict). The two possible classes are `<= 50K` (low-revenue) and
# `> 50K` (high-revenue).

# %%
target_column = 'class'
adult_census[target_column].value_counts()

# %% [markdown]
# Note: classes are slightly imbalanced. Class imbalance happens often in
# practice and may need special techniques for machine learning. For example in
# a medical setting, if we are trying to predict whether patients will develop
# a rare disease, there will be a lot more healthy patients than ill patients in
# the dataset.

# %% [markdown]
# The dataset contains both numerical and categorical data. Numerical values
# can take continuous values for example `age`. Categorical values can have a
# finite number of values, for example `native-country`.

# %%
numerical_columns = [
    'age', 'education-num', 'capital-gain', 'capital-loss',
    'hours-per-week']
categorical_columns = [
    'workclass', 'education', 'marital-status', 'occupation',
    'relationship', 'race', 'sex', 'native-country']
all_columns = numerical_columns + categorical_columns + [
    target_column]

adult_census = adult_census[all_columns]

# %% [markdown]
# Note that for simplicity, we have ignored the "fnlwgt" (final weight) column
# that was crafted by the creators of the dataset when sampling the dataset to
# be representative of the full census database.

# %% [markdown]
# ## Inspect the data
# Before building a machine learning model, it is a good idea to look at the
# data:
# * maybe the task you are trying to achieve can be solved without machine
#   learning
# * you need to check that the data you need for your task is indeed present in
# the dataset
# * inspecting the data is a good way to find peculiarities. These can can
#   arise during data collection (for example, malfunctioning sensor or missing
#   values), or from the way the data is processed afterwards (for example capped
#   values).

# %% [markdown]
# Let's look at the distribution of individual variables, to get some insights
# about the data. We can start by plotting histograms, note that this only
# works for numerical variables:

# %%
_ = adult_census.hist(figsize=(20, 10))

# %% [markdown]
# We can already make a few comments about some of the variables:
# * age: there are not that many points for 'age > 70'. The dataset description
# does indicate that retired people have been filtered out (`hours-per-week > 0`).
# * education-num: peak at 10 and 13, hard to tell what it corresponds to
# without looking much further. We'll do that later in this notebook.
# * hours per week peaks at 40, this was very likely the standard number of
# working hours at the time of the data collection
# * most values of capital-gain and capital-loss are close to zero

# %% [markdown]
# For categorical variables, we can look at the distribution of values:


# %%
adult_census['sex'].value_counts()

# %%
adult_census['education'].value_counts()

# %% [markdown]
# `pandas_profiling` is a nice tool for inspecting the data (both numerical and
# categorical variables).

# %%
import pandas_profiling
adult_census.profile_report()

# %% [markdown]
# As noted above, `education-num` distribution has two clear peaks around 10
# and 13. It would be reasonable to expect that `education-num` is the number of
# years of education. Let's look at the relationship between `education` and
# `education-num`.
# %%
pd.crosstab(index=adult_census['education'],
            columns=adult_census['education-num'])

# %% [markdown]
# This shows that education and education-num gives you the same information.
# For example, `education-num=2` is equivalent to `education='1st-4th'`. In
# practice that means we can remove `education-num` without losing information.
# Note that having redundant (or highly correlated) columns can be a problem
# for machine learning algorithms.

# %% [markdown]
# Another way to inspect the data is to do a pairplot and show how each variable
# differs according to our target, `class`. Plots along the diagonal show the
# distribution of individual variables for each `class`. The plots on the
# off-diagonal can reveal interesting interactions between variables.

# %%
n_samples_to_plot = 5000
columns = ['age', 'education-num', 'hours-per-week']
_ = sns.pairplot(data=adult_census[:n_samples_to_plot], vars=columns,
                 hue=target_column, plot_kws={'alpha': 0.2},
                 height=4, diag_kind='hist')

# %%
_ = sns.pairplot(data=adult_census[:n_samples_to_plot], x_vars='age',
                 y_vars='hours-per-week', hue=target_column,
                 markers=['o',
                          'v'], plot_kws={'alpha': 0.2}, height=12)

# %% [markdown]
#
# By looking at the data you could infer some hand-written rules to predict the
# class:
# * if you are young (less than 25 year-old roughly), you are in the `<= 50K` class.
# * if you are old (more than 70 year-old roughly), you are in the `<= 50K` class.
# * if you work part-time (less than 40 hours roughly) you are in the `<= 50K` class.
#
# These hand-written rules could work reasonably well without the need for any
# machine learning. Note however that it is not very easy to create rules for
# the region `40 < hours-per-week < 60` and `30 < age < 70`. We can hope that
# machine learning can help in this region. Also note that visualization can
# help creating hand-written rules but is limited to 2 dimensions (maybe 3
# dimensions), whereas machine learning models can build models in
# high-dimensional spaces.
#
# Another thing worth mentioning in this plot: if you are young (less than 25
# year-old roughly) or old (more than 70 year-old roughly) you tend to work
# less. This is a non-linear relationship between age and hours
# per week. Some machine learning models can only capture linear interactions so
# this may be a factor when deciding which model to chose.

# %% [markdown]
# In a machine-learning setting, we will use an algorithm to automatically
# decide what the "rules" should be in order to make predictions on new data.
# The following plot shows the rules a simple algorithm (called decision tree)
# creates:
#
# <img src="../figures/simple-decision-tree-adult-census.png" width=100%/>

# %% [markdown]
# * In the region `age < 28.5` (left region) the prediction is `low-income`. The blue color
#   indicates that the prediction is `low-income`. What is plotted is actually
#   the probability of the class `low-income` as predicted by the algorithm. So the dark
#   blue color indicates that the algorithm is quite sure about this prediction.
# * In the region `age > 28.5 AND hours-per-week < 40.5` (bottom-right region),
#   the prediction is `low-income`. Note that the blue is a bit lighter that for
#   the left region which means that the algorithm is not as certain in this
#   region.
# * In the region `age > 28.5 AND hours-per-week > 40.5` (top-right region),
#   the prediction is `low-income`. The probability of the class `low-income` is
#   very close to 0.5. In practice that means that the model is kind of saying "I
#   don't really know".

# %% [markdown]
# Comments: the decision tree was limited to have three regions. Also a model
# will be better in higher dimensions.

# %% [markdown]
#
# In this notebook we have:
# * loaded the data from a CSV file using `pandas`
# * looked at the kind of variables in the dataset, and differentiated
#   between categorical and numerical variables
# * inspected the data with `pandas`, `seaborn` and `pandas_profiling`. Data inspection
#   can allow you to decide whether using machine learning is appropriate for
#   your data and to highlight potential peculiarities in your data
#
# Key ideas discussed:
# * if your target variable is imbalanced (e.g., you have more samples from one
#   target category than another), you may need special techniques for machine
#   learning
# * having redundant (or highly correlated) columns can be a problem for
#   machine learning algorithms
# * some machine learning models can only capture linear interaction so be
#   aware of non-linear relationships in your data
# These hand-written rules# ---
# jupyter:
#   jupytext:
#     formats: python_scripts//py:percent,notebooks//ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.0
# ---

# %% [markdown]
# In this notebook, we will look at the necessary steps required before any machine learning takes place.
# * load the data
# * look at the variables in the dataset, in particular, differentiate
#   between numerical and categorical variables, which need different
#   preprocessing in most machine learning workflows
# * visualize the distribution of the variables to gain some insights into the dataset

# %%
# Inline plots
# %matplotlib inline

# plotting style
import seaborn as sns

# %% [markdown]
# ## Loading the adult census dataset

# %% [markdown]
# We will use data from the "Current Population adult_census" from 1994 that we
# downloaded from [OpenML](http://openml.org/).

# %%
import pandas as pd

adult_census = pd.read_csv(
    "https://www.openml.org/data/get_csv/1595261/adult-census.csv")

# Or use the local copy:
# adult_census = pd.read_csv('../datasets/adult-census.csv')

# %% [markdown]
# We can look at the OpenML webpage to know more about this dataset.

# %%
from IPython.display import IFrame
IFrame('https://www.openml.org/d/1590', width=1200, height=600)

# %% [markdown]
# ## Look at the variables in the dataset
# The data are stored in a pandas dataframe.

# %%
adult_census.head()

# %% [markdown]
# The column named **class** is our target variable (i.e., the variable which
# we want to predict). The two possible classes are `<= 50K` (low-revenue) and
# `> 50K` (high-revenue).

# %%
target_column = 'class'
adult_census[target_column].value_counts()

# %% [markdown]
# Note: classes are slightly imbalanced. Class imbalance happens often in
# practice and may need special techniques for machine learning. For example in
# a medical setting, if we are trying to predict whether patients will develop
# a rare disease, there will be a lot more healthy patients than ill patients in
# the dataset.

# %% [markdown]
# The dataset contains both numerical and categorical data. Numerical values
# can take continuous values for example `age`. Categorical values can have a
# finite number of values, for example `native-country`.

# %%
numerical_columns = [
    'age', 'education-num', 'capital-gain', 'capital-loss',
    'hours-per-week']
categorical_columns = [
    'workclass', 'education', 'marital-status', 'occupation',
    'relationship', 'race', 'sex', 'native-country']
all_columns = numerical_columns + categorical_columns + [
    target_column]

adult_census = adult_census[all_columns]

# %% [markdown]
# Note that for simplicity, we have ignored the "fnlwgt" (final weight) column
# that was crafted by the creators of the dataset when sampling the dataset to
# be representative of the full census database.

# %% [markdown]
# ## Inspect the data
# Before building a machine learning model, it is a good idea to look at the
# data:
# * maybe the task you are trying to achieve can be solved without machine
#   learning
# * you need to check that the data you need for your task is indeed present in
# the dataset
# * inspecting the data is a good way to find peculiarities. These can can
#   arise during data collection (for example, malfunctioning sensor or missing
#   values), or from the way the data is processed afterwards (for example capped
#   values).

# %% [markdown]
# Let's look at the distribution of individual variables, to get some insights
# about the data. We can start by plotting histograms, note that this only
# works for numerical variables:

# %%
_ = adult_census.hist(figsize=(20, 10))

# %% [markdown]
# We can already make a few comments about some of the variables:
# * age: there are not that many points for 'age > 70'. The dataset description
# does indicate that retired people have been filtered out (`hours-per-week > 0`).
# * education-num: peak at 10 and 13, hard to tell what it corresponds to
# without looking much further. We'll do that later in this notebook.
# * hours per week peaks at 40, this was very likely the standard number of
# working hours at the time of the data collection
# * most values of capital-gain and capital-loss are close to zero

# %% [markdown]
# For categorical variables, we can look at the distribution of values:


# %%
adult_census['sex'].value_counts()

# %%
adult_census['education'].value_counts()

# %% [markdown]
# `pandas_profiling` is a nice tool for inspecting the data (both numerical and
# categorical variables).

# %%
import pandas_profiling
adult_census.profile_report()

# %% [markdown]
# As noted above, `education-num` distribution has two clear peaks around 10
# and 13. It would be reasonable to expect that `education-num` is the number of
# years of education. Let's look at the relationship between `education` and
# `education-num`.
# %%
pd.crosstab(index=adult_census['education'],
            columns=adult_census['education-num'])

# %% [markdown]
# This shows that education and education-num gives you the same information.
# For example, `education-num=2` is equivalent to `education='1st-4th'`. In
# practice that means we can remove `education-num` without losing information.
# Note that having redundant (or highly correlated) columns can be a problem
# for machine learning algorithms.

# %% [markdown]
# Another way to inspect the data is to do a pairplot and show how each variable
# differs according to our target, `class`. Plots along the diagonal show the
# distribution of individual variables for each `class`. The plots on the
# off-diagonal can reveal interesting interactions between variables.

# %%
n_samples_to_plot = 5000
columns = ['age', 'education-num', 'hours-per-week']
_ = sns.pairplot(data=adult_census[:n_samples_to_plot], vars=columns,
                 hue=target_column, plot_kws={'alpha': 0.2},
                 height=4, diag_kind='hist')

# %%
_ = sns.pairplot(data=adult_census[:n_samples_to_plot], x_vars='age',
                 y_vars='hours-per-week', hue=target_column,
                 markers=['o',
                          'v'], plot_kws={'alpha': 0.2}, height=12)

# %% [markdown]
#
# By looking at the data you could infer some hand-written rules to predict the
# class:
# * if you are young (less than 25 year-old roughly), you are in the `<= 50K` class.
# * if you are old (more than 70 year-old roughly), you are in the `<= 50K` class.
# * if you work part-time (less than 40 hours roughly) you are in the `<= 50K` class.
#
# These hand-written rules could work reasonably well without the need for any
# machine learning. Note however that it is not very easy to create rules for
# the region `40 < hours-per-week < 60` and `30 < age < 70`. We can hope that
# machine learning can help in this region. Also note that visualization can
# help creating hand-written rules but is limited to 2 dimensions (maybe 3
# dimensions), whereas machine learning models can build models in
# high-dimensional spaces.
#
# Another thing worth mentioning in this plot: if you are young (less than 25
# year-old roughly) or old (more than 70 year-old roughly) you tend to work
# less. This is a non-linear relationship between age and hours
# per week. Some machine learning models can only capture linear interactions so
# this may be a factor when deciding which model to chose.

# %% [markdown]
# In a machine-learning setting, we will use an algorithm to automatically
# decide what the "rules" should be in order to make predictions on new data.
# The following plot shows the rules a simple algorithm (called decision tree)
# creates:
#
# <img src="../figures/simple-decision-tree-adult-census.png" width=100%/>

# %% [markdown]
# * In the region `age < 28.5` (left region) the prediction is `low-income`. The blue color
#   indicates that the prediction is `low-income`. What is plotted is actually
#   the probability of the class `low-income` as predicted by the algorithm. So the dark
#   blue color indicates that the algorithm is quite sure about this prediction.
# * In the region `age > 28.5 AND hours-per-week < 40.5` (bottom-right region),
#   the prediction is `low-income`. Note that the blue is a bit lighter that for
#   the left region which means that the algorithm is not as certain in this
#   region.
# * In the region `age > 28.5 AND hours-per-week > 40.5` (top-right region),
#   the prediction is `low-income`. The probability of the class `low-income` is
#   very close to 0.5. In practice that means that the model is kind of saying "I
#   don't really know".

# %% [markdown]
# Comments: the decision tree was limited to have three regions. Also a model
# will be better in higher dimensions.

# %% [markdown]
#
# In this notebook we have:
# * loaded the data from a CSV file using `pandas`
# * looked at the kind of variables in the dataset, and differentiated
#   between categorical and numerical variables
# * inspected the data with `pandas`, `seaborn` and `pandas_profiling`. Data inspection
#   can allow you to decide whether using machine learning is appropriate for
#   your data and to highlight potential peculiarities in your data
#
# Key ideas discussed:
# * if your target variable is imbalanced (e.g., you have more samples from one
#   target category than another), you may need special techniques for machine
#   learning
# * having redundant (or highly correlated) columns can be a problem for
#   machine learning algorithms
# * some machine learning models can only capture linear interaction so be
#   aware of non-linear relationships in your data could work reasonably well without the need for any
# machine learning. Note h# ---
# jupyter:
#   jupytext:
#     formats: python_scripts//py:percent,notebooks//ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.0
# ---

# %% [markdown]
# In this notebook, we will look at the necessary steps required before any machine learning takes place.
# * load the data
# * look at the variables in the dataset, in particular, differentiate
#   between numerical and categorical variables, which need different
#   preprocessing in most machine learning workflows
# * visualize the distribution of the variables to gain some insights into the dataset

# %%
# Inline plots
# %matplotlib inline

# plotting style
import seaborn as sns

# %% [markdown]
# ## Loading the adult census dataset

# %% [markdown]
# We will use data from the "Current Population adult_census" from 1994 that we
# downloaded from [OpenML](http://openml.org/).

# %%
import pandas as pd

adult_census = pd.read_csv(
    "https://www.openml.org/data/get_csv/1595261/adult-census.csv")

# Or use the local copy:
# adult_census = pd.read_csv('../datasets/adult-census.csv')

# %% [markdown]
# We can look at the OpenML webpage to know more about this dataset.

# %%
from IPython.display import IFrame
IFrame('https://www.openml.org/d/1590', width=1200, height=600)

# %% [markdown]
# ## Look at the variables in the dataset
# The data are stored in a pandas dataframe.

# %%
adult_census.head()

# %% [markdown]
# The column named **class** is our target variable (i.e., the variable which
# we want to predict). The two possible classes are `<= 50K` (low-revenue) and
# `> 50K` (high-revenue).

# %%
target_column = 'class'
adult_census[target_column].value_counts()

# %% [markdown]
# Note: classes are slightly imbalanced. Class imbalance happens often in
# practice and may need special techniques for machine learning. For example in
# a medical setting, if we are trying to predict whether patients will develop
# a rare disease, there will be a lot more healthy patients than ill patients in
# the dataset.

# %% [markdown]
# The dataset contains both numerical and categorical data. Numerical values
# can take continuous values for example `age`. Categorical values can have a
# finite number of values, for example `native-country`.

# %%
numerical_columns = [
    'age', 'education-num', 'capital-gain', 'capital-loss',
    'hours-per-week']
categorical_columns = [
    'workclass', 'education', 'marital-status', 'occupation',
    'relationship', 'race', 'sex', 'native-country']
all_columns = numerical_columns + categorical_columns + [
    target_column]

adult_census = adult_census[all_columns]

# %% [markdown]
# Note that for simplicity, we have ignored the "fnlwgt" (final weight) column
# that was crafted by the creators of the dataset when sampling the dataset to
# be representative of the full census database.

# %% [markdown]
# ## Inspect the data
# Before building a machine learning model, it is a good idea to look at the
# data:
# * maybe the task you are trying to achieve can be solved without machine
#   learning
# * you need to check that the data you need for your task is indeed present in
# the dataset
# * inspecting the data is a good way to find peculiarities. These can can
#   arise during data collection (for example, malfunctioning sensor or missing
#   values), or from the way the data is processed afterwards (for example capped
#   values).

# %% [markdown]
# Let's look at the distribution of individual variables, to get some insights
# about the data. We can start by plotting histograms, note that this only
# works for numerical variables:

# %%
_ = adult_census.hist(figsize=(20, 10))

# %% [markdown]
# We can already make a few comments about some of the variables:
# * age: there are not that many points for 'age > 70'. The dataset description
# does indicate that retired people have been filtered out (`hours-per-week > 0`).
# * education-num: peak at 10 and 13, hard to tell what it corresponds to
# without looking much further. We'll do that later in this notebook.
# * hours per week peaks at 40, this was very likely the standard number of
# working hours at the time of the data collection
# * most values of capital-gain and capital-loss are close to zero

# %% [markdown]
# For categorical variables, we can look at the distribution of values:


# %%
adult_census['sex'].value_counts()

# %%
adult_census['education'].value_counts()

# %% [markdown]
# `pandas_profiling` is a nice tool for inspecting the data (both numerical and
# categorical variables).

# %%
import pandas_profiling
adult_census.profile_report()

# %% [markdown]
# As noted above, `education-num` distribution has two clear peaks around 10
# and 13. It would be reasonable to expect that `education-num` is the number of
# years of education. Let's look at the relationship between `education` and
# `education-num`.
# %%
pd.crosstab(index=adult_census['education'],
            columns=adult_census['education-num'])

# %% [markdown]
# This shows that education and education-num gives you the same information.
# For example, `education-num=2` is equivalent to `education='1st-4th'`. In
# practice that means we can remove `education-num` without losing information.
# Note that having redundant (or highly correlated) columns can be a problem
# for machine learning algorithms.

# %% [markdown]
# Another way to inspect the data is to do a pairplot and show how each variable
# differs according to our target, `class`. Plots along the diagonal show the
# distribution of individual variables for each `class`. The plots on the
# off-diagonal can reveal interesting interactions between variables.

# %%
n_samples_to_plot = 5000
columns = ['age', 'education-num', 'hours-per-week']
_ = sns.pairplot(data=adult_census[:n_samples_to_plot], vars=columns,
                 hue=target_column, plot_kws={'alpha': 0.2},
                 height=4, diag_kind='hist')

# %%
_ = sns.pairplot(data=adult_census[:n_samples_to_plot], x_vars='age',
                 y_vars='hours-per-week', hue=target_column,
                 markers=['o',
                          'v'], plot_kws={'alpha': 0.2}, height=12)

# %% [markdown]
#
# By looking at the data you could infer some hand-written rules to predict the
# class:
# * if you are young (less than 25 year-old roughly), you are in the `<= 50K` class.
# * if you are old (more than 70 year-old roughly), you are in the `<= 50K` class.
# * if you work part-time (less than 40 hours roughly) you are in the `<= 50K` class.
#
# These hand-written rules could work reasonably well without the need for any
# machine learning. Note however that it is not very easy to create rules for
# the region `40 < hours-per-week < 60` and `30 < age < 70`. We can hope that
# machine learning can help in this region. Also note that visualization can
# help creating hand-written rules but is limited to 2 dimensions (maybe 3
# dimensions), whereas machine learning models can build models in
# high-dimensional spaces.
#
# Another thing worth mentioning in this plot: if you are young (less than 25
# year-old roughly) or old (more than 70 year-old roughly) you tend to work
# less. This is a non-linear relationship between age and hours
# per week. Some machine learning models can only capture linear interactions so
# this may be a factor when deciding which model to chose.

# %% [markdown]
# In a machine-learning setting, we will use an algorithm to automatically
# decide what the "rules" should be in order to make predictions on new data.
# The following plot shows the rules a simple algorithm (called decision tree)
# creates:
#
# <img src="../figures/simple-decision-tree-adult-census.png" width=100%/>

# %% [markdown]
# * In the region `age < 28.5` (left region) the prediction is `low-income`. The blue color
#   indicates that the prediction is `low-income`. What is plotted is actually
#   the probability of the class `low-income` as predicted by the algorithm. So the dark
#   blue color indicates that the algorithm is quite sure about this prediction.
# * In the region `age > 28.5 AND hours-per-week < 40.5` (bottom-right region),
#   the prediction is `low-income`. Note that the blue is a bit lighter that for
#   the left region which means that the algorithm is not as certain in this
#   region.
# * In the region `age > 28.5 AND hours-per-week > 40.5` (top-right region),
#   the prediction is `low-income`. The probability of the class `low-income` is
#   very close to 0.5. In practice that means that the model is kind of saying "I
#   don't really know".

# %% [markdown]
# Comments: the decision tree was limited to have three regions. Also a model
# will be better in higher dimensions.

# %% [markdown]
#
# In this notebook we have:
# * loaded the data from a CSV file using `pandas`
# * looked at the kind of variables in the dataset, and differentiated
#   between categorical and numerical variables
# * inspected the data with `pandas`, `seaborn` and `pandas_profiling`. Data inspection
#   can allow you to decide whether using machine learning is appropriate for
#   your data and to highlight potential peculiarities in your data
#
# Key ideas discussed:
# * if your target variable is imbalanced (e.g., you have more samples from one
#   target category than another), you may need special techniques for machine
#   learning
# * having redundant (or highly correlated) columns can be a problem for
#   machine learning algorithms
# * some machine learning models can only capture linear interaction so be
#   aware of non-linear relationships in your dataowever that it is not very easy to create rules for
# the region `40 < hours-p# ---
# jupyter:
#   jupytext:
#     formats: python_scripts//py:percent,notebooks//ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.0
# ---

# %% [markdown]
# In this notebook, we will look at the necessary steps required before any machine learning takes place.
# * load the data
# * look at the variables in the dataset, in particular, differentiate
#   between numerical and categorical variables, which need different
#   preprocessing in most machine learning workflows
# * visualize the distribution of the variables to gain some insights into the dataset

# %%
# Inline plots
# %matplotlib inline

# plotting style
import seaborn as sns

# %% [markdown]
# ## Loading the adult census dataset

# %% [markdown]
# We will use data from the "Current Population adult_census" from 1994 that we
# downloaded from [OpenML](http://openml.org/).

# %%
import pandas as pd

adult_census = pd.read_csv(
    "https://www.openml.org/data/get_csv/1595261/adult-census.csv")

# Or use the local copy:
# adult_census = pd.read_csv('../datasets/adult-census.csv')

# %% [markdown]
# We can look at the OpenML webpage to know more about this dataset.

# %%
from IPython.display import IFrame
IFrame('https://www.openml.org/d/1590', width=1200, height=600)

# %% [markdown]
# ## Look at the variables in the dataset
# The data are stored in a pandas dataframe.

# %%
adult_census.head()

# %% [markdown]
# The column named **class** is our target variable (i.e., the variable which
# we want to predict). The two possible classes are `<= 50K` (low-revenue) and
# `> 50K` (high-revenue).

# %%
target_column = 'class'
adult_census[target_column].value_counts()

# %% [markdown]
# Note: classes are slightly imbalanced. Class imbalance happens often in
# practice and may need special techniques for machine learning. For example in
# a medical setting, if we are trying to predict whether patients will develop
# a rare disease, there will be a lot more healthy patients than ill patients in
# the dataset.

# %% [markdown]
# The dataset contains both numerical and categorical data. Numerical values
# can take continuous values for example `age`. Categorical values can have a
# finite number of values, for example `native-country`.

# %%
numerical_columns = [
    'age', 'education-num', 'capital-gain', 'capital-loss',
    'hours-per-week']
categorical_columns = [
    'workclass', 'education', 'marital-status', 'occupation',
    'relationship', 'race', 'sex', 'native-country']
all_columns = numerical_columns + categorical_columns + [
    target_column]

adult_census = adult_census[all_columns]

# %% [markdown]
# Note that for simplicity, we have ignored the "fnlwgt" (final weight) column
# that was crafted by the creators of the dataset when sampling the dataset to
# be representative of the full census database.

# %% [markdown]
# ## Inspect the data
# Before building a machine learning model, it is a good idea to look at the
# data:
# * maybe the task you are trying to achieve can be solved without machine
#   learning
# * you need to check that the data you need for your task is indeed present in
# the dataset
# * inspecting the data is a good way to find peculiarities. These can can
#   arise during data collection (for example, malfunctioning sensor or missing
#   values), or from the way the data is processed afterwards (for example capped
#   values).

# %% [markdown]
# Let's look at the distribution of individual variables, to get some insights
# about the data. We can start by plotting histograms, note that this only
# works for numerical variables:

# %%
_ = adult_census.hist(figsize=(20, 10))

# %% [markdown]
# We can already make a few comments about some of the variables:
# * age: there are not that many points for 'age > 70'. The dataset description
# does indicate that retired people have been filtered out (`hours-per-week > 0`).
# * education-num: peak at 10 and 13, hard to tell what it corresponds to
# without looking much further. We'll do that later in this notebook.
# * hours per week peaks at 40, this was very likely the standard number of
# working hours at the time of the data collection
# * most values of capital-gain and capital-loss are close to zero

# %% [markdown]
# For categorical variables, we can look at the distribution of values:


# %%
adult_census['sex'].value_counts()

# %%
adult_census['education'].value_counts()

# %% [markdown]
# `pandas_profiling` is a nice tool for inspecting the data (both numerical and
# categorical variables).

# %%
import pandas_profiling
adult_census.profile_report()

# %% [markdown]
# As noted above, `education-num` distribution has two clear peaks around 10
# and 13. It would be reasonable to expect that `education-num` is the number of
# years of education. Let's look at the relationship between `education` and
# `education-num`.
# %%
pd.crosstab(index=adult_census['education'],
            columns=adult_census['education-num'])

# %% [markdown]
# This shows that education and education-num gives you the same information.
# For example, `education-num=2` is equivalent to `education='1st-4th'`. In
# practice that means we can remove `education-num` without losing information.
# Note that having redundant (or highly correlated) columns can be a problem
# for machine learning algorithms.

# %% [markdown]
# Another way to inspect the data is to do a pairplot and show how each variable
# differs according to our target, `class`. Plots along the diagonal show the
# distribution of individual variables for each `class`. The plots on the
# off-diagonal can reveal interesting interactions between variables.

# %%
n_samples_to_plot = 5000
columns = ['age', 'education-num', 'hours-per-week']
_ = sns.pairplot(data=adult_census[:n_samples_to_plot], vars=columns,
                 hue=target_column, plot_kws={'alpha': 0.2},
                 height=4, diag_kind='hist')

# %%
_ = sns.pairplot(data=adult_census[:n_samples_to_plot], x_vars='age',
                 y_vars='hours-per-week', hue=target_column,
                 markers=['o',
                          'v'], plot_kws={'alpha': 0.2}, height=12)

# %% [markdown]
#
# By looking at the data you could infer some hand-written rules to predict the
# class:
# * if you are young (less than 25 year-old roughly), you are in the `<= 50K` class.
# * if you are old (more than 70 year-old roughly), you are in the `<= 50K` class.
# * if you work part-time (less than 40 hours roughly) you are in the `<= 50K` class.
#
# These hand-written rules could work reasonably well without the need for any
# machine learning. Note however that it is not very easy to create rules for
# the region `40 < hours-per-week < 60` and `30 < age < 70`. We can hope that
# machine learning can help in this region. Also note that visualization can
# help creating hand-written rules but is limited to 2 dimensions (maybe 3
# dimensions), whereas machine learning models can build models in
# high-dimensional spaces.
#
# Another thing worth mentioning in this plot: if you are young (less than 25
# year-old roughly) or old (more than 70 year-old roughly) you tend to work
# less. This is a non-linear relationship between age and hours
# per week. Some machine learning models can only capture linear interactions so
# this may be a factor when deciding which model to chose.

# %% [markdown]
# In a machine-learning setting, we will use an algorithm to automatically
# decide what the "rules" should be in order to make predictions on new data.
# The following plot shows the rules a simple algorithm (called decision tree)
# creates:
#
# <img src="../figures/simple-decision-tree-adult-census.png" width=100%/>

# %% [markdown]
# * In the region `age < 28.5` (left region) the prediction is `low-income`. The blue color
#   indicates that the prediction is `low-income`. What is plotted is actually
#   the probability of the class `low-income` as predicted by the algorithm. So the dark
#   blue color indicates that the algorithm is quite sure about this prediction.
# * In the region `age > 28.5 AND hours-per-week < 40.5` (bottom-right region),
#   the prediction is `low-income`. Note that the blue is a bit lighter that for
#   the left region which means that the algorithm is not as certain in this
#   region.
# * In the region `age > 28.5 AND hours-per-week > 40.5` (top-right region),
#   the prediction is `low-income`. The probability of the class `low-income` is
#   very close to 0.5. In practice that means that the model is kind of saying "I
#   don't really know".

# %% [markdown]
# Comments: the decision tree was limited to have three regions. Also a model
# will be better in higher dimensions.

# %% [markdown]
#
# In this notebook we have:
# * loaded the data from a CSV file using `pandas`
# * looked at the kind of variables in the dataset, and differentiated
#   between categorical and numerical variables
# * inspected the data with `pandas`, `seaborn` and `pandas_profiling`. Data inspection
#   can allow you to decide whether using machine learning is appropriate for
#   your data and to highlight potential peculiarities in your data
#
# Key ideas discussed:
# * if your target variable is imbalanced (e.g., you have more samples from one
#   target category than another), you may need special techniques for machine
#   learning
# * having redundant (or highly correlated) columns can be a problem for
#   machine learning algorithms
# * some machine learning models can only capture linear interaction so be
#   aware of non-linear relationships in your dataer-week < 60` and `30 < age < 70`. We can hope that
# machine learning can help in this region. Also note that visualization can
# help creating hand-written rules but is limited to 2 dimensions (maybe 3
# dimensions), whereas machine learning models can build models in
# high-dimensional spaces.
#
# Another thing worth mentioning in this plot: if you are young (less than 25
# year-old roughly) or old (more than 70 year-old roughly) you tend to work
# less. This is a non-linear relationship between age and hours
# per week. Some machine learning models can only capture linear interactions so
# this may be a factor when deciding which model to chose.

# %% [markdown]
# In a machine-learning setting, we will use an algorithm to automatically
# decide what the "rules" should be in order to make predictions on new data.
# The following plot shows the rules a simple algorithm (called decision tree)
# creates:
#
# <img src="../figures/simple-decision-tree-adult-census.png" width=100%/>

# %% [markdown]
# * In the region `age < 28.5` (left region) the prediction is `low-income`. The blue color
#   indicates that the prediction is `low-income`. What is plotted is actually
#   the probability of the class `low-income` as predicted by the algorithm. So the dark
#   blue color indicates that the algorithm is quite sure about this prediction.
# * In the region `age > 28.5 AND hours-per-week < 40.5` (bottom-right region),
#   the prediction is `low-income`. Note that the blue is a bit lighter that for
#   the left region which means that the algorithm is not as certain in this
#   region.
# * In the region `age > 28.5 AND hours-per-week > 40.5` (top-right region),
#   the prediction is `low-income`. The probability of the class `low-income` is
#   very close to 0.5. In practice that means that the model is kind of saying "I
#   don't really know".

# %% [markdown]
# Comments: the decision tree was limited to have three regions. Also a model
# will be better in higher dimensions.

# %% [markdown]
#
# In this notebook we have:
# * loaded the data from a CSV file using `pandas`
# * looked at the kind of variables in the dataset, and differentiated
#   between categorical and numerical variables
# * inspected the data with `pandas`, `seaborn` and `pandas_profiling`. Data inspection
#   can allow you to decide whether using machine learning is appropriate for
#   your data and to highlight potential peculiarities in your data
#
# Key ideas discussed:
# * if your target variable is imbalanced (e.g., you have more samples from one
#   target category than another), you may need special techniques for machine
#   learning
# * having redundant (or highly correlated) columns can be a problem for
#   machine learning algorithms
# * some machine learning models can only capture linear interaction so be
#   aware of non-linear relationships in your data