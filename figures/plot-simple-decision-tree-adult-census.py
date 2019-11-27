import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.colors import ListedColormap

import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

adult_census = pd.read_csv(
    "https://www.openml.org/data/get_csv/1595261/adult-census.csv")

target_column = 'class'

numerical_columns = [
    'age', 'education-num', 'capital-gain', 'capital-loss',
    'hours-per-week']
categorical_columns = [
    'workclass', 'education', 'marital-status', 'occupation',
    'relationship', 'race', 'sex', 'native-country']
all_columns = numerical_columns + categorical_columns + [
    target_column]

adult_census = adult_census[all_columns]

n_samples_to_plot = 5000
columns = ['age', 'education-num', 'hours-per-week']
_ = sns.pairplot(data=adult_census[:n_samples_to_plot], vars=columns,
                 hue=target_column, plot_kws={'alpha': 0.2},
                 height=4, diag_kind='hist')

_ = sns.pairplot(data=adult_census[:n_samples_to_plot], x_vars='age',
                 y_vars='hours-per-week', hue=target_column,
                 markers=['o',
                          'v'], plot_kws={'alpha': 0.2}, height=12)

top = cm.get_cmap('Oranges', 128)
bottom = cm.get_cmap('Blues_r', 128)

colors = np.vstack([bottom(np.linspace(0, 1, 128)),
                    top(np.linspace(0, 1, 128))])
blue_orange_cmap = ListedColormap(colors, name='BlueOrange')


def plot_tree_decision_function(tree, X, y, ax):
    """Plot the different decision rules found by a `DecisionTreeClassifier`.

    Parameters
    ----------
    tree : DecisionTreeClassifier instance
        The decision tree to inspect.
    X : dataframe of shape (n_samples, n_features)
        The data used to train the `tree` estimator.
    y : ndarray of shape (n_samples,)
        The target used to train the `tree` estimator.
    ax : matplotlib axis
        The matplotlib axis where to plot the different decision rules.
    """
    import numpy as np
    from scipy import ndimage

    h = 0.02
    x_min, x_max = 0, 100
    y_min, y_max = 0, 100
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = tree.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)
    faces = tree.tree_.apply(
        np.c_[xx.ravel(), yy.ravel()].astype(np.float32))
    faces = faces.reshape(xx.shape)
    border = ndimage.laplace(faces) != 0
    ax.scatter(X.iloc[:, 0], X.iloc[:, 1],
               c=np.array(['tab:blue',
                           'tab:orange'])[y], s=60, alpha=0.7, vmin=0, vmax=1)
    levels = np.linspace(0, 1, 101)
    contours = ax.contourf(xx, yy, Z, levels=levels, alpha=.4, cmap=blue_orange_cmap)
    ax.get_figure().colorbar(contours, ticks=np.linspace(0, 1, 11))
    ax.scatter(xx[border], yy[border], marker='.', s=1)
    ax.set_xlabel(X.columns[0])
    ax.set_ylabel(X.columns[1])
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    sns.despine(offset=10)


# select a subset of data
data_subset = adult_census[:n_samples_to_plot]
X = data_subset[["age", "hours-per-week"]]
y = LabelEncoder().fit_transform(
    data_subset[target_column].to_numpy())

max_leaf_nodes = 3
tree = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes,
                              random_state=0)
tree.fit(X, y)

# plot the decision function learned by the tree
fig, ax = plt.subplots()
plot_tree_decision_function(tree, X, y, ax=ax)

fig.savefig('simple-decision-tree-adult-census.png')
