"""
Some simple visualizations on the iris data.
"""

from sklearn import datasets
from matplotlib import pyplot as plt
import style_figs

iris = datasets.load_iris()

# Plot the histograms of each class for each feature


X = iris.data
y = iris.target
for x, feature_name in zip(X.T, iris.feature_names):
    plt.figure(figsize=(3, 3))
    for this_y in [0, 1, 2]:
        plt.hist(x[y == this_y])
    style_figs.light_axis()
    feature_name = feature_name.replace(' ', '_')
    feature_name = feature_name.replace('(', '')
    feature_name = feature_name.replace(')', '')
    plt.savefig('iris_{}_hist.svg'.format(feature_name))


