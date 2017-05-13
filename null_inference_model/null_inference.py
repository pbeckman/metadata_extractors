import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
import pickle as pkl
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from pylab import rcParams

from metadata_extractors.ML_util import clean_data, cross_validation, train_and_save

rcParams['figure.figsize'] = 10, 7

np.set_printoptions(threshold=np.nan)


def bin_null_values(y):
    """Bins null values into integer bins. This is necessary if using float nulls.

        :param y: (np.array) column vector of data
        :returns: (list, np.array) null values and column vector with index of null in list"""

    y_output = np.zeros(y.shape)

    nulls = [0]
    num_rows, num_cols = y.shape
    for i in range(0, num_rows):
        for j in range(0, num_cols):
            if y[i][j] != 0:
                if y[i][j] not in nulls:
                    nulls.append(y[i][j])
                y_output[i][j] = nulls.index(y[i][j])

    return nulls, y_output


def pca_plot(X, y):
    """Creates PCA plot of null inference feature space.

        :param X: (np.array) data matrix
        :param y: (np.array) true value column vector"""

    # This generates the plot used in the Skluma paper
    pca = PCA(n_components=2)
    X_fit_pca = pca.fit(X)
    X_r = X_fit_pca.transform(X)

    # plt.xlim(-716200, -713200)
    # plt.ylim(-24500, -23000)

    one = plt.scatter([X_r[i, 0] for i in range(0, 4813) if y[i] == 0],
                      [X_r[i, 1] for i in range(0, 4813) if y[i] == 0],
                      c='r', s=30, alpha=.8, lw=0.3)
    plt.scatter([X_r[i, 0] for i in range(0, 4813) if y[i] == 1],
                [X_r[i, 1] for i in range(0, 4813) if y[i] == 1],
                c='#0cd642', s=30, alpha=.8, lw=0.3)
    plt.scatter([X_r[i, 0] for i in range(0, 4813) if y[i] == 2],
                [X_r[i, 1] for i in range(0, 4813) if y[i] == 2],
                c='#4289f4', s=30, alpha=.8, lw=0.3)

    one.axes.get_xaxis().set_visible(False)
    one.axes.get_yaxis().set_visible(False)
    plt.savefig('null2.png', format='png', dpi=600)
    # plt.show()


def get_best_params(model, X, y):
    """Find the best parameters for the model.

            :param model: (sklearn.model) model to fit
            :param X: (np.array) data matrix
            :param y: (np.array) true value column vector
            :return: (dict) optimal parameters"""

    params = {"n_neighbors": np.arange(18, 21, 1),
              "metric": ["euclidean", "cityblock"],
              "weights": ['uniform', 'distance']
              }

    model = GridSearchCV(model, params)

    model.fit(X, y.reshape(y.shape[0], ))

    return model.best_params_


if __name__ == "__main__":
    data = pd.read_csv('col_metadata.csv')

    X = data.iloc[:, 3:-2].values
    y = data.iloc[:, -1:].values

    X, y = clean_data(X, y)

    # nulls, y = bin_null_values(y)
    # y = y.reshape(y.shape[0], )
    # print nulls

    model = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='cityblock',
                                 metric_params=None, n_jobs=1, n_neighbors=5,
                                 weights='distance')

    # print get_best_params(model, X, y)

    cross_validation(model, X, y, splits=10000)

    # train_and_save(model, X, y, "ni_model.pkl")
