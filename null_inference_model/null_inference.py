import pandas as pd
import itertools
from math import isnan
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit, StratifiedKFold
import scipy as sp
import pickle as pkl
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from pylab import rcParams

rcParams['figure.figsize'] = 10, 7

np.set_printoptions(threshold=np.nan)


def is_number_or_none(field):
    """Determine if a string is a number or NaN by attempting to cast to it a float.

        :param field: (str) field
        :returns: (bool) whether field can be cast to a number"""

    if field is None:
        return True
    try:
        float(field)
        return True
    except ValueError:
        return False


def get_text_rows(matrix):
    """Get indices of all rows that have non-numerical aggregates.

        :param matrix: (np.array) matrix of data
        :returns: (list(int)) list of text indices to remove"""

    to_remove = []
    for i in range(0, len(matrix)):
        if not np.vectorize(is_number_or_none)(matrix[i]).all() \
                or np.vectorize(lambda x: str(x).lower() == "nan" or x is None)(matrix[i]).all():
            to_remove.append(i)

    return to_remove


def fill_zeros(matrix):
    """Fills all NaN and infinite entries with zeros.

        :param matrix: (np.array) matrix of data
        :returns: (np.array) matrix with zeros filled"""

    num_rows, num_cols = matrix.shape
    output_matrix = np.empty(matrix.shape)
    for i in range(0, num_rows):
        for j in range(0, num_cols):
            if matrix[i][j] is None or isnan(float(matrix[i][j])) or float(matrix[i][j]) == float('inf'):
                output_matrix[i][j] = np.float64(0)
            else:
                output_matrix[i][j] = matrix[i][j]

    return output_matrix


def clean_data(X, y=None):
    """Removes textual rows and fills zeros.

        :param X: (np.array) data matrix
        :param y: (np.array) true value column vector
        :returns: (np.array) cleaned matrix ready for model"""

    to_remove = get_text_rows(X)
    X = np.delete(X, to_remove, axis=0)
    X = fill_zeros(X)
    if y is not None:
        y = np.delete(y, to_remove, axis=0)
        return X, y
    else:
        return X


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


def cross_validation(model, X, y, splits=1000):
    """Runs cross-validation on a test set to find accuracy of model and prints results.

        :param model: (sklearn.model) model to test
        :param X: (np.array) data matrix
        :param y: (np.array) true value column vector
        :param splits: (int) number of times to perform cross-validation
        :returns: (list, np.array) null values and column vector with index of null in list"""

    all_y_test = np.zeros((0, 1))
    all_y_pred = np.zeros((0, 1))

    for train_inds, test_inds in ShuffleSplit(n_splits=splits, test_size=0.01).split(X, y):
        # Split off the train and test set
        X_test, y_test = X[test_inds, :], y[test_inds]
        X_train, y_train = X[train_inds, :], y[train_inds]

        # Train the model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test).reshape(-1, 1)

        # Append the results
        all_y_test = np.concatenate((all_y_test, y_test))
        all_y_pred = np.concatenate((all_y_pred, y_pred))

    print "accuracy: {}\nprecision: {}\nrecall: {}".format(
        accuracy_score(all_y_test, all_y_pred),
        precision_score(all_y_test, all_y_pred, average='macro'),
        recall_score(all_y_test, all_y_pred, average='macro')
    )


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


def train_and_save(model, X, y, file_name):
    """Train the model on the input data and save it for use in the pipeline.

        :param model: (sklearn.model) model to test
        :param X: (np.array) data matrix
        :param y: (np.array) true value column vector
        :param file_name: (string) file name of model"""

    model.fit(X, y)
    with open(file_name, "wb") as f:
        pkl.dump(model, f)


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
