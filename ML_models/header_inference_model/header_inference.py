import pandas as pd
import numpy as np
from sklearn.svm import SVC

from metadata_extractors.ML_models.ML_util import fill_zeros, get_text_rows, cross_validation, get_best_params


def clean_data(X, y):
    """Removes textual rows and fills zeros.

        :param X: (np.array) data matrix
        :param y: (np.array) true value column vector
        :returns: (np.array) cleaned matrix ready for model"""

    to_remove = get_text_rows(X)
    X = np.delete(X, to_remove, axis=0)
    y = np.delete(y, to_remove, axis=0)

    X = fill_zeros(X)

    return X, y


if __name__ == "__main__":
    data = pd.read_csv('header_training_data.csv')

    X = data.iloc[1:10000, 3:].values
    y = np.asarray([[header] for header in data.iloc[1:10000, 2].values])

    print "cleaning data"
    X, y = clean_data(X, y)

    model = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
                max_iter=-1, probability=False, random_state=None, shrinking=True,
                tol=0.001, verbose=False)

    # with open("header_model.pkl", "rb") as model_file:
    #     model = pkl.load(model_file)

    print "cross-validating model"
    cross_validation(model, X, y, splits=10, decision_threshold=2)

    # print "training and saving model"
    # train_and_save(model, X, y, "header_model.pkl")
    # print "model saved"
