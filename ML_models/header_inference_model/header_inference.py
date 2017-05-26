import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from metadata_extractors.ML_models.ML_util import fill_zeros, get_text_rows, \
    train_and_save, cross_validation, get_best_params


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

    X = data.iloc[:, 3:].values
    y = np.asarray([[header] for header in data.iloc[:, 2].values])

    print "cleaning data"
    X, y = clean_data(X, y)

    model = RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=7,
                                   min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                                   max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07,
                                   bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0,
                                   warm_start=False, class_weight=None)

    # with open("header_model.pkl", "rb") as model_file:
    #     model = pkl.load(model_file)

    # print "best parameters:"
    # params = {}
    # print get_best_params(model, params, X, y)

    print "cross-validating model"
    cross_validation(model, X, y, splits=5, certainty_threshold=0.98)

    # print "training and saving model"
    # train_and_save(model, X, y, "header_model.pkl")
    # print "model saved"
