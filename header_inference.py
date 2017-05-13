import numpy as np
import pandas as pd
import pickle as pkl
from sklearn.svm import SVC
from metadata_extractors.ML_util import clean_data, cross_validation, train_and_save


if __name__ == "__main__":
    data = pd.read_csv('null_inference_model/col_metadata.csv')

    X = data.iloc[:, 3:-2].values
    y = [[header[0].lower()] for header in data.iloc[:, 2:3].values]

    X, y = clean_data(X, y)

    model = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
                max_iter=-1, probability=False, random_state=None, shrinking=True,
                tol=0.001, verbose=False)

    # with open("header_model.pkl", "rb") as model_file:
    #     model = pkl.load(model_file)

    cross_validation(model, X, y, splits=10, decision_threshold=0.5)

    # train_and_save(model, X, y, "header_model.pkl")
