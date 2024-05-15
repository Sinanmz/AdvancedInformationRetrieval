import os
import sys
current_script_path = os.path.abspath(__file__)
classification_dir = os.path.dirname(current_script_path)
core_dir = os.path.dirname(classification_dir)
Logic_dir = os.path.dirname(core_dir)
project_root = os.path.dirname(Logic_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

import numpy as np
from sklearn.metrics import classification_report
from sklearn.svm import SVC

from Logic.core.classification.basic_classifier import BasicClassifier
from Logic.core.classification.data_loader import ReviewLoader

class SVMClassifier(BasicClassifier):
    def __init__(self):
        self.model = SVC()

    def fit(self, x, y):
        """
        Parameters
        ----------
        x: np.ndarray
            An m * n matrix - m is count of docs and n is embedding size

        y: np.ndarray
            The real class label for each doc
        """
        self.model.fit(x, y)

    def predict(self, x):
        """
        Parameters
        ----------
        x: np.ndarray
            An k * n matrix - k is count of docs and n is embedding size
        Returns
        -------
        np.ndarray
            Return the predicted class for each doc
            with the highest probability (argmax)
        """
        return self.model.predict(x)

    def prediction_report(self, x, y):
        """
        Parameters
        ----------
        x: np.ndarray
            An k * n matrix - k is count of docs and n is embedding size
        y: np.ndarray
            The real class label for each doc
        Returns
        -------
        str
            Return the classification report
        """
        y_pred = self.predict(x)
        return classification_report(y, y_pred)


# F1 accuracy : 78%
if __name__ == '__main__':
    """
    Fit the model with the training data and predict the test data, then print the classification report
    """
    model = SVMClassifier()
    data_path = project_root + '/data/IMDB_Reviews.csv'
    loader = ReviewLoader(data_path)
    loader.load_data()
    X_train, X_test, y_train, y_test = loader.split_data(test_data_ratio=0.2, embeddings=True)
    model.fit(X_train, y_train)
    print(model.prediction_report(X_test, y_test))

# Outputs:
#               precision    recall  f1-score   support

#            0       0.86      0.86      0.86      4961
#            1       0.86      0.87      0.86      5039

#     accuracy                           0.86     10000
#    macro avg       0.86      0.86      0.86     10000
# weighted avg       0.86      0.86      0.86     10000