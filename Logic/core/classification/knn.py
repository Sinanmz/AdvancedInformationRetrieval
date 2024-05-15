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
from tqdm import tqdm

from Logic.core.classification.basic_classifier import BasicClassifier
from Logic.core.classification.data_loader import ReviewLoader


class KnnClassifier(BasicClassifier):
    def __init__(self, n_neighbors):
        self.k = n_neighbors

    def fit(self, x, y):
        """
        Fit the model using X as training data and y as target values
        use the Euclidean distance to find the k nearest neighbors
        Warning: Maybe you need to reduce the size of X to avoid memory errors

        Parameters
        ----------
        x: np.ndarray
            An m * n matrix - m is count of docs and n is embedding size
        y: np.ndarray
            The real class label for each doc
        Returns
        -------
        self
            Returns self as a classifier
        """
        self.x_train = x
        self.y_train = y
        return self
        

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
        predictions = []
        for i in tqdm(range(x.shape[0]), desc="Predicting"):
            distances = np.linalg.norm(self.x_train - x[i], axis=1)
            nearest_neighbors = np.argsort(distances)[:self.k]
            nearest_labels = self.y_train[nearest_neighbors]
            predictions.append(np.bincount(nearest_labels).argmax())
        return np.array(predictions)

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


# F1 Accuracy : 70%
if __name__ == '__main__':
    """
    Fit the model with the training data and predict the test data, then print the classification report
    """
    model = KnnClassifier(n_neighbors=5)
    data_path = project_root + '/data/IMDB_Reviews.csv'
    loader = ReviewLoader(data_path)
    loader.load_data()
    X_train, X_test, y_train, y_test = loader.split_data(test_data_ratio=0.2, embeddings=True)
    model.fit(X_train, y_train)
    print(model.prediction_report(X_test, y_test))

# Outputs
#               precision    recall  f1-score   support

#            0       0.79      0.84      0.81      4961
#            1       0.83      0.78      0.80      5039

#     accuracy                           0.81     10000
#    macro avg       0.81      0.81      0.81     10000
# weighted avg       0.81      0.81      0.81     10000