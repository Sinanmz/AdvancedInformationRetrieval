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
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from Logic.core.classification.basic_classifier import BasicClassifier
from Logic.core.classification.data_loader import ReviewLoader



class NaiveBayes(BasicClassifier):
    def __init__(self, count_vectorizer, alpha=1):
        # super().__init__()
        self.cv = count_vectorizer
        self.num_classes = None
        self.classes = None
        self.id2class = None
        self.number_of_features = None
        self.number_of_samples = None
        self.prior = None
        self.feature_probabilities = None
        self.log_probs = None
        self.alpha = alpha

    def fit(self, x, y):
        """
        Fit the features and the labels
        Calculate prior and feature probabilities

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
        self.num_classes = len(np.unique(y))
        self.classes = np.unique(y)
        self.id2class = {i: c for i, c in enumerate(self.classes)}
        self.cv.fit(x)
        x = self.cv.transform(x).toarray()
        self.number_of_features = x.shape[1]
        self.number_of_samples = x.shape[0]

        self.prior = np.zeros(self.num_classes)
        self.feature_probabilities = np.zeros((self.num_classes, self.number_of_features))

        for i, c in self.id2class.items():
            x_c = x[y == c]
            self.prior[i] = len(x_c) / self.number_of_samples
            self.feature_probabilities[i] = (np.sum(x_c, axis=0) + self.alpha) / (np.sum(x_c) + self.alpha * self.number_of_features)
        
        self.log_probs = np.log(self.feature_probabilities)
        self.prior = np.log(self.prior)
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
        x = self.cv.transform(x).toarray()
        probabilities = np.zeros((x.shape[0], self.num_classes))
        for i in range(self.num_classes):
            probabilities[:, i] = np.sum(x * self.log_probs[i] + (1 - x) * np.log(1 - np.exp(self.log_probs[i])), axis=1) + self.prior[i]
        return np.argmax(probabilities, axis=1)

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
        predictions = self.predict(x)
        return classification_report(y, predictions)

    def get_percent_of_positive_reviews(self, sentences):
        """
        You have to override this method because we are using a different embedding method in this class.
        """
        predictions = self.predict(sentences)
        positive_reviews = np.sum(predictions)
        return positive_reviews / len(predictions)
        




# F1 Accuracy : 85%
if __name__ == '__main__':
    """
    First, find the embeddings of the revies using the CountVectorizer, then fit the model with the training data.
    Finally, predict the test data and print the classification report
    You can use scikit-learn's CountVectorizer to find the embeddings.
    """
    model = NaiveBayes(count_vectorizer=CountVectorizer())
    data_path = project_root + '/data/IMDB_Reviews.csv'
    loader = ReviewLoader(data_path)
    loader.load_data()
    X_train, X_test, y_train, y_test = loader.split_data(test_data_ratio=0.2, embeddings=False)
    model.fit(X_train, y_train)
    print(model.prediction_report(X_test, y_test))

# Outputs:
#               precision    recall  f1-score   support

#            0       0.85      0.88      0.86      4961
#            1       0.87      0.85      0.86      5039

#     accuracy                           0.86     10000
#    macro avg       0.86      0.86      0.86     10000
# weighted avg       0.86      0.86      0.86     10000