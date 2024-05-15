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
import pandas as pd
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# from sklearn.feature_extraction.text import CountVectorizer


from Logic.core.word_embedding.fasttext_model import FastText
# from Logic.core.word_embedding.fasttext_data_loader import preprocess_text
from Logic.core.utility.preprocess import Preprocessor
 

class ReviewLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.fasttext_model = None
        self.review_tokens = []
        self.sentiments = []
        self.embeddings = []

    def load_data(self):
        """
        Load the data from the csv file and preprocess the text. Then save the normalized tokens and the sentiment labels.
        Also, load the fasttext model.
        """
        self.fasttext_model = FastText(preprocessor=None)

        data = pd.read_csv(self.file_path)
        data = data.dropna()
        data = data.reset_index(drop=True)
        preprocessor = Preprocessor(data["review"].values)
        data["review"] = preprocessor.preprocess()
        self.review_tokens = data["review"].values
        data["sentiment"] = data["sentiment"].map({"positive": 1, "negative": 0})
        self.sentiments = data["sentiment"].values


    def get_embeddings(self):
        """
        Get the embeddings for the reviews using the fasttext model.
        """
        if not os.path.exists(project_root+"/models/FastText_model.bin"):
            self.fasttext_model.prepare(dataset=self.review_tokens, mode="train", 
                                        save=True, path=project_root+"/models/FastText_model.bin")
        else:
            self.fasttext_model.prepare(dataset=self.review_tokens, mode="load", 
                                        save=False, path=project_root+"/models/FastText_model.bin")
        for review in tqdm(self.review_tokens, desc="Getting embeddings"):
            self.embeddings.append(self.fasttext_model.get_query_embedding(review))

        self.embeddings = np.array(self.embeddings)


    def split_data(self, test_data_ratio=0.2, embeddings=True):
        """
        Split the data into training and testing data.

        Parameters
        ----------
        test_data_ratio: float
            The ratio of the test data
        Returns
        -------
        np.ndarray, np.ndarray, np.ndarray, np.ndarray
            Return the training and testing data for the embeddings and the sentiments.
            in the order of x_train, x_test, y_train, y_test
        """
        if embeddings == False:
            embeddings = self.review_tokens
        else:
            self.get_embeddings()
            embeddings = self.embeddings
            print(embeddings.shape)

        X_train, X_test, y_train, y_test = train_test_split(embeddings, self.sentiments, test_size=test_data_ratio, random_state=42)
        return X_train, X_test, y_train, y_test
