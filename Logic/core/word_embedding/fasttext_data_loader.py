import os
import sys
current_script_path = os.path.abspath(__file__)
word_embedding_dir = os.path.dirname(current_script_path)
core_dir = os.path.dirname(word_embedding_dir)
Logic_dir = os.path.dirname(core_dir)
project_root = os.path.dirname(Logic_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import json
import numpy as np

import nltk
from nltk.corpus import stopwords
import string

def preprocess_text(text, minimum_length=1, stopword_removal=True, stopwords_domain=[], lower_case=True,
                       punctuation_removal=True):
    """
    preprocess text by removing stopwords, punctuations, and converting to lowercase, and also filter based on a min length
    for stopwords use nltk.corpus.stopwords.words('english')
    for punctuations use string.punctuation

    Parameters
    ----------
    text: str
        text to be preprocessed
    minimum_length: int
        minimum length of the token
    stopword_removal: bool
        whether to remove stopwords
    stopwords_domain: list
        list of stopwords to be removed base on domain
    lower_case: bool
        whether to convert to lowercase
    punctuation_removal: bool
        whether to remove punctuations
    """
    nltk.download('stopwords', quiet=True)

    if lower_case:
        text = text.lower()

    tokens = text.split()

    if punctuation_removal:
        text = ''.join([char for char in text if char not in string.punctuation])
    
    tokens = text.split()

    if stopword_removal:
        stop_words = set(stopwords.words('english')).union(set(stopwords_domain))
        tokens = [token for token in tokens if token not in stop_words]

    tokens = [token for token in tokens if len(token) >= minimum_length]

    return ' '.join(tokens)


class FastTextDataLoader:
    """
    This class is designed to load and pre-process data for training a FastText model.

    It takes the file path to a data source containing movie information (synopses, summaries, reviews, titles, genres) as input.
    The class provides methods to read the data into a pandas DataFrame, pre-process the text data, and create training data (features and labels)
    """
    def __init__(self, file_path=project_root + '/data/IMDB_Crawled.json'):
        """
        Initializes the FastTextDataLoader class with the file path to the data source.

        Parameters
        ----------
        file_path: str
            The path to the file containing movie information.
        """
        self.file_path = file_path


    def read_data_to_df(self):
        """
        Reads data from the specified file path and creates a pandas DataFrame containing movie information.

        You can use an IndexReader class to access the data based on document IDs.
        It extracts synopses, summaries, reviews, titles, and genres for each movie.
        The extracted data is then stored in a pandas DataFrame with appropriate column names.

        Returns
        ----------
            pd.DataFrame: A pandas DataFrame containing movie information (synopses, summaries, reviews, titles, genres).
        """
        with open(self.file_path, 'r') as file:
            data = json.load(file)
        
        df_data = []
        for movie in data:
            synopses = movie['synposis']
            summaries = movie['summaries']
            reviews = movie['reviews']
            # title = movie['title']
            genres = movie['genres']
            df_data.append([synopses, summaries, reviews, genres])
        df = pd.DataFrame(df_data, columns=['synopsis', 'summaries', 'reviews', 'genres'])


        return df


    def create_train_data(self):
        """
        Reads data using the read_data_to_df function, pre-processes the text data, and creates training data (features and labels).

        Returns:
            tuple: A tuple containing two NumPy arrays: X (preprocessed text data) and y (encoded genre labels).
        """
        df = self.read_data_to_df()
        X = []
        y = []
        le = LabelEncoder()
        df['genres'] = df['genres'].apply(lambda x: ' '.join(sorted(x)))
        df['genres'] = le.fit_transform(df['genres'])
        for index, row in tqdm(df.iterrows(), total=df.shape[0], desc='Creating training data'):
            inp = ""
            synopses = row['synopsis']
            if synopses:
                for synopsis in synopses:
                    inp += preprocess_text(synopsis) + "\n"
            summaries = row['summaries']
            if summaries:
                for summary in summaries:
                    inp += preprocess_text(summary) + "\n"
            reviews = row['reviews']
            if reviews:
                for review in reviews:
                    inp += preprocess_text(review[0]) + "\n"
                    
            X.append(inp)
            y.append(row['genres'])

        X = np.array(X)
        y = np.array(y)
        return X, y


