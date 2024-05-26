import os
import sys
current_script_path = os.path.abspath(__file__)
word_embedding_dir = os.path.dirname(current_script_path)
core_dir = os.path.dirname(word_embedding_dir)
Logic_dir = os.path.dirname(core_dir)
project_root = os.path.dirname(Logic_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

import fasttext
import re
import string

from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from scipy.spatial import distance
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


from Logic.core.word_embedding.fasttext_data_loader import FastTextDataLoader

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

class FastText:
    """
    A class used to train a FastText model and generate embeddings for text data.

    Attributes
    ----------
    method : str
        The training method for the FastText model.
    model : fasttext.FastText._FastText
        The trained FastText model.
    """

    def __init__(self, preprocessor=preprocess_text, method='skipgram'):
        """
        Initializes the FastText with a preprocessor and a training method.

        Parameters
        ----------
        method : str, optional
            The training method for the FastText model.
        """
        self.method = method
        self.model = None
        self.preprocessor = preprocessor


    def train(self, texts, lr=0.05, dim=100, epoch=30, word_ngrams=5, do_preprocess=False):
        """
        Trains the FastText model with the given texts.

        Parameters
        ----------
        texts : list of str
            The texts to train the FastText model.
        """
        if do_preprocess:
            prepprocessed_texts = []
            for text in texts:
                prepprocessed_texts.append(self.preprocessor(str(text)))
            texts = prepprocessed_texts

        with open(project_root+"/data/temp_train_data.txt", "w", encoding="utf-8") as f:
            for text in texts:
                f.write(str(text) + "\n")

        self.model = fasttext.train_unsupervised(project_root+"/data/temp_train_data.txt", 
                                                 model=self.method, 
                                                 lr=lr, 
                                                 dim=dim, 
                                                 epoch=epoch, 
                                                 wordNgrams=word_ngrams)
        
        os.remove(project_root+"/data/temp_train_data.txt")
        return self.model

    def get_query_embedding(self, query, tf_idf_vectorizer=None, do_preprocess=False):
        """
        Generates an embedding for the given query.

        Parameters
        ----------
        query : str
            The query to generate an embedding for.
        tf_idf_vectorizer : sklearn.feature_extraction.text.TfidfVectorizer
            The TfidfVectorizer to transform the query.
        do_preprocess : bool, optional
            Whether to preprocess the query.

        Returns
        -------
        np.ndarray
            The embedding for the query.
        """
        if do_preprocess:
            query = self.preprocessor(query)

        tokens = query.split()

        if tf_idf_vectorizer is not None:
            tf_idf_weights = tf_idf_vectorizer.transform([' '.join(tokens)])
            query_embedding = np.zeros(self.model.get_dimension())
            for token in tokens:
                if token in self.model.words:
                    token_embedding = self.model.get_word_vector(token)
                    token_index = tf_idf_vectorizer.vocabulary_[token]
                    token_weight = tf_idf_weights[0, token_index]
                    query_embedding += token_weight * token_embedding
        else:
            embeddings = np.array([self.model.get_word_vector(token) for token in tokens if token in self.model.words])
            query_embedding = np.mean(embeddings, axis=0) if len(embeddings) > 0 else np.zeros(self.model.get_dimension())

        return query_embedding

    def analogy(self, word1, word2, word3):
        """
        Perform an analogy task: word1 is to word2 as word3 is to __.

        Args:
            word1 (str): The first word in the analogy.
            word2 (str): The second word in the analogy.
            word3 (str): The third word in the analogy.

        Returns:
            str: The word that completes the analogy.
        """
        # Obtain word embeddings for the words in the analogy
        # TODO
        word1_embedding = self.model.get_word_vector(word1)
        word2_embedding = self.model.get_word_vector(word2)
        word3_embedding = self.model.get_word_vector(word3)

        # Perform vector arithmetic
        # TODO
        result_vector = word2_embedding - word1_embedding + word3_embedding

        # Create a dictionary mapping each word in the vocabulary to its corresponding vector
        # TODO
        all_words = self.model.get_words()
        word_vectors = {word: self.model.get_word_vector(word) for word in all_words}

        # Exclude the input words from the possible results
        # TODO
        word_vectors.pop(word1)
        word_vectors.pop(word2)
        word_vectors.pop(word3)

        # Find the word whose vector is closest to the result vector
        # TODO
        closest_word = min(word_vectors.keys(), key=lambda word: distance.cosine(word_vectors[word], result_vector))
        return closest_word

    def save_model(self, path=project_root+'/models/FastText_model.bin'):
        """
        Saves the FastText model to a file.

        Parameters
        ----------
        path : str, optional
            The path to save the FastText model.
        """
        self.model.save_model(path)

    def load_model(self, path=project_root+'/models/FastText_model.bin'):
        """
        Loads the FastText model from a file.

        Parameters
        ----------
        path : str, optional
            The path to load the FastText model.
        """
        self.model = fasttext.load_model(path)

    def prepare(self, dataset, mode, save=False, path=project_root+'/models/FastText_model.bin'):
        """
        Prepares the FastText model.

        Parameters
        ----------
        dataset : list of str
            The dataset to train the FastText model.
        mode : str
            The mode to prepare the FastText model.
        """
        if mode == 'train':
            self.train(dataset)
        if mode == 'load':
            self.load_model(path)
        if save:
            self.save_model(path)

if __name__ == "__main__":
    ft_model = FastText(preprocessor=preprocess_text, method='skipgram')


    data_path = project_root+'/data/IMDB_Crawled.json'
    model_path = project_root+'/models/FastText_model.bin'
    if not os.path.exists(model_path):
        ft_data_loader = FastTextDataLoader(data_path)

        X, y = ft_data_loader.create_train_data()
        
        ft_model.prepare(dataset=X, mode="train", save=True)
    else:
        ft_model.prepare(dataset=None, mode="load", save=False, 
                         path=project_root+'/models/FastText_model.bin')

    print(10 * "*" + "Similarity" + 10 * "*")

    word = 'queen'
    neighbors = ft_model.model.get_nearest_neighbors(word, k=5)

    for neighbor in neighbors:
        print(f"Word: {neighbor[1]}, Similarity: {neighbor[0]}")

    print(10 * "*" + "Analogy" + 10 * "*")
    word1 = "man"
    word2 = "king"
    word3 = "woman"
    print(f"Similarity between {word1} and {word2} is like similarity between {word3} and {ft_model.analogy(word1, word2, word3)}")

# Outputs:

# **********Similarity**********
# Word: queens, Similarity: 0.7692434191703796
# Word: queenthe, Similarity: 0.7094634771347046
# Word: elizabeth, Similarity: 0.6992816925048828
# Word: princess, Similarity: 0.6784615516662598
# Word: tremaine, Similarity: 0.6622841358184814
# **********Analogy**********
# Similarity between man and king is like similarity between woman and queen