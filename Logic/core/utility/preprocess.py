import os
import sys
current_script_path = os.path.abspath(__file__)
utility_dir = os.path.dirname(current_script_path)
core_dir = os.path.dirname(utility_dir)
Logic_dir = os.path.dirname(core_dir)
project_root = os.path.dirname(Logic_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

import re
import nltk
import contractions
import unidecode
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import json


try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

class Preprocessor:

    def __init__(self, documents: list):
        """
        Initialize the class.

        Parameters
        ----------
        documents : list
            The list of documents to be preprocessed.
        """
        self.documents = documents

    def preprocess(self):
        """
        Preprocess the text using the methods in the class.

        Returns
        ----------
        List[str]
            The preprocessed documents.
        """
        preprocessed_docs = []
        for doc in self.documents:
            doc = self.expand_contractions(doc)
            doc = self.normalize(doc)
            doc = self.remove_links(doc)
            doc = self.remove_punctuations(doc)
            words = self.tokenize(doc)
            words = self.remove_stopwords(words)
            preprocessed_docs.append(' '.join(words))
        return preprocessed_docs

    def expand_contractions(self, text: str):
        return contractions.fix(text)

    def normalize(self, text: str):
        """
        Normalize the text by converting it to lower case and using lemmatization.

        Parameters
        ----------
        text : str
            The text to be normalized.

        Returns
        ----------
        str
            The normalized text.
        """
        self.lemmatizer = WordNetLemmatizer()
        text = text.lower()
        text = unidecode.unidecode(text)
        words = word_tokenize(text)
        normalized_text = ' '.join([self.lemmatizer.lemmatize(word) for word in words])
        return normalized_text

    def remove_links(self, text: str):
        """
        Remove links from the text.

        Parameters
        ----------
        text : str
            The text to be processed.

        Returns
        ----------
        str
            The text with links removed.
        """
        patterns = [r'\S*http\S*', r'\S*www\S*', r'\S+\.ir\S*', r'\S+\.com\S*', r'\S+\.org\S*', r'\S*@\S*']
        for pattern in patterns:
            text = re.sub(pattern, '', text)
        return text

    def remove_punctuations(self, text: str):
        """
        Remove punctuations from the text.

        Parameters
        ----------
        text : str
            The text to be processed.

        Returns
        ----------
        str
            The text with punctuations removed.
        """
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def tokenize(self, text: str):
        """
        Tokenize the words in the text.

        Parameters
        ----------
        text : str
            The text to be tokenized.

        Returns
        ----------
        list
            The list of words.
        """
        return word_tokenize(text)

    def remove_stopwords(self, words: list):
        """
        Remove stopwords from the list of words.

        Parameters
        ----------
        words : list
            The list of words to remove stopwords from.

        Returns
        ----------
        list
            The list of words with stopwords removed.
        """
        self.stopwords = set(stopwords.words('english'))
        stop_words_path = project_root+'/Logic/core/utility/stopwords.txt'
        with open(stop_words_path, 'r') as file:
            additional_stopwords = file.read().splitlines()
        self.stopwords.update(additional_stopwords)
        
        return [word for word in words if word not in self.stopwords]
    

if __name__ == '__main__':
    data_path = project_root+'/data/IMDB_Crawled.json'
    with open(data_path, 'r') as file:
        data = json.load(file)
    preprocessed_docs = []
    for doc in data:
        id = doc['id']

        stars = doc['stars']
        if stars == None:
            preprocessed_stars = []
        else:
            preprocessor = Preprocessor(stars)
            preprocessed_stars = preprocessor.preprocess()

        genres = doc['genres']
        if genres == None:
            preprocessed_genres = []
        else:
            preprocessor = Preprocessor(genres)
            preprocessed_genres = preprocessor.preprocess()

        summaries = doc['summaries']
        if summaries == None:
            preprocessed_sumaries = []
        else:
            preprocessor = Preprocessor(summaries)
            preprocessed_sumaries = preprocessor.preprocess()

        preprocessed_docs.append({
            'id': id,
            'stars': preprocessed_stars,
            'genres': preprocessed_genres,
            'summaries': preprocessed_sumaries
        })
    with open(project_root+'/data/IMDB_Preprocessed.json', 'w') as file:
        json.dump(preprocessed_docs, file)



    
    
