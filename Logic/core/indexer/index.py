import os
import sys
current_script_path = os.path.abspath(__file__)
indexer_dir = os.path.dirname(current_script_path)
core_dir = os.path.dirname(indexer_dir)
Logic_dir = os.path.dirname(core_dir)
project_root = os.path.dirname(Logic_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from Logic.core.indexer.indexes_enum import Indexes

import time
import json
import copy


class Index:
    def __init__(self, preprocessed_documents: list):
        """
        Create a class for indexing.
        """

        self.preprocessed_documents = preprocessed_documents

        self.index = {
            Indexes.DOCUMENTS.value: self.index_documents(),
            Indexes.STARS.value: self.index_stars(),
            Indexes.GENRES.value: self.index_genres(),
            Indexes.SUMMARIES.value: self.index_summaries(),
        }

    def index_documents(self):
        """
        Index the documents based on the document ID. In other words, create a dictionary
        where the key is the document ID and the value is the document.

        Returns
        ----------
        dict
            The index of the documents based on the document ID.
        """

        current_index = {}
        # TODO

        for document in self.preprocessed_documents:
            current_index[document['id']] = document
        
        return current_index
    
    def index_stars(self):
        """
        Index the documents based on the stars.

        Returns
        ----------
        dict
            The index of the documents based on the stars. You should also store each terms' tf in each document.
            So the index type is: {term: {document_id: tf}}
        """

        # TODO
        stars_index = {}
        for document in self.preprocessed_documents:
            for star in document['stars']:
                if star == 'N/A':
                    continue
                for word in star.split():
                    if word not in stars_index:
                        stars_index[word] = {}
                    if document['id'] not in stars_index[word]:
                        stars_index[word][document['id']] = 0
                    stars_index[word][document['id']] += 1
        
        return stars_index

    def index_genres(self):
        """
        Index the documents based on the genres.

        Returns
        ----------
        dict
            The index of the documents based on the genres. You should also store each terms' tf in each document.
            So the index type is: {term: {document_id: tf}}
        """

        # TODO
        genres_index = {}
        for document in self.preprocessed_documents:
            for genre in document['genres']:
                if genre == 'N/A':
                    continue
                for word in genre.split():
                    if word not in genres_index:
                        genres_index[word] = {}
                    if document['id'] not in genres_index[word]:
                        genres_index[word][document['id']] = 0
                    genres_index[word][document['id']] += 1

        return genres_index

    def index_summaries(self):
        """
        Index the documents based on the summaries (not first_page_summary).

        Returns
        ----------
        dict
            The index of the documents based on the summaries. You should also store each terms' tf in each document.
            So the index type is: {term: {document_id: tf}}
        """

        current_index = {}
        # TODO
        for document in self.preprocessed_documents:
            for summary in document['summaries']:
                for word in summary.split():
                    if word not in current_index:
                        current_index[word] = {}
                    if document['id'] not in current_index[word]:
                        current_index[word][document['id']] = 0
                    current_index[word][document['id']] += 1

        return current_index

    def get_posting_list(self, word: str, index_type: str):
        """
        get posting_list of a word

        Parameters
        ----------
        word: str
            word we want to check
        index_type: str
            type of index we want to check (documents, stars, genres, summaries)

        Return
        ----------
        list
            posting list of the word (you should return the list of document IDs that contain the word and ignore the tf)
        """

        try:
            # TODO
            return list(self.index[index_type][word].keys())
        except:
            return []

    def add_document_to_index(self, document: dict):
        """
        Add a document to all the indexes

        Parameters
        ----------
        document : dict
            Document to add to all the indexes
        """

        # TODO
        self.index[Indexes.DOCUMENTS.value][document['id']] = document
        for star in document['stars']:
            if star == 'N/A':
                continue
            if star not in self.index[Indexes.STARS.value]:
                self.index[Indexes.STARS.value][star] = {}
            if document['id'] not in self.index[Indexes.STARS.value][star]:
                self.index[Indexes.STARS.value][star][document['id']] = 0
            self.index[Indexes.STARS.value][star][document['id']] += 1
        
        for genre in document['genres']:
            if genre == 'N/A':
                continue
            if genre not in self.index[Indexes.GENRES.value]:
                self.index[Indexes.GENRES.value][genre] = {}
            if document['id'] not in self.index[Indexes.GENRES.value][genre]:
                self.index[Indexes.GENRES.value][genre][document['id']] = 0
            self.index[Indexes.GENRES.value][genre][document['id']] += 1
        
        for summary in document['summaries']:
            for word in summary.split():
                if word not in self.index[Indexes.SUMMARIES.value]:
                    self.index[Indexes.SUMMARIES.value][word] = {}
                if document['id'] not in self.index[Indexes.SUMMARIES.value][word]:
                    self.index[Indexes.SUMMARIES.value][word][document['id']] = 0
                self.index[Indexes.SUMMARIES.value][word][document['id']] += 1

    def remove_document_from_index(self, document_id: str):
        """
        Remove a document from all the indexes

        Parameters
        ----------
        document_id : str
            ID of the document to remove from all the indexes
        """

        # TODO
        document = self.index[Indexes.DOCUMENTS.value][document_id]
        for star in document['stars']:
            if star == 'N/A':
                continue
            del self.index[Indexes.STARS.value][star][document_id]
            if len(self.index[Indexes.STARS.value][star]) == 0:
                del self.index[Indexes.STARS.value][star]
        for genre in document['genres']:
            if genre == 'N/A':
                continue
            del self.index[Indexes.GENRES.value][genre][document_id]
            if len(self.index[Indexes.GENRES.value][genre]) == 0:
                del self.index[Indexes.GENRES.value][genre]
        for summary in document['summaries']:
            for word in summary.split():
                del self.index[Indexes.SUMMARIES.value][word][document_id]
                if len(self.index[Indexes.SUMMARIES.value][word]) == 0:
                    del self.index[Indexes.SUMMARIES.value][word]
        
        del self.index[Indexes.DOCUMENTS.value][document_id]


    def check_add_remove_is_correct(self):
        """
        Check if the add and remove is correct
        """

        dummy_document = {
            'id': '100',
            'stars': ['tim', 'henry'],
            'genres': ['drama', 'crime'],
            'summaries': ['good']
        }

        index_before_add = copy.deepcopy(self.index)
        self.add_document_to_index(dummy_document)
        index_after_add = copy.deepcopy(self.index)

        if index_after_add[Indexes.DOCUMENTS.value]['100'] != dummy_document:
            print('Add is incorrect, document')
            return

        if (set(index_after_add[Indexes.STARS.value]['tim']).difference(set(index_before_add[Indexes.STARS.value]['tim']))
                != {dummy_document['id']}):
            print('Add is incorrect, tim')
            return

        if (set(index_after_add[Indexes.STARS.value]['henry']).difference(set(index_before_add[Indexes.STARS.value]['henry']))
                != {dummy_document['id']}):
            print('Add is incorrect, henry')
            return
        if (set(index_after_add[Indexes.GENRES.value]['drama']).difference(set(index_before_add[Indexes.GENRES.value]['drama']))
                != {dummy_document['id']}):
            print('Add is incorrect, drama')
            return

        if (set(index_after_add[Indexes.GENRES.value]['crime']).difference(set(index_before_add[Indexes.GENRES.value]['crime']))
                != {dummy_document['id']}):
            print('Add is incorrect, crime')
            return

        if (set(index_after_add[Indexes.SUMMARIES.value]['good']).difference(set(index_before_add[Indexes.SUMMARIES.value]['good']))
                != {dummy_document['id']}):
            print('Add is incorrect, good')
            return

        print('Add is correct')

        self.remove_document_from_index('100')
        index_after_remove = copy.deepcopy(self.index)

        if index_after_remove == index_before_add:
            print('Remove is correct')
        else:
            print('Remove is incorrect')

    def store_index(self, path: str, index_name: str = None):
        """
        Stores the index in a file (such as a JSON file)

        Parameters
        ----------
        path : str
            Path to store the file
        index_name: str
            name of index we want to store (documents, stars, genres, summaries)
        """

        if not os.path.exists(path):
            os.makedirs(path)

        if index_name not in self.index:
            raise ValueError('Invalid index name')

        # TODO
        # pass
        absolute_path = os.path.join(path, index_name + '_index.json')
        with open(absolute_path, 'w') as f:
            json.dump(self.index[index_name], f)
        

    def load_index(self, path: str):
        """
        Loads the index from a file (such as a JSON file)

        Parameters
        ----------
        path : str
            Path to load the file
        """

        # TODO
        # pass
        for index_type in self.index:
            with open(os.path.join(path, index_type + '_index.json'), 'r') as f:
                self.index[index_type] = json.load(f)

    def check_if_index_loaded_correctly(self, index_type: str, loaded_index: dict):
        """
        Check if the index is loaded correctly

        Parameters
        ----------
        index_type : str
            Type of index to check (documents, stars, genres, summaries)
        loaded_index : dict
            The loaded index

        Returns
        ----------
        bool
            True if index is loaded correctly, False otherwise
        """

        return self.index[index_type] == loaded_index

    def check_if_indexing_is_good(self, index_type: str, check_word: str = 'good'):
        """
        Checks if the indexing is good. Do not change this function. You can use this
        function to check if your indexing is correct.

        Parameters
        ----------
        index_type : str
            Type of index to check (documents, stars, genres, summaries)
        check_word : str
            The word to check in the index

        Returns
        ----------
        bool
            True if indexing is good, False otherwise
        """

        # brute force to check check_word in the summaries
        start = time.time()
        docs = []
        for document in self.preprocessed_documents:
            if index_type not in document or document[index_type] is None:
                continue

            for field in document[index_type]:
                if check_word in field.split():
                    docs.append(document['id'])
                    break

            # if we have found 3 documents with the word, we can break
            if len(docs) == 3:
                break

        end = time.time()
        brute_force_time = end - start

        # check by getting the posting list of the word
        start = time.time()
        # TODO: based on your implementation, you may need to change the following line
        posting_list = self.get_posting_list(check_word, index_type)

        end = time.time()
        implemented_time = end - start

        print('Brute force time: ', brute_force_time)
        print('Implemented time: ', implemented_time)

        if set(docs).issubset(set(posting_list)):
            print('Indexing is correct')

            if implemented_time < brute_force_time:
                print('Indexing is good')
                return True
            else:
                print('Indexing is bad')
                return False
        else:
            print('Indexing is wrong')
            return False

# TODO: Run the class with needed parameters, then run check methods and finally report the results of check 


if __name__ == '__main__':
    prepreprocessed_documents_path = 'data/IMDB_Preprocessed.json'
    with open(prepreprocessed_documents_path, 'r') as f:
        preprocessed_documents = json.load(f)
    
    index = Index(preprocessed_documents)

    print("Cheking add/remove:")
    index.check_add_remove_is_correct()

    index.store_index('index', Indexes.DOCUMENTS.value)
    index.store_index('index', Indexes.STARS.value)
    index.store_index('index', Indexes.GENRES.value)
    index.store_index('index', Indexes.SUMMARIES.value)

    index.load_index('index')
    print("Cheking if index loaded correctly (documents):")
    print(index.check_if_index_loaded_correctly(Indexes.DOCUMENTS.value, index.index[Indexes.DOCUMENTS.value]))
    print("Cheking if index loaded correctly (stars):")
    print(index.check_if_index_loaded_correctly(Indexes.STARS.value, index.index[Indexes.STARS.value]))
    print("Cheking if index loaded correctly (genres):")
    print(index.check_if_index_loaded_correctly(Indexes.GENRES.value, index.index[Indexes.GENRES.value]))
    print("Cheking if index loaded correctly (summaries):")
    print(index.check_if_index_loaded_correctly(Indexes.SUMMARIES.value, index.index[Indexes.SUMMARIES.value]))

    print("Cheking if indexing is good (documents):")
    print(index.check_if_indexing_is_good(Indexes.DOCUMENTS.value))
    print("Cheking if indexing is good (stars):")
    print(index.check_if_indexing_is_good(Indexes.STARS.value))
    print("Cheking if indexing is good (genres):")
    print(index.check_if_indexing_is_good(Indexes.GENRES.value))
    print("Cheking if indexing is good (summaries):")
    print(index.check_if_indexing_is_good(Indexes.SUMMARIES.value))

    # Outputs:
    
    # Cheking add/remove:
    # Add is correct
    # Remove is correct
    # Cheking if index loaded correctly (documents):
    # True
    # Cheking if index loaded correctly (stars):
    # True
    # Cheking if index loaded correctly (genres):
    # True
    # Cheking if index loaded correctly (summaries):
    # True
    # Cheking if indexing is good (documents):
    # Brute force time:  8.988380432128906e-05
    # Implemented time:  5.245208740234375e-06
    # Indexing is correct
    # Indexing is good
    # True
    # Cheking if indexing is good (stars):
    # Brute force time:  0.0013611316680908203
    # Implemented time:  2.86102294921875e-06
    # Indexing is correct
    # Indexing is good
    # True
    # Cheking if indexing is good (genres):
    # Brute force time:  0.0007009506225585938
    # Implemented time:  9.5367431640625e-07
    # Indexing is correct
    # Indexing is good
    # True
    # Cheking if indexing is good (summaries):
    # Brute force time:  5.888938903808594e-05
    # Implemented time:  5.245208740234375e-06
    # Indexing is correct
    # Indexing is good
    # True




