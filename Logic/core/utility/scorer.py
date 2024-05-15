import os
import sys
current_script_path = os.path.abspath(__file__)
core_dir = os.path.dirname(current_script_path)
Logic_dir = os.path.dirname(core_dir)
project_root = os.path.dirname(Logic_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

import numpy as np


class Scorer:    
    def __init__(self, index, number_of_documents):
        """
        Initializes the Scorer.

        Parameters
        ----------
        index : dict
            The index to score the documents with.
        number_of_documents : int
            The number of documents in the index.
        """

        self.index = index
        self.idf = {}
        self.N = number_of_documents
        self.normalization_factor = {}

    def get_list_of_documents(self, query):
        """
        Returns a list of documents that contain at least one of the terms in the query.

        Parameters
        ----------
        query: List[str]
            The query to be scored

        Returns
        -------
        list
            A list of documents that contain at least one of the terms in the query.

        Note
        ---------
            The current approach is not optimal but we use it due to the indexing structure of the dict we're using.
            If we had pairs of (document_id, tf) sorted by document_id, we could improve this.
                We could initialize a list of pointers, each pointing to the first element of each list.
                Then, we could iterate through the lists in parallel.

        """
        list_of_documents = []
        for term in query:
            if term in self.index.keys():
                list_of_documents.extend(self.index[term].keys())
        return list(set(list_of_documents))

    def get_idf(self, term):
        """
        Returns the inverse document frequency of a term.

        Parameters
        ----------
        term : str
            The term to get the inverse document frequency for.

        Returns
        -------
        float
            The inverse document frequency of the term.

        Note
        -------
            It was better to store dfs in a separate dict in preprocessing.
        """
        idf = self.idf.get(term, None)
        if idf is None:
            # TODO
            # pass
            df = len(self.index.get(term, {}))
            idf = np.log((self.N)/(df+1))
        return idf

    def get_query_tfs(self, query):
        """
        Returns the term frequencies of the terms in the query.

        Parameters
        ----------
        query : List[str]
            The query to get the term frequencies for.

        Returns
        -------
        dict
            A dictionary of the term frequencies of the terms in the query.
        """
        
        #TODO
        query_tfs = {}
        for term in query:
            if term not in query_tfs:
                query_tfs[term] = 1
            else:
                query_tfs[term] += 1
        return query_tfs

        # TODO

    def compute_scores_with_vector_space_model(self, query, method):
        """
        compute scores with vector space model

        Parameters
        ----------
        query: List[str]
            The query to be scored
        method : str ((n|l)(n|t)(n|c).(n|l)(n|t)(n|c))
            The method to use for searching.

        Returns
        -------
        dict
            A dictionary of the document IDs and their scores.
        """

        # TODO
        # pass
        query_method = method.split('.')[0]
        document_method = method.split('.')[1]
        query_tfs = self.get_query_tfs(query)

        doc_list = self.get_list_of_documents(query)

        scores = {}
        for doc_id in doc_list:
            scores[doc_id] = self.get_vector_space_model_score(query, query_tfs, doc_id, document_method, query_method)

        return scores


    def get_vector_space_model_score(
        self, query, query_tfs, document_id, document_method, query_method
    ):
        """
        Returns the Vector Space Model score of a document for a query.

        Parameters
        ----------
        query: List[str]
            The query to be scored
        query_tfs : dict
            The term frequencies of the terms in the query.
        document_id : str
            The document to calculate the score for.
        document_method : str (n|l)(n|t)(n|c)
            The method to use for the document.
        query_method : str (n|l)(n|t)(n|c)
            The method to use for the query.

        Returns
        -------
        float
            The Vector Space Model score of the document for the query.
        """

        #TODO
        # pass
        query_vector = query_tfs

        if query_method[0] == 'l':
            for term, tf in query_vector.items():
                query_vector[term] = np.log(1+tf)
        if query_method[1] == 't':
            for term, tf in query_vector.items():
                query_vector[term] *= self.get_idf(term)
        if query_method[2] == 'c':
            norm = np.sqrt(sum([tf**2 for tf in query_vector.values()]))
            if norm != 0:
                for term in query_vector:
                    query_vector[term] /= norm
        
        
        doc_vector = {}
        for term in query:
            if term in self.index.keys():
                if document_id in self.index[term].keys():
                    doc_vector[term] = self.index[term][document_id]
                else:
                    doc_vector[term] = 0
            else:
                doc_vector[term] = 0
        if document_method[0] == 'l':
            for term, tf in doc_vector.items():
                doc_vector[term] = np.log(1+tf)
        if document_method[1] == 't':
            for term, tf in doc_vector.items():
                doc_vector[term] *= self.get_idf(term)
        if document_method[2] == 'c':
            norm = self.normalization_factor.get(document_id, 0)
            if norm != 0:
                for term in doc_vector:
                    doc_vector[term] /= norm
        
        
        score = 0.0
        for term in query_vector:
            score += query_vector[term]*doc_vector[term]
                
        return score

    def compute_socres_with_okapi_bm25(
        self, query, average_document_field_length, document_lengths
    ):
        """
        compute scores with okapi bm25

        Parameters
        ----------
        query: List[str]
            The query to be scored
        average_document_field_length : float
            The average length of the documents in the index.
        document_lengths : dict
            A dictionary of the document lengths. The keys are the document IDs, and the values are
            the document's length in that field.

        Returns
        -------
        dict
            A dictionary of the document IDs and their scores.
        """

        # TODO
        # pass
        doc_list = self.get_list_of_documents(query)

        scores = {}
        for doc_id in doc_list:
            scores[doc_id] = self.get_okapi_bm25_score(query, doc_id, average_document_field_length, document_lengths)

        return scores


    def get_okapi_bm25_score(
        self, query, document_id, average_document_field_length, document_lengths
    ):
        """
        Returns the Okapi BM25 score of a document for a query.

        Parameters
        ----------
        query: List[str]
            The query to be scored
        document_id : str
            The document to calculate the score for.
        average_document_field_length : float
            The average length of the documents in the index.
        document_lengths : dict
            A dictionary of the document lengths. The keys are the document IDs, and the values are
            the document's length in that field.

        Returns
        -------
        float
            The Okapi BM25 score of the document for the query.
        """

        # TODO

        k1 = 1.5  # Free parameter
        b = 0.75  # Free parameter

        score = 0.0
        document_length = document_lengths.get(document_id, 0)
        
        for term in query:
            idf = self.get_idf(term)
            tf = self.index.get(term, {}).get(document_id, 0)

            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (document_length / average_document_field_length))
            score += idf * (numerator / denominator)
        
        return score
        pass

    def compute_scores_with_unigram_model(
        self, query, smoothing_method, document_lengths=None, alpha=0.5, lamda=0.5
    ):
        """
        Calculates the scores for each document based on the unigram model.

        Parameters
        ----------
        query : str
            The query to search for.
        smoothing_method : str (bayes | naive | mixture)
            The method used for smoothing the probabilities in the unigram model.
        document_lengths : dict
            A dictionary of the document lengths. The keys are the document IDs, and the values are
            the document's length in that field.
        alpha : float, optional
            The parameter used in bayesian smoothing method. Defaults to 0.5.
        lamda : float, optional
            The parameter used in some smoothing methods to balance between the document
            probability and the collection probability. Defaults to 0.5.

        Returns
        -------
        float
            A dictionary of the document IDs and their scores.
        """

        # TODO

        doc_list = self.get_list_of_documents(query)

        scores = {}
        for doc_id in doc_list:
            scores[doc_id] = self.compute_score_with_unigram_model(query, doc_id, smoothing_method, document_lengths, alpha, lamda)

        return scores
        

    def compute_score_with_unigram_model(
        self, query, document_id, smoothing_method, document_lengths, alpha, lamda
    ):
        """
        Calculates the scores for each document based on the unigram model.

        Parameters
        ----------
        query : str
            The query to search for.
        document_id : str
            The document to calculate the score for.
        smoothing_method : str (bayes | naive | mixture)
            The method used for smoothing the probabilities in the unigram model.
        document_lengths : dict
            A dictionary of the document lengths. The keys are the document IDs, and the values are
            the document's length in that field.
        alpha : float, optional
            The parameter used in bayesian smoothing method. Defaults to 0.5.
        lamda : float, optional
            The parameter used in some smoothing methods to balance between the document
            probability and the collection probability. Defaults to 0.5.

        Returns
        -------
        float
            The Unigram score of the document for the query.
        """

        # TODO
        
        document_length = document_lengths.get(document_id, 0)
        score = 0.0
        for term in query:
            tf = self.index.get(term, {}).get(document_id, 0)
            cf = sum(self.index.get(term, {}).values())
            if smoothing_method == 'bayes':
                score += np.log((tf + alpha * (cf / self.N)) / (document_length + alpha))
            elif smoothing_method == 'naive':
                score += np.log((tf + lamda) / (document_length + lamda))
            elif smoothing_method == 'mixture':
                doc_probability = tf / document_length if document_length > 0 else 0
                collection_probability = cf / self.N
                score += np.log(lamda * doc_probability + (1 - lamda) * collection_probability)
            else:
                raise ValueError("Invalid smoothing method.")
        return score

        
