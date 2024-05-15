import os
import sys
current_script_path = os.path.abspath(__file__)
core_dir = os.path.dirname(current_script_path)
Logic_dir = os.path.dirname(core_dir)
project_root = os.path.dirname(Logic_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from Logic.core.utility.preprocess import Preprocessor
from Logic.core.utility.scorer import Scorer
from Logic.core.indexer.indexes_enum import Indexes, Index_types
from Logic.core.indexer.index_reader import Index_reader

import json
import numpy as np


class SearchEngine:
    def __init__(self):
        """
        Initializes the search engine.

        """
        path = project_root+'/index/'
        self.document_indexes = {
            Indexes.STARS: Index_reader(path, Indexes.STARS).index,
            Indexes.GENRES: Index_reader(path, Indexes.GENRES).index,
            Indexes.SUMMARIES: Index_reader(path, Indexes.SUMMARIES).index
        }
        self.tiered_index = {
            Indexes.STARS: Index_reader(path, Indexes.STARS, Index_types.TIERED).index,
            Indexes.GENRES: Index_reader(path, Indexes.GENRES, Index_types.TIERED).index,
            Indexes.SUMMARIES: Index_reader(path, Indexes.SUMMARIES, Index_types.TIERED).index
        }
        self.document_lengths_index = {
            Indexes.STARS: Index_reader(path, Indexes.STARS, Index_types.DOCUMENT_LENGTH).index,
            Indexes.GENRES: Index_reader(path, Indexes.GENRES, Index_types.DOCUMENT_LENGTH).index,
            Indexes.SUMMARIES: Index_reader(path, Indexes.SUMMARIES, Index_types.DOCUMENT_LENGTH).index
        }
        self.metadata_index = Index_reader(path, Indexes.DOCUMENTS, Index_types.METADATA).index

        number_of_documents = self.metadata_index['document_count']

        # My codes:
        # Calculating idf for each term in each field:
        self.idf = {}
        all_terms = {
            Indexes.STARS: set(),
            Indexes.GENRES: set(),
            Indexes.SUMMARIES: set()
        }
        for field in self.document_indexes:
            for term in self.document_indexes[field]:
                all_terms[field].add(term)
        self.idf = {
            Indexes.STARS: {},
            Indexes.GENRES: {},
            Indexes.SUMMARIES: {}
        }
        for field in all_terms:
            for term in all_terms[field]:
                df = len(self.document_indexes[field].get(term, {}))
                if df == 0:
                    self.idf[field][term] = 0
                else:
                    self.idf[field][term] = np.log((number_of_documents)/(df))


        # Calculating normalization factor for each document in each field:
        self.normalization_factors_idf = {
            Indexes.STARS: {},
            Indexes.GENRES: {},
            Indexes.SUMMARIES: {}
        }
        self.normalization_factors_no_idf = {
            Indexes.STARS: {},
            Indexes.GENRES: {},
            Indexes.SUMMARIES: {}
        }
        for field in [Indexes.STARS, Indexes.GENRES, Indexes.SUMMARIES]:
            for term in self.document_indexes[field]:
                for doc_id in self.document_indexes[field][term]:
                    if doc_id not in self.normalization_factors_no_idf[field]:
                        self.normalization_factors_no_idf[field][doc_id] = 0
                    self.normalization_factors_no_idf[field][doc_id] += self.document_indexes[field][term][doc_id] ** 2
                    if doc_id not in self.normalization_factors_idf[field]:
                        self.normalization_factors_idf[field][doc_id] = 0
                    self.normalization_factors_idf[field][doc_id] += (self.document_indexes[field][term][doc_id] * self.idf[field].get(term, 0)) ** 2
        for field in [Indexes.STARS, Indexes.GENRES, Indexes.SUMMARIES]:
            for doc_id in self.normalization_factors_no_idf[field]:
                self.normalization_factors_no_idf[field][doc_id] = np.sqrt(self.normalization_factors_no_idf[field][doc_id])
            for doc_id in self.normalization_factors_idf[field]:
                self.normalization_factors_idf[field][doc_id] = np.sqrt(self.normalization_factors_idf[field][doc_id])



    def search(
        self,
        query,
        method,
        weights,
        safe_ranking=True,
        max_results=10,
        smoothing_method=None,
        alpha=0.5,
        lamda=0.5,
    ):
        """
        searches for the query in the indexes.

        Parameters
        ----------
        query : str
            The query to search for.
        method : str ((n|l)(n|t)(n|c).(n|l)(n|t)(n|c)) | OkapiBM25 | Unigram
            The method to use for searching.
        weights: dict
            The weights of the fields.
        safe_ranking : bool
            If True, the search engine will search in whole index and then rank the results.
            If False, the search engine will search in tiered index.
        max_results : int
            The maximum number of results to return. If None, all results are returned.
        smoothing_method : str (bayes | naive | mixture)
            The method used for smoothing the probabilities in the unigram model.
        alpha : float, optional
            The parameter used in bayesian smoothing method. Defaults to 0.5.
        lamda : float, optional
            The parameter used in some smoothing methods to balance between the document
            probability and the collection probability. Defaults to 0.5.

        Returns
        -------
        list
            A list of tuples containing the document IDs and their scores sorted by their scores.
        """
        preprocessor = Preprocessor([query])
        query = preprocessor.preprocess()[0].split()

        scores = {}
        if method == "unigram":
            self.find_scores_with_unigram_model(
                query, smoothing_method, weights, scores, alpha, lamda
            )
        elif safe_ranking:
            self.find_scores_with_safe_ranking(
                query, method, weights, scores
            )
        else:
            self.find_scores_with_unsafe_ranking(
                query, method, weights, max_results, scores
            )

        final_scores = {}

        self.aggregate_scores(weights, scores, final_scores)

        result = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        if max_results is not None:
            result = result[:max_results]

        return result

    def aggregate_scores(self, weights, scores, final_scores):
        """
        Aggregates the scores of the fields.

        Parameters
        ----------
        weights : dict
            The weights of the fields.
        scores : dict
            The scores of the fields.
        final_scores : dict
            The final scores of the documents.
        """
        # TODO
        # pass
        all_doc_ids = set()
        for field in scores:
            all_doc_ids = all_doc_ids.union(set(scores[field].keys()))

        for field in weights:
            for doc_id in all_doc_ids:
                if doc_id not in final_scores:
                    final_scores[doc_id] = 0
                if doc_id in scores[field]:
                    final_scores[doc_id] += weights[field] * scores[field].get(doc_id, 0)

    def find_scores_with_unsafe_ranking(
        self, query, method, weights, max_results, scores
    ):
        """
        Finds the scores of the documents using the unsafe ranking method using the tiered index.

        Parameters
        ----------
        query: List[str]
            The query to be scored
        method : str ((n|l)(n|t)(n|c).(n|l)(n|t)(n|c)) | OkapiBM25
            The method to use for searching.
        weights: dict
            The weights of the fields.
        max_results : int
            The maximum number of results to return.
        scores : dict
            The scores of the documents.
        """
        number_of_documents = self.metadata_index['document_count']
        
        for tier in ["first_tier", "second_tier", "third_tier"]:
            for field in weights:
                #TODO
                # pass
                scorer = Scorer(self.tiered_index[field][tier], number_of_documents)
                scorer.idf = self.idf[field]                    
                        
                if method == 'OkapiBM25':
                    average_document_field_length = self.metadata_index['averge_document_length'][field.value]
                    document_lengths = self.document_lengths_index[field]

                    temp_scores = scorer.compute_socres_with_okapi_bm25(query, average_document_field_length, document_lengths)
                else:
                    if method[5] == 't':
                        scorer.normalization_factor = self.normalization_factors_idf[field]
                    else:
                        scorer.normalization_factor = self.normalization_factors_no_idf[field]
                    temp_scores = scorer.compute_scores_with_vector_space_model(query, method)
                
        
                if field not in scores:
                    scores[field] = temp_scores
                else:
                    for doc_id in temp_scores:
                        if doc_id not in scores[field]:
                            scores[field][doc_id] = 0
                        scores[field][doc_id] += temp_scores[doc_id]
            
            aggregated_scores = {}
            for field in weights:
                for doc_id in scores[field]:
                    if doc_id not in aggregated_scores:
                        aggregated_scores[doc_id] = 0
                    aggregated_scores[doc_id] += scores[field][doc_id]

            retrieved = set()
            for doc_id in aggregated_scores:
                if aggregated_scores[doc_id] > 0.0:
                    retrieved.add(doc_id)

            if len(retrieved) >= max_results:
                break

    def find_scores_with_safe_ranking(self, query, method, weights, scores):
        """
        Finds the scores of the documents using the safe ranking method.

        Parameters
        ----------
        query: List[str]
            The query to be scored
        method : str ((n|l)(n|t)(n|c).(n|l)(n|t)(n|c)) | OkapiBM25
            The method to use for searching.
        weights: dict
            The weights of the fields.
        scores : dict
            The scores of the documents.
        """

        number_of_documents = self.metadata_index['document_count']

        for field in weights:
            #TODO
            scorer = Scorer(self.document_indexes[field], number_of_documents)
            scorer.idf = self.idf[field]                
            if method == 'OkapiBM25':
                average_document_field_length = self.metadata_index['averge_document_length'][field.value]
                document_lengths = self.document_lengths_index[field]
                scores[field] = scorer.compute_socres_with_okapi_bm25(query, average_document_field_length, document_lengths)
            else:
                if method[5] == 't':
                    scorer.normalization_factor = self.normalization_factors_idf[field]
                else:
                    scorer.normalization_factor = self.normalization_factors_no_idf[field]
                scores[field] = scorer.compute_scores_with_vector_space_model(query, method)
        

    def find_scores_with_unigram_model(
        self, query, smoothing_method, weights, scores, alpha=0.5, lamda=0.5
    ):
        """
        Calculates the scores for each document based on the unigram model.

        Parameters
        ----------
        query : str
            The query to search for.
        smoothing_method : str (bayes | naive | mixture)
            The method used for smoothing the probabilities in the unigram model.
        weights : dict
            A dictionary mapping each field (e.g., 'stars', 'genres', 'summaries') to its weight in the final score. Fields with a weight of 0 are ignored.
        scores : dict
            The scores of the documents.
        alpha : float, optional
            The parameter used in bayesian smoothing method. Defaults to 0.5.
        lamda : float, optional
            The parameter used in some smoothing methods to balance between the document
            probability and the collection probability. Defaults to 0.5.
        """
        # TODO
        # pass
        number_of_documents = self.metadata_index['document_count']
        for field in weights:
            scorer = Scorer(self.document_indexes[field], number_of_documents)
            document_lengths = self.document_lengths_index[field]
            scores[field] = scorer.compute_scores_with_unigram_model(query, smoothing_method, document_lengths, alpha, lamda)


    def merge_scores(self, scores1, scores2):
        """
        Merges two dictionaries of scores.

        Parameters
        ----------
        scores1 : dict
            The first dictionary of scores.
        scores2 : dict
            The second dictionary of scores.

        Returns
        -------
        dict
            The merged dictionary of scores.
        """

        #TODO
        final_scores = {}
        for doc_id in scores1:
            final_scores[doc_id] = scores1[doc_id] + scores2[doc_id]
        return final_scores


if __name__ == "__main__":
    search_engine = SearchEngine()
    query = "Spiderman far from dune"
    for method in ['lnc.ltc', 'ltn.lnn', 'OkapiBM25', 'unigram']:
        print("Method:", method)
        weights = {
            Indexes.STARS: 1,
            Indexes.GENRES: 1,
            Indexes.SUMMARIES: 1
        }
        if method == 'unigram':
            for smoothing in ['bayes', 'naive', 'mixture']:
                print("Smoothing:", smoothing)
                result = search_engine.search(query, method, weights, smoothing_method=smoothing)
                with open(project_root+'/data/IMDB_Crawled.json') as file:
                    data = json.load(file)
                for i in result:
                    for movie in data:
                        if movie['id'] == i[0]:
                            print(movie['title'])
                            break
                print('*'*50)
        else:
            result = search_engine.search(query, method, weights, max_results=10, safe_ranking=True)
            
            with open(project_root+'/data/IMDB_Crawled.json') as file:
                data = json.load(file)
            for i in result:
                for movie in data:
                    if movie['id'] == i[0]:
                        print(movie['title'])
                        break
            print('_'*50)

    
# Outputs:
            
# Method: lnc.ltc
# Spider-Man: Beyond the Spider-Verse
# The Secrets of Frank Herbert's Dune
# Dune: The Prophecy
# The Amazing Adventures of Spider-Man
# Watch Out, We're Mad
# How Can I Help You
# Spider-Man: No Way Home
# Atlantis
# Spider-Man 2: Another World
# Spider-Man: Lotus
# __________________________________________________
# Method: ltn.lnn
# The Amazing Spider-Man 2
# Spider-Man: No Way Home
# Spider-Man 2
# Spider-Man: Into the Spider-Verse
# Spider-Man 3
# Watch Out, We're Mad
# Spider-Man: Homecoming
# The Amazing Spider-Man
# Spider-Man
# Dune
# __________________________________________________
# Method: OkapiBM25
# Spider-Man: No Way Home
# The Amazing Spider-Man 2
# Watch Out, We're Mad
# Spider-Man: Into the Spider-Verse
# The Amazing Adventures of Spider-Man
# Spider-Man 2: Another World
# The Secrets of Frank Herbert's Dune
# Dune: The Prophecy
# Spider-Man: Homecoming
# Spider-Man 2
# __________________________________________________
# Method: unigram
# Smoothing: bayes
# The Secrets of Frank Herbert's Dune
# Spider-Man: Beyond the Spider-Verse
# Dune: The Prophecy
# How Can I Help You
# Atlantis
# Watch Out, We're Mad
# Spider-Man: No Way Home
# Children of Dune
# The Amazing Spider-Man 2
# Barefoot Jenny: Undercover
# **************************************************
# Smoothing: naive
# Spider-Man: Beyond the Spider-Verse
# The Secrets of Frank Herbert's Dune
# How Can I Help You
# Atlantis
# Dune: The Prophecy
# Barefoot Jenny: Undercover
# JFK Revisited: Through the Looking Glass
# Untitled Tom Cruise/SpaceX Project
# Sergeant Slaughter, My Big Brother
# Jalsa
# **************************************************
# Smoothing: mixture
# The Secrets of Frank Herbert's Dune
# Dune: The Prophecy
# Watch Out, We're Mad
# Spider-Man: Beyond the Spider-Verse
# Children of Dune
# The Amazing Spider-Man 2
# The Amazing Adventures of Spider-Man
# Spider-Man 2: Another World
# Spider-Man: No Way Home
# Dune
# **************************************************