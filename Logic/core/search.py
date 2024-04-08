import os
import sys
current_script_path = os.path.abspath(__file__)
core_dir = os.path.dirname(current_script_path)
Logic_dir = os.path.dirname(core_dir)
project_root = os.path.dirname(Logic_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from Logic.core.preprocess import Preprocessor
from Logic.core.scorer import Scorer
from Logic.core.indexer.indexes_enum import Indexes, Index_types
from Logic.core.indexer.index_reader import Index_reader

import json
import numpy as np


class SearchEngine:
    def __init__(self):
        """
        Initializes the search engine.

        """
        path = '/Users/sina/Sem-5/MIR/Project/MIR/index/'
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


    def search(self, query, method, weights, safe_ranking = True, max_results=10):
        """
        searches for the query in the indexes.

        Parameters
        ----------
        query : str
            The query to search for.
        method : str ((n|l)(n|t)(n|c).(n|l)(n|t)(n|c)) | OkapiBM25
            The method to use for searching.
        weights: dict
            The weights of the fields.
        safe_ranking : bool
            If True, the search engine will search in whole index and then rank the results. 
            If False, the search engine will search in tiered index.
        max_results : int
            The maximum number of results to return. If None, all results are returned.

        Returns
        -------
        list
            A list of tuples containing the document IDs and their scores sorted by their scores.
        """

        preprocessor = Preprocessor([query])
        query = preprocessor.preprocess()[0].split()

        scores = {}
        if safe_ranking:
            self.find_scores_with_safe_ranking(query, method, weights, scores)
        else:
            self.find_scores_with_unsafe_ranking(query, method, weights, max_results, scores)

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

    def find_scores_with_unsafe_ranking(self, query, method, weights, max_results, scores):
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
        idf = {}
        for term in query:
            if term not in idf:
                df = len(self.document_indexes.get(term, {}))
                idf[term] = np.log((number_of_documents)/(df+1))

        
        for tier in ["first_tier", "second_tier", "third_tier"]:
            for field in weights:
                #TODO
                # pass
                scorer = Scorer(self.tiered_index[field][tier], number_of_documents)
                scorer.idf = idf
                if method == 'OkapiBM25':
                    average_document_field_length = self.metadata_index['averge_document_length'][field.value]
                    document_lengths = self.document_lengths_index[field]

                    temp_scores = scorer.compute_socres_with_okapi_bm25(query, average_document_field_length, document_lengths)
                else:
                    temp_scores = scorer.compute_scores_with_vector_space_model(query, method)
                
                if field not in scores:
                    scores[field] = temp_scores
                else:
                    for doc_id in temp_scores:
                        scores[field][doc_id] = temp_scores[doc_id]

            retrieved = set()
            for field in weights:
                retrieved = retrieved.union(set(scores[field].keys()))
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
            if method == 'OkapiBM25':
                average_document_field_length = self.metadata_index['averge_document_length'][field.value]
                document_lengths = self.document_lengths_index[field]
                scores[field] = scorer.compute_socres_with_okapi_bm25(query, average_document_field_length, document_lengths)
            else:
                scores[field] = scorer.compute_scores_with_vector_space_model(query, method)
        
        


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


if __name__ == '__main__':
    search_engine = SearchEngine()
    query = "Christoph waltz"
    method = 'OkapiBM25'
    weights = {
        Indexes.STARS: 2,
        Indexes.GENRES: 1,
        Indexes.SUMMARIES: 1
    }
    result = search_engine.search(query, method, weights, safe_ranking=True, max_results=10)

    print(result)

    with open('/Users/sina/Sem-5/MIR/Project/MIR/data/IMDB_Crawled.json') as file:
        data = json.load(file)
    for i in result:
        for movie in data:
            if movie['id'] == i[0]:
                print(movie['title'])
                break
