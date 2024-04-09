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
    query = "spider man in wonderland"
    method = "lnc.ltc"
    weights = {
        Indexes.STARS: 1,
        Indexes.GENRES: 1,
        Indexes.SUMMARIES: 1
    }
    result = search_engine.search(query, method, weights, max_results=10, safe_ranking=True)

    print(result)

    with open(project_root+'/data/IMDB_Crawled.json') as file:
        data = json.load(file)
    for i in result:
        for movie in data:
            if movie['id'] == i[0]:
                print(movie['title'])
                break

    
    # Outputs:
            
    # [('tt0043274', 0.06577479354537515), ('tt1601792', 0.0573517010753171), ('tt12453114', 0.05460346636164671), ('tt9362722', 0.05200234869459513), ('tt0145487', 0.04831880083004559), ('tt1414867', 0.04532154476740904), ('tt24485052', 0.04472669500975058), ('tt10270200', 0.0442610718877456), ('tt0316654', 0.04139961209621638), ('tt11847842', 0.04107104913022986)]
    # Alice in Wonderland
    # Aa Naluguru
    # Groundhog Day for a Black Man
    # Spider-Man: Across the Spider-Verse
    # Spider-Man
    # Why Did You Come to My House?
    # Sirf Ek Bandaa Kaafi Hai
    # The Vanishing Triangle
    # Spider-Man 2
    # The Tourist
