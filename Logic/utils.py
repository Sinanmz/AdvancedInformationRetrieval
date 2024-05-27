import os
import sys
current_script_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_script_path)
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from Logic.core.search import SearchEngine
from Logic.core.utility.spell_correction import SpellCorrection
from Logic.core.utility.snippet import Snippet
from Logic.core.indexer.indexes_enum import Indexes, Index_types
from Logic.core.utility.preprocess import Preprocessor


from typing import Dict, List
import json


data_path = project_root + '/data/IMDB_Crawled.json'
with open(data_path, 'r') as f:
    data = json.load(f)

movies_dataset = {}
for movie in data:
    temp = {'title': movie['title'] if movie['title'] else '', 
            'first_page_summary': movie['first_page_summary'] if movie['first_page_summary'] else '', 
            'URL': f"https://www.imdb.com/title/{movie['id']}", 
            'stars': movie['stars'] if movie['stars'] else [], 
            'genres': movie['genres'] if movie['genres'] else [], 
            'id': movie['id'] if movie['id'] else '',
            'directors': movie['directors'] if movie['directors'] else [], 
            'summaries': movie['summaries'] if movie['summaries'] else [],
            'writers': movie['writers'] if movie['writers'] else [], 
            'reviews': movie['reviews'] if movie['reviews'] else [], 
            'synopsis': movie['synposis'] if movie['synposis'] else [], 
            'Image_URL': 'https://m.media-amazon.com/images/M/MV5BNDE3ODcxYzMtY2YzZC00NmNlLWJiNDMtZDViZWM2MzIxZDYwXkEyXkFqcGdeQXVyNjAwNDUxODI@._V1_.jpg',
            }
    movies_dataset[movie['id']] = temp
search_engine = SearchEngine()


preprocessor = Preprocessor([])
all_documents = []
for movie in movies_dataset.values():
    if movie["title"]:
        all_documents.append(movie["title"])

    # all_documents.extend(movie["stars"])
    if movie["stars"]:
        for star in movie["stars"]:
            if star:
                all_documents.append(preprocessor.normalize(star))
    if movie["genres"]:
        all_documents.extend(movie["genres"])

    # all_documents.extend(movie["directors"])
    # if movie["directors"]:
    #     for director in movie["directors"]:
    #         if director:
    #             all_documents.append(preprocessor.normalize(director))
    
    if movie["summaries"]:
        all_documents.extend(movie["summaries"][:5])

    # all_documents.extend(movie["writers"])
    # if movie["writers"]:
    #     for writer in movie["writers"]:
    #         if writer:
    #             all_documents.append(preprocessor.normalize(writer))

    # if movie["synopsis"]:
    #     all_documents.extend(movie["synopsis"][:5])

    # if movie["reviews"]:
    #     for review in movie["reviews"][:5]:
    #         if review:
    #             all_documents.append(review[0])


def correct_text(text: str, all_documents: List[str]) -> str:
    """
    Correct the give query text, if it is misspelled using Jacard similarity

    Paramters
    ---------
    text: str
        The query text
    all_documents : list of str
        The input documents.

    Returns
    str
        The corrected form of the given text
    """
    # TODO: You can add any preprocessing steps here, if needed!
    spell_correction_obj = SpellCorrection(all_documents)
    text = spell_correction_obj.spell_check(text)
    return text


def search(
    query: str,
    max_result_count: int,
    method: str = "ltn.lnn",
    weights: list = [0.3, 0.3, 0.4],
    should_print=False,
    preferred_genre: str = None,
    unigram_smoothing = 'mixture', 
    alpha = .5, 
    lamda = .5

):
    """
    Finds relevant documents to query

    Parameters
    ---------------------------------------------------------------------------------------------------
    query:
        The query text

    max_result_count: Return top 'max_result_count' docs which have the highest scores.
                      notice that if max_result_count = -1, then you have to return all docs

    method: 'ltn.lnn' or 'ltc.lnc' or 'OkapiBM25'

    weights:
        The list, containing importance weights in the search result for each of these items:
            Indexes.STARS: weights[0],
            Indexes.GENRES: weights[1],
            Indexes.SUMMARIES: weights[2],

    preferred_genre:
        A list containing preference rates for each genre. If None, the preference rates are equal.
        (You can leave it None for now)

    Returns
    ----------------------------------------------------------------------------------------------------
    list
    Retrieved documents with snippet
    """
    # TODO
    weights = {
        Indexes.STARS: weights[0],
        Indexes.GENRES: weights[1],
        Indexes.SUMMARIES: weights[2]
    }
    if max_result_count == -1:
        max_result_count = None
    
    if method in ['ltn.lnn', 'ltc.lnc', 'OkapiBM25']:
        return search_engine.search(
            query, method, weights, max_results=max_result_count, safe_ranking=True
        )
    else:
        return search_engine.search(
            query, method, weights, max_results=max_result_count, smoothing_method=unigram_smoothing, alpha=alpha, lamda=lamda
        )
    


def get_movie_by_id(id: str, movies_dataset: List[Dict[str, str]]) -> Dict[str, str]:
    """
    Get movie by its id

    Parameters
    ---------------------------------------------------------------------------------------------------
    id: str
        The id of the movie

    movies_dataset: List[Dict[str, str]]
        The dataset of movies

    Returns
    ----------------------------------------------------------------------------------------------------
    dict
        The movie with the given id
    """
    # result = movies_dataset.get(
    #     id,
    #     {
    #         "Title": "This is movie's title",
    #         "Summary": "This is a summary",
    #         "URL": "https://www.imdb.com/title/tt0111161/",
    #         "Cast": ["Morgan Freeman", "Tim Robbins"],
    #         "Genres": ["Drama", "Crime"],
    #         "Image_URL": "https://m.media-amazon.com/images/M/MV5BNDE3ODcxYzMtY2YzZC00NmNlLWJiNDMtZDViZWM2MzIxZDYwXkEyXkFqcGdeQXVyNjAwNDUxODI@._V1_.jpg",
    #     },
    # )

    # result["Image_URL"] = (
    #     "https://m.media-amazon.com/images/M/MV5BNDE3ODcxYzMtY2YzZC00NmNlLWJiNDMtZDViZWM2MzIxZDYwXkEyXkFqcGdeQXVyNjAwNDUxODI@._V1_.jpg"  # a default picture for selected movies
    # )
    # result["URL"] = (
    #     f"https://www.imdb.com/title/{result['id']}"  # The url pattern of IMDb movies
    # )
    # return result
    movie = movies_dataset.get(id)
    return movie
