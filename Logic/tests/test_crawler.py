import json
from typing import List


<<<<<<< HEAD
def type_check(obj, expected_type):
    if not hasattr(expected_type, "__origin__"):
        return isinstance(obj, expected_type)

    assert expected_type.__origin__ == list, "only list type is supported"
    inner_type = expected_type.__args__[0]
    return isinstance(obj, list) and all(type_check(item, inner_type) for item in obj)


=======
>>>>>>> 439171a459e325085de4b00d6b77c99da2a7c9ff
def check_field_types(json_file_path, expected_fields):
    with open(json_file_path, "r") as file:
        data = json.load(file)
    # check len of the data
<<<<<<< HEAD
    assert len(data) >= 1000, f"Expected at least 1000 movies, but got {len(data)}"
=======
    assert len(data) > 500, f"Expected at least 1000 movies, but got {len(data)}"
>>>>>>> 439171a459e325085de4b00d6b77c99da2a7c9ff

    # check data types
    for movie in data:
        for field, expected_type in expected_fields.items():
<<<<<<< HEAD
            if field not in movie or movie[field] is None:
                print(
                    f'Warning: Expected field {field} not found in movie {movie["id"]}'
                )
            else:
                assert type_check(
                    movie[field], expected_type
                ), f'Error: Expected field {field} to be of type {expected_type}, but got {type(movie[field])} in movie {movie["id"]}'
=======
            assert (
                field in movie
            ), f'Expected field {field} not found in movie {movie["id"]}'
            if expected_type is not None:
                assert isinstance(
                    movie[field], expected_type
                ), f'Expected field {field} to be of type {expected_type}, but got {type(movie[field])} in movie {movie["id"]}'
>>>>>>> 439171a459e325085de4b00d6b77c99da2a7c9ff


expected_fields = {
    "id": str,
    "title": str,
    "first_page_summary": str,
    "release_year": str,
    "mpaa": str,
    "budget": str,
    "gross_worldwide": str,
    "rating": str,
    "directors": List[str],
    "writers": List[str],
    "stars": List[str],
    "related_links": List[str],
    "genres": List[str],
    "languages": List[str],
    "countries_of_origin": List[str],
    "summaries": List[str],
    "synopsis": List[str],
    "reviews": List[List[str]],
}

<<<<<<< HEAD
json_file_path = "data/IMDB_Crawled.json"
=======
json_file_path = "../IMDB_crawled.json"
>>>>>>> 439171a459e325085de4b00d6b77c99da2a7c9ff
check_field_types(json_file_path, expected_fields)
