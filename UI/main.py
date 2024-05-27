import os
import sys
current_script_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_script_path)
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from Logic import utils
from Logic.core.utility.snippet import Snippet

import streamlit as st
import time
from enum import Enum
import random
from Logic.core.utility.snippet import Snippet
from Logic.core.link_analysis.analyzer import LinkAnalyzer
from Logic.core.indexer.index_reader import Index_reader, Indexes


import re
import json

snippet_obj = Snippet()


class color(Enum):
    RED = "#FF0000"
    GREEN = "#00FF00"
    BLUE = "#0000FF"
    YELLOW = "#FFFF00"
    WHITE = "#FFFFFF"
    CYAN = "#00FFFF"
    MAGENTA = "#FF00FF"


def get_top_x_movies_by_rank(x: int, results: list):
    # path = project_root+'/index/' # Link to the index folder
    # document_index = Index_reader(path, Indexes.DOCUMENTS)
    # corpus = []
    # root_set = []
    # for movie_id, movie_detail in document_index.index.items():
    #     movie_title = movie_detail["title"]
    #     stars = movie_detail["stars"]
    #     corpus.append({"id": movie_id, "title": movie_title, "stars": stars})

    # for element in results:
    #     movie_id = element[0]
    #     movie_detail = document_index.index[movie_id]
    #     movie_title = movie_detail["title"]
    #     stars = movie_detail["stars"]
    #     root_set.append({"id": movie_id, "title": movie_title, "stars": stars})

    crawled_path = project_root + "/data/IMDB_Crawled.json"
    with open(crawled_path, "r") as f:
        data = json.load(f)
    root_set = []
    corpus = []
    root_ids = [element[0] for element in results]
    for movie in data:
        if movie['id'] and movie['title']:
            id = movie['id']
            title = movie['title']
            stars = movie['stars'] if movie['stars'] else []
            corpus.append({"id": id, "title": title, "stars": stars})
            if id in root_ids:
                root_set.append({"id": id, "title": title, "stars": stars})

    analyzer = LinkAnalyzer(root_set=root_set)
    analyzer.expand_graph(corpus=corpus)
    movies, actors = analyzer.hits(max_result=x)
    return actors, movies


def get_summary_with_snippet(movie_info, query):
    summary = movie_info["first_page_summary"]
    summary = summary.split()
    summary = ' '.join(summary)
    snippet, not_exist_words = snippet_obj.find_snippet(summary, query)
    # if "***" in snippet:
    #     snippet = snippet.split()
    #     for i in range(len(snippet)):
    #         current_word = snippet[i]
    #         if current_word.startswith("***") and current_word.endswith("***"):
    #             current_word_without_star = current_word[3:-3]
    #             summary = summary.replace(
    #                 current_word_without_star,
    #                 f"<b><font size='4' color={random.choice(list(color)).value}>{current_word_without_star}</font></b>",
    #             )

    if ' ... ' in snippet:
        snips = snippet.split(' ... ')
    else:
        snips = [snippet]
    
    for snip in snips:
        indexes = []
        for i in range(len(snip.split())):
            if snip.split()[i].startswith("***") and snip.split()[i].endswith("***"):
                indexes.append(i)
        
        summary_before_change = ''
        summary_after_change = ''
        for i in range(len(snip.split())):
            if i in indexes:
                summary_before_change += snip.split()[i][3:-3] + ' '
        
                match_end = re.search(r"([\W_]+)$", snip.split()[i][3:-3])
                if match_end != None:
                    match_end = match_end.group(1)
                else:
                    match_end = ''

                match_start = re.search(r"^([\W_]+)", snip.split()[i][3:-3])
                if match_start != None:
                    match_start = match_start.group(1)
                else:
                    match_start = ''

                word = snip.split()[i][3:-3]
                word = word[:len(word)-len(match_end)]
                word = word[len(match_start):]

                summary_after_change += f"{match_start}<b><font size='4' color={random.choice(list(color)).value}>{word}</font></b>{match_end} "

            else:
                summary_before_change += snip.split()[i] + ' '
                summary_after_change += snip.split()[i] + ' '
        
        summary_before_change = summary_before_change.strip()
        summary_after_change = summary_after_change.strip()
        summary = summary.replace(summary_before_change, summary_after_change)
        
    return summary


def search_time(start, end):
    st.success("Search took: {:.6f} milli-seconds".format((end - start) * 1e3))


def search_handling(
    search_button,
    search_term,
    search_max_num,
    search_weights,
    search_method,
    unigram_smoothing,
    alpha,
    lamda,
    filter_button,
    num_filter_results,
):
    if filter_button:
        if "search_results" in st.session_state:
            top_actors, top_movies = get_top_x_movies_by_rank(
                num_filter_results, st.session_state["search_results"]
            )
            print(top_actors)
            st.markdown(f"**Top {num_filter_results} Actors:**")
            actors_ = ", ".join(top_actors)
            st.markdown(
                f"<span style='color:{random.choice(list(color)).value}'>{actors_}</span>",
                unsafe_allow_html=True,
            )
            st.divider()

        st.markdown(f"**Top {num_filter_results} Movies:**")
        for i in range(len(top_movies)):
            card = st.columns([3, 1])
            info = utils.get_movie_by_id(top_movies[i], utils.movies_dataset)
            with card[0].container():
                st.title(info["title"])
                st.markdown(f"[Link to movie]({info['URL']})")
                st.markdown(
                    f"<b><font size = '4'>Summary:</font></b> {get_summary_with_snippet(info, search_term)}",
                    unsafe_allow_html=True,
                )

            with st.container():
                st.markdown("**Directors:**")
                num_authors = len(info["directors"])
                for j in range(num_authors):
                    st.text(info["directors"][j])

            with st.container():
                st.markdown("**Stars:**")
                num_authors = len(info["stars"])
                stars = "".join(star + ", " for star in info["stars"])
                st.text(stars[:-2])

                topic_card = st.columns(1)
                with topic_card[0].container():
                    st.write("Genres:")
                    num_topics = len(info["genres"])
                    for j in range(num_topics):
                        st.markdown(
                            f"<span style='color:{random.choice(list(color)).value}'>{info['genres'][j]}</span>",
                            unsafe_allow_html=True,
                        )
            with card[1].container():
                st.image(info["Image_URL"], use_column_width=True)

            st.divider()
        return

    if search_button:
        corrected_query = utils.correct_text(search_term, utils.all_documents)

        if corrected_query != search_term:
            st.warning(f"Your search terms were corrected to: {corrected_query}")
            search_term = corrected_query

        with st.spinner("Searching..."):
            time.sleep(0.5)  # for showing the spinner! (can be removed)
            start_time = time.time()
            result = utils.search(
                search_term,
                search_max_num,
                search_method,
                search_weights,
                unigram_smoothing=unigram_smoothing,
                alpha=alpha,
                lamda=lamda,
            )
            if "search_results" in st.session_state:
                st.session_state["search_results"] = result
            print(f"Result: {result}")
            end_time = time.time()
            if len(result) == 0:
                st.warning("No results found!")
                return

            search_time(start_time, end_time)

        for i in range(len(result)):
            card = st.columns([3, 1])
            info = utils.get_movie_by_id(result[i][0], utils.movies_dataset)
            with card[0].container():
                st.title(info["title"])
                st.markdown(f"[Link to movie]({info['URL']})")
                st.write(f"Relevance Score: {result[i][1]}")
                st.markdown(
                    f"<b><font size = '4'>Summary:</font></b> {get_summary_with_snippet(info, search_term)}",
                    unsafe_allow_html=True,
                )

            with st.container():
                st.markdown("**Directors:**")
                num_authors = len(info["directors"])
                for j in range(num_authors):
                    st.text(info["directors"][j])

            with st.container():
                st.markdown("**Stars:**")
                num_authors = len(info["stars"])
                stars = "".join(star + ", " for star in info["stars"])
                st.text(stars[:-2])

                topic_card = st.columns(1)
                with topic_card[0].container():
                    st.write("Genres:")
                    num_topics = len(info["genres"])
                    for j in range(num_topics):
                        st.markdown(
                            f"<span style='color:{random.choice(list(color)).value}'>{info['genres'][j]}</span>",
                            unsafe_allow_html=True,
                        )
            with card[1].container():
                st.image(info["Image_URL"], use_column_width=True)

            st.divider()

        st.session_state["search_results"] = result
        if "filter_state" in st.session_state:
            st.session_state["filter_state"] = (
                "search_results" in st.session_state
                and len(st.session_state["search_results"]) > 0
            )


def main():
    st.title("Search Engine")
    st.write(
        "This is a simple search engine for IMDB movies. You can search through IMDB dataset and find the most relevant movie to your search terms."
    )
    st.markdown(
        '<span style="color:yellow">Developed By: MIR Team at Sharif University</span>',
        unsafe_allow_html=True,
    )

    search_term = st.text_input("Seacrh Term")
    with st.expander("Advanced Search"):
        search_max_num = st.number_input(
            "Maximum number of results", min_value=5, max_value=100, value=10, step=5
        )
        weight_stars = st.slider(
            "Weight of stars in search",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.1,
        )

        weight_genres = st.slider(
            "Weight of genres in search",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.1,
        )

        weight_summary = st.slider(
            "Weight of summary in search",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.1,
        )
        slider_ = st.slider("Select the number of top movies to show", 1, 10, 5)

        search_weights = [weight_stars, weight_genres, weight_summary]
        search_method = st.selectbox(
            "Search method", ("OkapiBM25", "ltn.lnn", "ltc.lnc", "unigram")
        )

        unigram_smoothing = None
        alpha, lamda = None, None
        if search_method == "unigram":
            unigram_smoothing = st.selectbox(
                "Smoothing method",
                ("naive", "bayes", "mixture"),
            )
            if unigram_smoothing == "bayes":
                alpha = st.slider(
                    "Alpha",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.1,
                )
            if unigram_smoothing == "mixture":
                alpha = st.slider(
                    "Alpha",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.1,
                )
                lamda = st.slider(
                    "Lambda",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.1,
                )

    if "search_results" not in st.session_state:
        st.session_state["search_results"] = []

    search_button = st.button("Search!")
    filter_button = st.button("Filter movies by ranking")

    search_handling(
        search_button,
        search_term,
        search_max_num,
        search_weights,
        search_method,
        unigram_smoothing,
        alpha,
        lamda,
        filter_button,
        slider_,
    )


if __name__ == "__main__":
    main()