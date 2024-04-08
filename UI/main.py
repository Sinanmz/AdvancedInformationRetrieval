import os
import sys
current_script_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_script_path)
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from Logic import utils
from Logic.core.snippet import Snippet
from Logic.core.preprocess import Preprocessor

import streamlit as st
import time
from enum import Enum
import random




snippet_obj = Snippet(
    number_of_words_on_each_side=3
)  # You can change this parameter, if needed.


class color(Enum):
    RED = "#FF0000"
    GREEN = "#00FF00"
    BLUE = "#0000FF"
    YELLOW = "#FFFF00"
    WHITE = "#FFFFFF"
    CYAN = "#00FFFF"
    MAGENTA = "#FF00FF"


def get_summary_with_snippet(movie_info, query):
    summary = movie_info["first_page_summary"]
    summary = ' '.join(summary.split())
    snippet, not_exist_words = snippet_obj.find_snippet(summary, query)
    print(snippet)
    if "***" in snippet:
        # snippet = snippet.split()
        # for i in range(len(snippet)):
            # current_word = snippet[i]
            # if current_word.startswith("***") and current_word.endswith("***"):
            #     current_word_without_star = current_word[3:-3]
            #     summary = summary.replace(
            #         current_word_without_star,
            #         f"<b><font size='4' color={random.choice(list(color)).value}>{current_word_without_star}</font></b>",
            #     )
        if '...' in snippet:
            snips = snippet.split('...')
        else:
            snips = [snippet]
        print(snips)
        for snip in snips:
            idx1 = snip.find('***')
            idx2 = snip[idx1+1:].find('***')
            idx2 = idx2+idx1+1
            print('-------------------------------')
            print(idx1)
            print(idx2)
            print(snip[:idx1])
            print(snip[idx1+3:idx2])
            print(snip[idx2+3:])
            print(snip)
            print(snip[:idx1].rstrip() + ' ' + f"<b><font size='4' color={random.choice(list(color)).value}>{snip[idx1+3:idx2].strip()}</font></b>" + ' '+ snip[idx2+3:].lstrip())
            print('-------------------------------')
            summary = summary.replace(
                snip,
                snip[:idx1].rstrip() + ' ' + f"<b><font size='4' color={random.choice(list(color)).value}>{snip[idx1+3:idx2].strip()}</font></b>" + ' '+ snip[idx2+3:].lstrip(),
                )
    print(summary)
    return summary


def search_time(start, end):
    st.success("Search took: {:.6f} milli-seconds".format((end - start) * 1e3))


def search_handling(
    search_button,
    search_term,
    search_max_num,
    search_weights,
    search_method,
):
    preprocessor = Preprocessor([])
    all_documents = []
    for movie in utils.movies_dataset.values():
        all_documents.append(movie["title"])
        # all_documents.extend(movie["stars"])
        for star in movie["stars"]:
            all_documents.append(preprocessor.normalize(star))
        all_documents.extend(movie["genres"])
        # all_documents.extend(movie["directors"])
        for director in movie["directors"]:
            all_documents.append(preprocessor.normalize(director))
        all_documents.extend(movie["summaries"])
        # all_documents.extend(movie["writers"])
        for writer in movie["writers"]:
            all_documents.append(preprocessor.normalize(writer))
        all_documents.extend(movie["synopsis"])
        for review in movie["reviews"]:
            all_documents.append(review[0])
    
    if search_button:
        corrected_query = utils.correct_text(search_term, all_documents)
        
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
            )
            print(f"Result: {result}")
            end_time = time.time()
            if len(result) == 0:
                st.warning("No results found!")
                return

            search_time(start_time, end_time)

            for i in range(len(result)):
                card = st.columns([3, 1])
                info = utils.get_movie_by_id(result[i][0], utils.movies_dataset)
                fps = info['first_page_summary']
                if len(fps) > 30 and fps[1:].rfind(fps[:30]) != -1:
                    info['first_page_summary'] = fps[fps[1:].rfind(fps[:20])+1:]
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
    # search_summary_terms = st.text_input("Search in summary of movie")
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

        search_weights = [weight_stars, weight_genres, weight_summary]
        search_method = st.selectbox(
            "Search method",
            ("OkapiBM25", "ltn.lnn", "ltc.lnc"),
        )

    search_button = st.button("Search!")

    search_handling(
        search_button,
        search_term,
        search_max_num,
        search_weights,
        search_method,
    )


if __name__ == "__main__":
    main()
