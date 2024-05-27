import os
import sys
current_script_path = os.path.abspath(__file__)
link_analysis = os.path.dirname(current_script_path)
core_dir = os.path.dirname(link_analysis)
Logic_dir = os.path.dirname(core_dir)
project_root = os.path.dirname(Logic_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from Logic.core.link_analysis.graph import LinkGraph
from Logic.core.indexer.indexes_enum import Indexes
from Logic.core.indexer.index_reader import Index_reader
import json
from tqdm import tqdm

class LinkAnalyzer:
    def __init__(self, root_set):
        """
        Initialize the Link Analyzer attributes:

        Parameters
        ----------
        root_set: list
            A list of movie dictionaries with the following keys:
            "id": A unique ID for the movie
            "title": string of movie title
            "stars": A list of movie star names
        """
        self.root_set = root_set
        self.graph = LinkGraph()
        self.hubs = []
        self.authorities = []
        self.initiate_params()

    def initiate_params(self):
        """
        Initialize links graph, hubs list and authorities list based of root set

        Parameters
        ----------
        This function has no parameters. You can use self to get or change attributes
        """
        self.movie_id2name = {}
        self.stars = set()
        for movie in tqdm(self.root_set, desc='Initiating Graph'):
            movie_id = movie['id']
            self.graph.add_node(movie_id)
            self.movie_id2name[movie_id] = movie['title']
            
            if not movie['stars']:
                continue
            for star in movie['stars']:
                star_name = star
                self.graph.add_node(star_name)
                self.graph.add_edge(star_name, movie_id)
                self.stars.add(star_name)

        self.hubs = list(self.graph.graph.nodes())
        self.authorities = list(self.graph.graph.nodes())
            

    def expand_graph(self, corpus):
        """
        expand hubs, authorities and graph using given corpus

        Parameters
        ----------
        corpus: list
            A list of movie dictionaries with the following keys:
            "id": A unique ID for the movie
            "stars": A list of movie star names

        Note
        ---------
        To build the base set, we need to add the hubs and authorities that are inside the corpus
        and refer to the nodes in the root set to the graph and to the list of hubs and authorities.
        """
        for movie in tqdm(corpus, desc='Expanding Graph'):
            #TODO
            movie_id = movie['id']
            self.movie_id2name[movie_id] = movie['title']
            if movie['stars'] == None:
                continue
            for star in movie['stars']:
                if star in self.hubs:
                    self.graph.add_node(star)
                    self.graph.add_edge(star, movie_id)


    def hits(self, num_iteration=5, max_result=10):
        """
        Return the top movies and actors using the Hits algorithm

        Parameters
        ----------
        num_iteration: int
            Number of algorithm execution iterations
        max_result: int
            The maximum number of results to return. If None, all results are returned.

        Returns
        -------
        list
            List of names of 10 actors with the most scores obtained by Hits algorithm in descending order
        list
            List of names of 10 movies with the most scores obtained by Hits algorithm in descending order
        """
        # a_s = []
        # h_s = []

        #TODO
        hub_scores = {node: 1.0 for node in self.graph.graph.nodes()}  # Start with equal scores
        auth_scores = {node: 1.0 for node in self.graph.graph.nodes()}
        for _ in range(num_iteration):
            # Update authority scores
            for node in self.graph.graph.nodes():
                auth_scores[node] = sum(hub_scores[predecessor] for predecessor in self.graph.get_predecessors(node))

            # Update hub scores
            for node in self.graph.graph.nodes():
                hub_scores[node] = sum(auth_scores[successor] for successor in self.graph.get_successors(node))

            # Normalize authority scores
            norm = sum(auth_scores.values())
            for node in auth_scores:
                auth_scores[node] /= norm

            # Normalize hub scores
            norm = sum(hub_scores.values())
            for node in hub_scores:
               hub_scores[node] /= norm

        sorted_auths = sorted(auth_scores.items(), key=lambda x: x[1], reverse=True)[:max_result]
        sorted_hubs = sorted(hub_scores.items(), key=lambda x: x[1], reverse=True)[:max_result]
        top_auths = [node for node, _ in sorted_auths]
        top_hubs = [node for node, _ in sorted_hubs]

        return top_auths, top_hubs

if __name__ == "__main__":
    # You can use this section to run and test the results of your link analyzer
    with open(project_root + '/data/IMDB_Crawled.json', 'r') as file:
        data = json.load(file)
    corpus = data    # TODO: it shoud be your crawled data
    root_set = [movie for movie in data if "Avengers" in movie['title']]   # TODO: it shoud be a subset of your corpus

    analyzer = LinkAnalyzer(root_set=root_set)
    analyzer.expand_graph(corpus=corpus)
    movies, actors = analyzer.hits(max_result=10, num_iteration=100)
    movies = [analyzer.movie_id2name[movie] for movie in movies]
    actors = [actor for actor in actors if actor in analyzer.stars]
    print("Top Actors:")
    print(*actors, sep=' - ')
    print("Top Movies:")
    print(*movies, sep=' - ')

# Outputs:
# Top Actors:
# Robert Downey Jr. - Chris Evans - Mark Ruffalo - Scarlett Johansson - Chris Hemsworth - Ralph Fiennes - Sean Connery - Uma Thurman - Patrick Macnee - Diana Rigg
# Top Movies:
# Avengers: Endgame - Avengers: Age of Ultron - The Avengers - Captain America: Civil War - Avengers: Infinity War - Zodiac - Captain America: The Winter Soldier - Iron Man - Spider-Man: Homecoming - Iron Man 3