import os
import sys
current_script_path = os.path.abspath(__file__)
clustering_dir = os.path.dirname(current_script_path)
core_dir = os.path.dirname(clustering_dir)
Logic_dir = os.path.dirname(core_dir)
project_root = os.path.dirname(Logic_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

import matplotlib.pyplot as plt
import numpy as np
import random
import operator
import wandb

from typing import List, Tuple
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from collections import Counter
from Logic.core.clustering.clustering_metrics import *

from sklearn.manifold import TSNE


class ClusteringUtils:

    def __init__(self):
        pass

    def cluster_kmeans(self, emb_vecs: List, n_clusters: int, max_iter: int = 500) -> Tuple[List, List]:
        """
        Clusters input vectors using the K-means method.

        Parameters
        -----------
        emb_vecs: List
            A list of vectors to be clustered.
        n_clusters: int
            The number of clusters to form.

        Returns
        --------
        Tuple[List, List]
            Two lists:
            1. A list containing the cluster centers.
            2. A list containing the cluster index for each input vector.
        """
        cluster_centers = random.sample(list(emb_vecs), n_clusters)
        cluster_indices = [-1] * len(emb_vecs)

        for _ in range(max_iter):
            for i, vec in enumerate(emb_vecs):
                min_dist = float('inf')
                for j, center in enumerate(cluster_centers):
                    dist = np.linalg.norm(np.array(vec) - np.array(center))
                    if dist < min_dist:
                        min_dist = dist
                        cluster_indices[i] = j

            new_cluster_centers = []
            for j in range(n_clusters):
                cluster_points = [emb_vecs[i] for i in range(len(emb_vecs)) if cluster_indices[i] == j]
                if len(cluster_points) > 0:
                    new_center = np.mean(cluster_points, axis=0)
                    new_cluster_centers.append(new_center)
                else:
                    new_center = random.choice(emb_vecs)
                    new_cluster_centers.append(new_center)

            if np.allclose(cluster_centers, new_cluster_centers):
                break
            else:
                cluster_centers = new_cluster_centers

        return cluster_centers, cluster_indices


    def get_most_frequent_words(self, documents: List[str], top_n: int = 10) -> List[Tuple[str, int]]:
        """
        Finds the most frequent words in a list of documents.

        Parameters
        -----------
        documents: List[str]
            A list of documents, where each document is a string representing a list of words.
        top_n: int, optional
            The number of most frequent words to return. Default is 10.

        Returns
        --------
        List[Tuple[str, int]]
            A list of tuples, where each tuple contains a word and its frequency, sorted in descending order of frequency.
        """
        words = ' '.join(documents).split()
        word_freq = Counter(words)
        most_common = word_freq.most_common(top_n)
        return most_common

    def cluster_kmeans_WCSS(self, emb_vecs: List, n_clusters: int) -> Tuple[List, List, float]:
        """ This function performs K-means clustering on a list of input vectors and calculates the Within-Cluster Sum of Squares (WCSS) for the resulting clusters.

        This function implements the K-means algorithm and returns the cluster centroids, cluster assignments for each input vector, and the WCSS value.

        The WCSS is a measure of the compactness of the clustering, and it is calculated as the sum of squared distances between each data point and its assigned cluster centroid. A lower WCSS value indicates that the data points are closer to their respective cluster centroids, suggesting a more compact and well-defined clustering.

        The K-means algorithm works by iteratively updating the cluster centroids and reassigning data points to the closest centroid until convergence or a maximum number of iterations is reached. This function uses a random initialization of the centroids and runs the algorithm for a maximum of 100 iterations.

        Parameters
        -----------
        emb_vecs: List
            A list of vectors to be clustered.
        n_clusters: int
            The number of clusters to form.

        Returns
        --------
        Tuple[List, List, float]
            Three elements:
            1) A list containing the cluster centers.
            2) A list containing the cluster index for each input vector.
            3) The Within-Cluster Sum of Squares (WCSS) value for the clustering.
        """
        cluster_centers, cluster_indices = self.cluster_kmeans(emb_vecs, n_clusters)
        wcss = 0
        for i, vec in enumerate(emb_vecs):
            wcss += np.linalg.norm(np.array(vec) - np.array(cluster_centers[cluster_indices[i]])) ** 2

        return cluster_centers, cluster_indices, wcss

    def cluster_hierarchical_single(self, emb_vecs: List, n_clusters=None, distance_threshold=None) -> List:
        """
        Clusters input vectors using the hierarchical clustering method with single linkage.

        Parameters
        -----------
        emb_vecs: List
            A list of vectors to be clustered.

        Returns
        --------
        List
            A list containing the cluster index for each input vector.
        """
        if distance_threshold is None and n_clusters is None:
            n_clusters = 10
        hierarchical = AgglomerativeClustering(linkage='single', n_clusters=n_clusters, distance_threshold=distance_threshold)
        cluster_indices = hierarchical.fit_predict(emb_vecs)
        return cluster_indices

    def cluster_hierarchical_complete(self, emb_vecs: List, n_clusters=None, distance_threshold=None) -> List:
        """
        Clusters input vectors using the hierarchical clustering method with complete linkage.

        Parameters
        -----------
        emb_vecs: List
            A list of vectors to be clustered.

        Returns
        --------
        List
            A list containing the cluster index for each input vector.
        """
        if distance_threshold is None and n_clusters is None:
            n_clusters = 10
        hierarchical = AgglomerativeClustering(linkage='complete', n_clusters=n_clusters, distance_threshold=distance_threshold)
        cluster_indices = hierarchical.fit_predict(emb_vecs)
        return cluster_indices

    def cluster_hierarchical_average(self, emb_vecs: List, n_clusters=None, distance_threshold=None) -> List:
        """
        Clusters input vectors using the hierarchical clustering method with average linkage.

        Parameters
        -----------
        emb_vecs: List
            A list of vectors to be clustered.

        Returns
        --------
        List
            A list containing the cluster index for each input vector.
        """
        if distance_threshold is None and n_clusters is None:
            n_clusters = 10
        hierarchical = AgglomerativeClustering(linkage='average', n_clusters=n_clusters, distance_threshold=distance_threshold)
        cluster_indices = hierarchical.fit_predict(emb_vecs)
        return cluster_indices

    def cluster_hierarchical_ward(self, emb_vecs: List, n_clusters=None, distance_threshold=None) -> List:
        """
        Clusters input vectors using the hierarchical clustering method with Ward's method.

        Parameters
        -----------
        emb_vecs: List
            A list of vectors to be clustered.

        Returns
        --------
        List
            A list containing the cluster index for each input vector.
        """
        if distance_threshold is None and n_clusters is None:
            n_clusters = 10
        hierarchical = AgglomerativeClustering(linkage='ward', n_clusters=n_clusters, distance_threshold=distance_threshold)
        cluster_indices = hierarchical.fit_predict(emb_vecs)
        return cluster_indices

    def visualize_kmeans_clustering_wandb(self, data, n_clusters, project_name, run_name):
        """ This function performs K-means clustering on the input data and visualizes the resulting clusters by logging a scatter plot to Weights & Biases (wandb).

        This function applies the K-means algorithm to the input data and generates a scatter plot where each data point is colored according to its assigned cluster.
        For visualization use convert_to_2d_tsne to make your scatter plot 2d and visualizable.
        The function performs the following steps:
        1. Initialize a new wandb run with the provided project and run names.
        2. Perform K-means clustering on the input data with the specified number of clusters.
        3. Obtain the cluster labels for each data point from the K-means model.
        4. Create a scatter plot of the data, coloring each point according to its cluster label.
        5. Log the scatter plot as an image to the wandb run, allowing visualization of the clustering results.
        6. Close the plot display window to conserve system resources (optional).

        Parameters
        -----------
        data: np.ndarray
            The input data to perform K-means clustering on.
        n_clusters: int
            The number of clusters to form during the K-means clustering process.
        project_name: str
            The name of the wandb project to log the clustering visualization.
        run_name: str
            The name of the wandb run to log the clustering visualization.

        Returns
        --------
        None
        """
        # Initialize wandb
        run = wandb.init(project=project_name, name=run_name)

        # Perform K-means clustering
        cluster_centers, cluster_indices = self.cluster_kmeans(data, n_clusters)
        tsne = TSNE(n_components=2)
        data_2d = tsne.fit_transform(data)

        # Plot the clusters
        # TODO
        plt.figure(figsize=(10, 7))
        scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], c=cluster_indices, cmap='viridis')
        plt.title(f'K-means Clustering with {n_clusters} Clusters')
        plt.xlabel('TSNE Component 1')
        plt.ylabel('TSNE Component 2')
        plt.colorbar(scatter)
        
        # Log the plot to wandb
        # TODO
        wandb.log({"K-means Clustering": wandb.Image(plt)})

        # Close the plot display window if needed (optional)
        # TODO
        plt.close()
        run.finish()

    def wandb_plot_hierarchical_clustering_dendrogram(self, data, project_name, linkage_method, run_name, labels=None):
        """ This function performs hierarchical clustering on the provided data and generates a dendrogram plot, which is then logged to Weights & Biases (wandb).

        The dendrogram is a tree-like diagram that visualizes the hierarchical clustering process. It shows how the data points (or clusters) are progressively merged into larger clusters based on their similarity or distance.

        The function performs the following steps:
        1. Initialize a new wandb run with the provided project and run names.
        2. Perform hierarchical clustering on the input data using the specified linkage method.
        3. Create a linkage matrix, which represents the merging of clusters at each step of the hierarchical clustering process.
        4. Generate a dendrogram plot using the linkage matrix.
        5. Log the dendrogram plot as an image to the wandb run.
        6. Close the plot display window to conserve system resources.

        Parameters
        -----------
        data: np.ndarray
            The input data to perform hierarchical clustering on.
        linkage_method: str
            The linkage method for hierarchical clustering. It can be one of the following: "average", "ward", "complete", or "single".
        project_name: str
            The name of the wandb project to log the dendrogram plot.
        run_name: str
            The name of the wandb run to log the dendrogram plot.

        Returns
        --------
        None
        """
        run = wandb.init(project=project_name, name=run_name)
        # Perform hierarchical clustering
        # TODO
        linkage_matrix = linkage(data, method=linkage_method)

        # Create linkage matrix for dendrogram
        # TODO
        plt.figure(figsize=(15, 10))
        dendrogram(linkage_matrix, labels=labels, leaf_rotation=90)
        plt.title(f'Hierarchical Clustering Dendrogram ({linkage_method} linkage)')
        plt.xlabel('Sample index')
        plt.ylabel('Distance')
        plt.subplots_adjust(bottom=0.4)

        wandb.log({"Hierarchical Clustering Dendrogram": wandb.Image(plt)})
        plt.close()
        run.finish()


    def plot_kmeans_cluster_scores(self, embeddings: List, true_labels: List, k_values: List[int], project_name=None,
                                   run_name=None):
        """ This function, using implemented metrics in clustering_metrics, calculates and plots both purity scores and silhouette scores for various numbers of clusters.
        Then using wandb plots the respective scores (each with a different color) for each k value.

        Parameters
        -----------
        embeddings : List
            A list of vectors representing the data points.
        true_labels : List
            A list of ground truth labels for each data point.
        k_values : List[int]
            A list containing the various values of 'k' (number of clusters) for which the scores will be calculated.
            Default is range(2, 9), which means it will calculate scores for k values from 2 to 8.
        project_name : str
            Your wandb project name. If None, the plot will not be logged to wandb. Default is None.
        run_name : str
            Your wandb run name. If None, the plot will not be logged to wandb. Default is None.

        Returns
        --------
        None
        """
        silhouette_scores = []
        purity_scores = []
        adjusted_rand_scores = []
        num_tries = 10
        # Calculating Silhouette Scores and Purity Scores for different values of k
        for k in k_values:
            # TODO
            silhouette_score = 0
            purity_score = 0
            adjusted_rand_score = 0
            for _ in range(num_tries):
                centers, cluster_indices = self.cluster_kmeans(embeddings, k)
                silhouette_score += ClusteringMetrics().silhouette_score(embeddings, cluster_indices)
                purity_score += ClusteringMetrics().purity_score(true_labels, cluster_indices)
                adjusted_rand_score += ClusteringMetrics().adjusted_rand_score(true_labels, cluster_indices)
            silhouette_score /= num_tries
            purity_score /= num_tries
            adjusted_rand_score /= num_tries
            silhouette_scores.append(silhouette_score)
            purity_scores.append(purity_score)
            adjusted_rand_scores.append(adjusted_rand_score)
            # Using implemented metrics in clustering_metrics, get the score for each k in k-means clustering
            # and visualize it.
            # TODO

            

        # Plotting the scores
        # TODO
        plt.plot(k_values, silhouette_scores)
        plt.plot(k_values, purity_scores)
        plt.plot(k_values, adjusted_rand_scores)
        plt.xticks(k_values)
        plt.xlabel('Number of Clusters')
        plt.ylabel('Score')
        plt.legend(['Silhouette Score', 'Purity Score', 'Adjusted Rand Score'])
        plt.title('K-Means Clustering Scores')

        

        # Logging the plot to wandb
        if project_name and run_name:
            import wandb
            run = wandb.init(project=project_name, name=run_name)
            wandb.log({"Cluster Scores": wandb.Image(plt)})
            metrics_table = wandb.Table(data=[[k_values[i], silhouette_scores[i], purity_scores[i], adjusted_rand_scores[i]] for i in range(len(k_values))], 
                                    columns=["K", "Silhouette Score", "Purity Score", "Adjusted Rand Score"])
            wandb.log({"Cluster Metrics": metrics_table})

        plt.close()
        run.finish()


    def visualize_elbow_method_wcss(self, embeddings: List, k_values: List[int], project_name: str, run_name: str):
        """ This function implements the elbow method to determine the optimal number of clusters for K-means clustering based on the Within-Cluster Sum of Squares (WCSS).

        The elbow method is a heuristic used to determine the optimal number of clusters in K-means clustering. It involves plotting the WCSS values for different values of K (number of clusters) and finding the "elbow" point in the curve, where the marginal improvement in WCSS starts to diminish. This point is considered as the optimal number of clusters.

        The function performs the following steps:
        1. Iterate over the specified range of K values.
        2. For each K value, perform K-means clustering using the `cluster_kmeans_WCSS` function and store the resulting WCSS value.
        3. Create a line plot of WCSS values against the number of clusters (K).
        4. Log the plot to Weights & Biases (wandb) for visualization and tracking.

        Parameters
        -----------
        embeddings: List
            A list of vectors representing the data points to be clustered.
        k_values: List[int]
            A list of K values (number of clusters) to explore for the elbow method.
        project_name: str
            The name of the wandb project to log the elbow method plot.
        run_name: str
            The name of the wandb run to log the elbow method plot.

        Returns
        --------
        None
        """
        # Initialize wandb
        run = wandb.init(project=project_name, name=run_name)

        # Compute WCSS values for different K values
        wcss_values = []
        for k in k_values:
            # TODO
            wcss = 0
            for _ in range(25):
                _, _, temp = self.cluster_kmeans_WCSS(embeddings, k)
                wcss += temp
            wcss /= 25
            wcss_values.append(wcss)


        # Plot the elbow method
        # TODO
        plt.figure(figsize=(10, 7))
        plt.plot(k_values, wcss_values, marker='o')
        plt.title('Elbow Method for Optimal K')
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
        plt.xticks(k_values)
        plt.grid(True)


        # Log the plot to wandb
        wandb.log({"Elbow Method": wandb.Image(plt)})

        plt.close()
        run.finish()
