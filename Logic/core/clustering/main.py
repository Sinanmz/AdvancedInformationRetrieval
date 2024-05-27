import os
import sys
current_script_path = os.path.abspath(__file__)
clustering_dir = os.path.dirname(current_script_path)
core_dir = os.path.dirname(clustering_dir)
Logic_dir = os.path.dirname(core_dir)
project_root = os.path.dirname(Logic_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from Logic.core.word_embedding.fasttext_data_loader import FastTextDataLoader
from Logic.core.word_embedding.fasttext_model import FastText
from Logic.core.word_embedding.fasttext_model import preprocess_text

from Logic.core.clustering.dimension_reduction import DimensionReduction
from Logic.core.clustering.clustering_metrics import ClusteringMetrics
from Logic.core.clustering.clustering_utils import ClusteringUtils

import json
import matplotlib.pyplot as plt
import wandb

# Main Function: Clustering Tasks

# 0. Embedding Extraction
# TODO: Using the previous preprocessor and fasttext model, collect all the embeddings of our data and store them.
ft_model = FastText()

if not os.path.exists(project_root+'/models/FastText_model.bin'):
    crawled_path = project_root+'/data/IMDB_Crawled.json'
    ft_data_loader = FastTextDataLoader(crawled_path)
    X, y = ft_data_loader.create_train_data()
    ft_model.prepare(dataset=X, mode="train", save=True)
else:
    ft_model.prepare(dataset=None, mode="load", save=False, 
                        path=project_root+'/models/FastText_model.bin')

crawled_path = project_root+'/data/IMDB_Crawled.json'
with open(crawled_path, 'r') as f:
    data = json.load(f)

num_samples = 100

movie_titles = []
movie_embeddings = []
movie_genres = []
for movie in tqdm(data[:num_samples], desc='Extracting Movie Embeddings'):
    if movie['synposis'] and movie['reviews'] and movie['summaries']and movie['title'] and movie['genres']:
        movie_titles.append(movie['title'])
        synopsis_vector = ft_model.get_query_embedding(preprocess_text(' '.join(movie['synposis'])))
        summaries_vector = ft_model.get_query_embedding(preprocess_text(' '.join(movie['summaries'])))
        reviews_vector = ft_model.get_query_embedding(preprocess_text(' '.join([i[0] for i in movie['reviews']])))
        movie_embeddings.append(np.concatenate([synopsis_vector, summaries_vector, reviews_vector]))
        movie_genres.append(movie['genres'][0])


genre2idx = {genre: idx for idx, genre in enumerate(set(movie_genres))}
idx2genre = {idx: genre for genre, idx in genre2idx.items()}
movie_genres_ids = [genre2idx[genre] for genre in movie_genres]

embeddings = np.array(movie_embeddings)
# print(embeddings)

# 1. Dimension Reduction
# TODO: Perform Principal Component Analysis (PCA):
#     - Reduce the dimensionality of features using PCA. (you can use the reduced feature afterward or use to the whole embeddings)
#     - Find the Singular Values and use the explained_variance_ratio_ attribute to determine the percentage of variance explained by each principal component.
#     - Draw plots to visualize the results.
dimension_reduction = DimensionReduction()
reduced_embeddings = dimension_reduction.pca_reduce_dimension(embeddings, n_components=10)
dimension_reduction.wandb_plot_explained_variance_by_components(embeddings, project_name=f'MIR_Clustering_{num_samples}', run_name='Explained Variance')
for i, explained_variance in enumerate(dimension_reduction.pca.explained_variance_ratio_):
    print(f"Explained Variance for Component {i+1}: {explained_variance}")
    print(f"Singluar Value for Component {i+1}: {dimension_reduction.pca.singular_values_[i]}")

# TODO: Implement t-SNE (t-Distributed Stochastic Neighbor Embedding):
#     - Create the convert_to_2d_tsne function, which takes a list of embedding vectors as input and reduces the dimensionality to two dimensions using the t-SNE method.
#     - Use the output vectors from this step to draw the diagram.
# dimension_reduction.wandb_plot_2d_tsne(embeddings, project_name=f'Clustering_{num_samples}', run_name='2D t-SNE')

# 2. Clustering
## K-Means Clustering
# TODO: Implement the K-means clustering algorithm from scratch.
# TODO: Create document clusters using K-Means.
# TODO: Run the algorithm with several different values of k.
# TODO: For each run:
#     - Determine the genre of each cluster based on the number of documents in each cluster.
#     - Draw the resulting clustering using the two-dimensional vectors from the previous section.
#     - Check the implementation and efficiency of the algorithm in clustering similar documents.
k = [5, 10, 15, 20, 25]
for k_i in k:
    cluster_centers, clusters = ClusteringUtils().cluster_kmeans(embeddings, n_clusters=k_i)
    cluster_genres = []
    for cluster in range(k_i):
        cluster_genres.append(np.argmax(np.bincount([movie_genres_ids[i] for i in range(len(clusters)) if clusters[i] == cluster])))
    print(f"Genres for k={k_i}: {[idx2genre[genre] for genre in cluster_genres]}")
    ClusteringUtils().visualize_kmeans_clustering_wandb(embeddings, k_i, project_name=f'MIR_Clustering_{num_samples}', run_name=f'KMeans_{k_i}')

k = [i+5 for i in range(46)]
ClusteringUtils().visualize_elbow_method_wcss(embeddings, k, project_name=f'MIR_Clustering_{num_samples}', run_name='Elbow Method WCSS')

# TODO: Draw the silhouette score graph for different values of k and perform silhouette analysis to choose the appropriate k.
# TODO: Plot the purity value for k using the labeled data and report the purity value for the final k. (Use the provided functions in utilities)
silhouette_scores = []
purity_scores = []
adjusted_rand_scores = []
num_tries = 10
for k_i in k:
    silhouette_score = 0
    purity_score = 0
    adjusted_rand_score = 0
    for _ in range(num_tries):
        cluster_centers, clusters = ClusteringUtils().cluster_kmeans(embeddings, n_clusters=k_i)
        silhouette_score += ClusteringMetrics().silhouette_score(embeddings, clusters)
        purity_score += ClusteringMetrics().purity_score(movie_genres_ids, clusters)
        adjusted_rand_score += ClusteringMetrics().adjusted_rand_score(movie_genres_ids, clusters)
    silhouette_score /= num_tries
    purity_score /= num_tries
    adjusted_rand_score /= num_tries
    silhouette_scores.append(silhouette_score)
    purity_scores.append(purity_score)
    adjusted_rand_scores.append(adjusted_rand_score)
    print(f"Silhouette Score for k={k_i}: {silhouette_score}")
    print(f"Purity Score for k={k_i}: {purity_score}")
    print(f"Adjusted Rand Score for k={k_i}: {adjusted_rand_score}")

ClusteringUtils().plot_kmeans_cluster_scores(embeddings, movie_genres_ids, k, project_name=f'MIR_Clustering_{num_samples}', run_name='KMeans Scores')

## Hierarchical Clustering
# TODO: Perform hierarchical clustering with all different linkage methods.
# TODO: Visualize the results.
# 3. Evaluation
# TODO: Using clustering metrics, evaluate how well your clustering method is performing.
silhouette_scores = []
purity_scores = []
adjusted_rand_scores = []
linkage_methods = ['ward', 'complete', 'average', 'single']
for method in linkage_methods:
    if method == 'ward':
        cluster_labels = ClusteringUtils().cluster_hierarchical_ward(embeddings)
    elif method == 'complete':
        cluster_labels = ClusteringUtils().cluster_hierarchical_complete(embeddings)
    elif method == 'average':
        cluster_labels = ClusteringUtils().cluster_hierarchical_average(embeddings)
    elif method == 'single':
        cluster_labels = ClusteringUtils().cluster_hierarchical_single(embeddings)
        
    silhouette_score = ClusteringMetrics().silhouette_score(embeddings, cluster_labels)
    purity_score = ClusteringMetrics().purity_score(movie_genres_ids, cluster_labels)
    adjusted_rand_score = ClusteringMetrics().adjusted_rand_score(movie_genres_ids, cluster_labels)
    
    silhouette_scores.append(silhouette_score)
    purity_scores.append(purity_score)
    adjusted_rand_scores.append(adjusted_rand_score)
    print(f"Silhouette Score for Hierarchical Clustering with {method} linkage: {silhouette_score}")
    print(f"Purity Score for Hierarchical Clustering with {method} linkage: {purity_score}")
    print(f"Adjusted Rand Score for Hierarchical Clustering with {method} linkage: {adjusted_rand_score}")
    ClusteringUtils().wandb_plot_hierarchical_clustering_dendrogram(data=embeddings, project_name=f'MIR_Clustering_{num_samples}', 
                                                                    linkage_method=method, run_name=f"Hierarchical Clustering Dendrogram with {method} linkage", labels=movie_titles)

run = wandb.init(project=f'MIR_Clustering_{num_samples}', name='Hierarchical Clustering Scores')
plt.scatter(linkage_methods, silhouette_scores)
plt.scatter(linkage_methods, purity_scores)
plt.scatter(linkage_methods, adjusted_rand_scores)
plt.xlabel('Linkage Method')
plt.ylabel('Score')
plt.legend(['Silhouette Score', 'Purity Score', 'Adjusted Rand Score'])
plt.title('Hierarchical Clustering Scores')
metrics_table = wandb.Table(data = [[linkage_methods[i], silhouette_scores[i], purity_scores[i], adjusted_rand_scores[i]] for i in range(len(linkage_methods))], 
                            columns = ["Linkage Method", "Silhouette Score", "Purity Score", "Adjusted Rand Score"])

wandb.log({"Hierarchical Clustering Scores": wandb.Image(plt)})
wandb.log({"Hierarchical Clustering Metrics": metrics_table})

plt.close()
run.finish()








