import os
import sys
current_script_path = os.path.abspath(__file__)
vec_dir = os.path.dirname(current_script_path)
core_dir = os.path.dirname(vec_dir)
Logic_dir = os.path.dirname(core_dir)
project_root = os.path.dirname(Logic_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

import pickle

from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores.utils import DistanceStrategy
from langchain.vectorstores.faiss import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

import pandas as pd
from tqdm import tqdm

def vectorize():
    EMBEDDING_MODEL_NAME="thenlper/gte-base"

    df = pd.read_json(project_root+'/data/IMDB_Crawled.json')
    df = df[['id', 'title', 'genres', 'rating', 'release_year', 'summaries', 'stars', 'directors']]
    df = df.dropna()
    df = df.reset_index(drop=True)
    df.columns = ['Movie ID', 'Title','Genres','Movie Rating','Release Year','Movie Plot', 'Stars', 'Directors']
    df['Movie Plot'] = df['Movie Plot'].apply(lambda x: x[0])
    df['Genres'] = df['Genres'].apply(lambda x: ' '.join(x))
    df['Stars'] = df['Stars'].apply(lambda x: ', '.join(x))
    df['Directors'] = df['Directors'].apply(lambda x: ', '.join(x))
    df['rating'] = df['Movie Rating']
    df.to_csv(project_root+'/data/tmp.csv', index=False)


    # Load the CSV
    loader = CSVLoader(file_path=project_root+'/data/tmp.csv', metadata_columns=['Movie ID', 'rating'])
    documents = loader.load()

    # Embed the documents using the model in a vectorstore
    vectorstore = FAISS.from_documents(documents, HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME), distance_strategy=DistanceStrategy.COSINE)

    # Save the vectorstore
    with open(project_root+"/data/vectorstore.pkl", "wb") as f:
        pickle.dump(vectorstore, f)

    # Remove the temporary file
    os.remove(project_root+'/data/tmp.csv')

if __name__ == '__main__':
    vectorize()