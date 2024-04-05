import numpy as np
import itertools
import random


class MinHashLSH:
    def __init__(self, documents, num_hashes):
        """
        Initialize the MinHashLSH

        Parameters
        ----------
        documents : list of str
            The input documents for similarity analysis.
        num_hashes : int
            Number of hashes for mini-hashing.
        """
        self.documents = documents
        self.num_hashes = num_hashes
        self.all_shingles = set()
        self.doc_shingles = {}
        for i, document in enumerate(self.documents):
            shingles = self.shingle_document(document)
            self.doc_shingles[i] = shingles
        self.hash_functions = [self._make_hash_function(i) for i in range(num_hashes)]

    def shingle_document(self, document, k=2):
        """
        Convert a document into a set of shingles.

        Parameters
        ----------
        document : str
            The input document.
        k : int
            The size of each shingle.

        Returns
        ----------
        set
            A set of shingles.
        """
        shingles = set()
        for i in range(len(document) - k + 1):
            shingle = document[i:i + k]
            shingles.add(shingle)
            self.all_shingles.add(shingle)

        return shingles

    def build_characteristic_matrix(self):
        """
        Build the characteristic matrix representing the presence of shingles in documents.

        Returns
        ----------
        numpy.ndarray
            The binary characteristic matrix.
        """

        shingles = list(self.all_shingles)
        shingles.sort()
        matrix = np.zeros((len(shingles), len(self.documents)))

        for i, shingle in enumerate(shingles):
            for j in range(len(self.documents)):
                if shingle in self.doc_shingles[j]:
                    matrix[i, j] = 1

        return matrix
    

    def _make_hash_function(self, seed):
        def hash_func(x):
            random.seed(x + seed)
            return random.randint(0, 2**16 - 1)
        return hash_func


    def min_hash_signature(self):
        """
        Perform Min-Hashing to generate hash signatures for documents.

        Returns
        ----------
        numpy.ndarray
            The Min-Hash signatures matrix.
        """
        num_docs = len(self.documents)
        signature_matrix = np.full((self.num_hashes, num_docs), np.inf)

        for doc_idx, shingles in self.doc_shingles.items():
            for i, hash_func in enumerate(self.hash_functions):
                for shingle in shingles:
                    hash_value = hash_func(hash(shingle))
                    signature_matrix[i, doc_idx] = min(signature_matrix[i, doc_idx], hash_value)
            
        return signature_matrix

        
    def lsh_buckets(self, signature, bands=10, rows_per_band=10):
        """
        Group documents into Locality-Sensitive Hashing (LSH) buckets based on Min-Hash signatures.

        Parameters
        ----------
        signature : numpy.ndarray
            Min-Hash signatures for documents.
        bands : int
            Number of bands for LSH.
        rows_per_band : int
            Number of rows per band.

        Returns
        ----------
        dict
            A dictionary mapping bucket IDs to lists of document indices.
        """
        # TODO
        buckets = {}
        num_docs = signature.shape[1]
        for band in range(bands):
            for doc_idx in range(num_docs):
                start_row = band * rows_per_band
                end_row = start_row + rows_per_band
                sub_signature = tuple(signature[start_row:end_row, doc_idx])

                hash_id = hash(sub_signature)
                bucket_id = int(str(hash_id) + str(band))

                if bucket_id not in buckets:
                    buckets[bucket_id] = [doc_idx]
                else:
                    buckets[bucket_id].append(doc_idx)

        return buckets
    

    def perform_lsh(self):
        """
        Perform the entire Locality-Sensitive Hashing (LSH) process.

        Returns
        ----------
        dict
            A dictionary mapping bucket IDs to lists of document indices.
        """
        signature = self.min_hash_signature()
        buckets = self.lsh_buckets(signature)
        return buckets

    def jaccard_score(self, first_set, second_set):
        """
        Calculate jaccard score for two sets.

        Parameters
        ----------
        first_set : set
            Set of first shingled document.
        second_set : set
            Set of second shingled document.

        Returns
        ----------
        float
            Jaccard score.
        """
        union = len(first_set.union(second_set))
        intersection = len(first_set.intersection(second_set))
        jaccard_score = intersection / union
        return jaccard_score

    def jaccard_similarity_test(self, buckets, all_documents):
        """
        Test your near duplicate detection code based on jaccard similarity.

        Parameters
        ----------
        buckets : dict
            A dictionary mapping bucket IDs to lists of document indices.
        all_documents : list
            The input documents for similarity analysis.
        """
        correct_near_duplicates = 0
        all_near_duplicates = 0

        for bucket_id in buckets.keys():
            docs_in_this_bucket = buckets[bucket_id]
            unique_doc_ids = set(docs_in_this_bucket)
            if len(unique_doc_ids) > 1:
                combinations = list(itertools.combinations(unique_doc_ids, 2))
                for comb in combinations:
                    all_near_duplicates += 1

                    first_doc_id = comb[0]
                    second_doc_id = comb[1]

                    first_shingled_doc = self.shingle_document(all_documents[first_doc_id], 2)
                    second_shingled_doc = self.shingle_document(all_documents[second_doc_id], 2)

                    near_duplicated_jaccard_score = self.jaccard_score(first_shingled_doc, second_shingled_doc)
                    current_score = 0

                    for _ in range(5):
                        random_doc_id = first_doc_id
                        while random_doc_id == first_doc_id or random_doc_id == second_doc_id:
                            random_doc_id = random.randint(0, len(all_documents) - 1)
                        random_shingled_doc = self.shingle_document(all_documents[random_doc_id], 2)

                        random_jaccard_score = self.jaccard_score(first_shingled_doc, random_shingled_doc)

                        if near_duplicated_jaccard_score > random_jaccard_score:
                            current_score += 1

                    if current_score == 5:
                        correct_near_duplicates += 1

        # a good score is around 0.8
        print("your final score in near duplicate detection:", correct_near_duplicates / all_near_duplicates)


import json

if __name__ == '__main__':
    docs = []
    with open('Logic/core/LSHFakeData.json') as f:
        docs = json.load(f)

    summaris = []
    for doc in docs:
        temp = doc['summaries']
        summary = ''
        for t in temp:
            summary += t
        summaris.append(summary)

    num_hashes = 100
    min_hash_lsh = MinHashLSH(summaris, num_hashes)
    buckets = min_hash_lsh.perform_lsh()

    printed_buckets = []
    print("Buckets:")
    for bucket_id, bucket in buckets.items():
        if len(bucket) > 1 and bucket not in printed_buckets:
            print(bucket, end=' ')
            printed_buckets.append(bucket)
    print()
    min_hash_lsh.jaccard_similarity_test(buckets, summaris)


    # Outputs:
    
    # Buckets:
    # [0, 1] [6, 7] [14, 15] [18, 19] [8, 9] [12, 13] [16, 17] [2, 3] [4, 5] [10, 11] [10, 11, 14] 
    # your final score in near duplicate detection: 0.9459459459459459
        
        
