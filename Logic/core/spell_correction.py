class SpellCorrection:
    def __init__(self, all_documents):
        """
        Initialize the SpellCorrection

        Parameters
        ----------
        all_documents : list of str
            The input documents.
        """
        all_documents = [document.lower() for document in all_documents]
        self.all_shingled_words, self.word_counter = self.shingling_and_counting(all_documents)

    def shingle_word(self, word, k=3):
        """
        Convert a word into a set of shingles.

        Parameters
        ----------
        word : str
            The input word.
        k : int
            The size of each shingle.

        Returns
        -------
        set
            A set of shingles.
        """
        shingles = set()
        
        # TODO: Create shingle here
        for i in range(len(word) - k + 1):
            shingles.add(word[i:i+k])

        return shingles
    
    def jaccard_score(self, first_set, second_set):
        """
        Calculate jaccard score.

        Parameters
        ----------
        first_set : set
            First set of shingles.
        second_set : set
            Second set of shingles.

        Returns
        -------
        float
            Jaccard score.
        """

        # TODO: Calculate jaccard score here.
        intersection = len(first_set.intersection(second_set))
        union = len(first_set.union(second_set))
        return intersection / union if union != 0 else 0

    def shingling_and_counting(self, all_documents):
        """
        Shingle all words of the corpus and count TF of each word.

        Parameters
        ----------
        all_documents : list of str
            The input documents.

        Returns
        -------
        all_shingled_words : dict
            A dictionary from words to their shingle sets.
        word_counter : dict
            A dictionary from words to their TFs.
        """
        all_shingled_words = dict()
        word_counter = dict()

        # TODO: Create shingled words dictionary and word counter dictionary here.
        for document in all_documents:
            for word in document.split():
                if word not in all_shingled_words:
                    all_shingled_words[word] = self.shingle_word(word)
                if word not in word_counter:
                    word_counter[word] = 1
                else:
                    word_counter[word] += 1
                
        return all_shingled_words, word_counter
    
    def find_nearest_words(self, word):
        """
        Find correct form of a misspelled word.

        Parameters
        ----------
        word : stf
            The misspelled word.

        Returns
        -------
        list of str
            5 nearest words.
        """
        top5_candidates = list()

        # TODO: Find 5 nearest candidates here.
        for candidate in self.all_shingled_words:
            top5_candidates.append((candidate, self.jaccard_score(self.shingle_word(word), self.all_shingled_words[candidate])))

        top5_candidates.sort(key=lambda x: x[1], reverse=True)
        top5_candidates = [candidate[0] for candidate in top5_candidates[:5]]

        return top5_candidates
    
    def spell_check(self, query):
        """
        Find correct form of a misspelled query.

        Parameters
        ----------
        query : stf
            The misspelled query.

        Returns
        -------
        str
            Correct form of the query.
        """
        final_result = ""
        
        # TODO: Do spell correction here.
        for word in query.split():
            word = word.lower()
            top5_candidates = self.find_nearest_words(word)
            top5_tfs = [self.word_counter[candidate] for candidate in top5_candidates]
            normalized_tfs = [tf / max(top5_tfs) for tf in top5_tfs]
            jaccard_scores = [self.jaccard_score(self.shingle_word(word), self.all_shingled_words[candidate]) for candidate in top5_candidates]
            scores = [normalized_tfs[i] * jaccard_scores[i] for i in range(5)]
            final_result += top5_candidates[scores.index(max(scores))] + " "

        final_result = final_result[:-1]

        for i, word in enumerate(query.split()):
            if word[0].isupper():
                final_result = final_result.split()
                final_result[i] = final_result[i].capitalize()
                final_result = ' '.join(final_result)
            if word.isupper():
                final_result = final_result.split()
                final_result[i] = final_result[i].upper()
                final_result = ' '.join(final_result)
            
        return final_result 