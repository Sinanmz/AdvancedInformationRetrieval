import os
import sys
current_script_path = os.path.abspath(__file__)
core_dir = os.path.dirname(current_script_path)
Logic_dir = os.path.dirname(core_dir)
project_root = os.path.dirname(Logic_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

import re
import nltk
from nltk.corpus import stopwords

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

class Snippet:
    def __init__(self, number_of_words_on_each_side=3):
        """
        Initialize the Snippet

        Parameters
        ----------
        number_of_words_on_each_side : int
            The number of words on each side of the query word in the doc to be presented in the snippet.
        """
        self.number_of_words_on_each_side = number_of_words_on_each_side

    def remove_stop_words_from_query(self, query):
        """
        Remove stop words from the input string.

        Parameters
        ----------
        query : str
            The query that you need to delete stop words from.

        Returns
        -------
        str
            The query without stop words.
        """

        # TODO: remove stop words from the query.
        stop_words = set(stopwords.words('english'))
        stop_words_path = project_root+'/Logic/core/stopwords.txt'
        with open(stop_words_path, 'r') as file:
            additional_stopwords = file.read().splitlines()

        stop_words = stop_words.union(set(additional_stopwords))
        query = query.split()

        new_query = []
        for word in query:
            match_end = re.search(r"([\W_]+)$", word)
            if match_end != None:
                match_end = match_end.group(1)
            else:
                match_end = ''

            match_start = re.search(r"^([\W_]+)", word)
            if match_start != None:
                match_start = match_start.group(1)
            else:
                match_start = ''
            word = word[:len(word)-len(match_end)]
            word = word[len(match_start):]
            word = word.lower()
            if word not in stop_words:
                new_query.append(word)
        return new_query

    def find_snippet(self, doc, query):
        """
        Find snippet in a doc based on a query.

        Parameters
        ----------
        doc : str
            The retrieved doc which the snippet should be extracted from that.
        query : str
            The query which the snippet should be extracted based on that.

        Returns
        -------
        final_snippet : str
            The final extracted snippet. IMPORTANT: The keyword should be wrapped by *** on both sides.
            For example: Sahwshank ***redemption*** is one of ... (for query: redemption)
        not_exist_words : list
            Words in the query which don't exist in the doc.
        """
        final_snippet = ""
        not_exist_words = []

        # TODO: Extract snippet and the tokens which are not present in the doc.

        doc = doc.split()
        query = self.remove_stop_words_from_query(query)
        occurances_dict = {query_word:{} for query_word in query}

        for i in range(len(doc)):
            temp = re.sub(r'[^\w\s]', '', doc[i])

            match_end = re.search(r"([\W_]+)$", doc[i])
            if match_end != None:
                match_end = match_end.group(1)
            else:
                match_end = ''

            match_start = re.search(r"^([\W_]+)", doc[i])
            if match_start != None:
                match_start = match_start.group(1)
            else:
                match_start = ''
            temp = doc[i][:len(doc[i])-len(match_end)]
            temp = temp[len(match_start):]

            if temp.lower() in query:
                occurances_dict[temp.lower()][i] = 0
        
        for query_word in query:
            if len(occurances_dict[query_word]) == 0:
                not_exist_words.append(query_word)
        
        best_occurrences = {query_word:None for query_word in occurances_dict if query_word not in not_exist_words}

        for query_word in occurances_dict:
            for i in occurances_dict[query_word]:
                for other_query_word in occurances_dict:
                    if other_query_word != query_word:
                        for j in occurances_dict[other_query_word]:
                            if abs(i-j) <= self.number_of_words_on_each_side:
                                occurances_dict[query_word][i] += 1
        
        for query_word in occurances_dict:
            if query_word not in not_exist_words:
                query_word_best_point = max(occurances_dict[query_word].values())
                best_occurrences[query_word] = []
                for i in occurances_dict[query_word]:
                    if occurances_dict[query_word][i] == query_word_best_point:
                        best_occurrences[query_word].append(i)
        
        all_occurances = []
        for query_word in best_occurrences:
            if best_occurrences[query_word] != None:
                all_occurances.extend(best_occurrences[query_word])
        all_occurances.sort()

        occurance_groups = []

        for occurance in all_occurances:
            if len(occurance_groups) == 0 or occurance - occurance_groups[-1][-1] > 1 + 2 * self.number_of_words_on_each_side:
                occurance_groups.append([occurance])
            else:
                occurance_groups[-1].append(occurance)

        for occurance_groups in occurance_groups:
            start = max(0, occurance_groups[0] - self.number_of_words_on_each_side)
            end = min(len(doc), occurance_groups[-1] + self.number_of_words_on_each_side + 1)
            snippet = ' '.join(doc[start:end])
            for occurance in occurance_groups:
                replacement = doc[occurance]
                snippet = snippet.replace(replacement, f' ***{replacement}*** ')
            final_snippet += snippet + ' ... '

        final_snippet = final_snippet[:-5]
        final_snippet = final_snippet.replace('  ',' ')

        return final_snippet, not_exist_words
