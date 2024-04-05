import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import json
class Snippet:
    def __init__(self, number_of_words_on_each_side=5):
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
        query = query.split()
        query = [word for word in query if word not in stop_words]
        query = ' '.join(query)
        return query

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
        query = query.split()
        occurances_dict = {query_word:[] for query_word in query}

        for i in range(len(doc)):
            if doc[i] in query or doc[i].lower() in query or doc[i].upper() in query or doc[i].capitalize() in query:
                try:
                    occurances_dict[doc[i]].append(i)
                except:
                    try:
                        occurances_dict[doc[i].lower()].append(i)
                    except:
                        try:
                            occurances_dict[doc[i].upper()].append(i)
                        except:
                            occurances_dict[doc[i].capitalize()].append(i)
                

        for query_word in query:
            if len(occurances_dict[query_word]) == 0:
                not_exist_words.append(query_word)
                
        all_occurances = []
        for query_word in query:
            all_occurances += occurances_dict[query_word]
        all_occurances.sort()

        occurance_groups = []

        for occurance in all_occurances:
            if len(occurance_groups) == 0 or occurance - occurance_groups[-1][-1] > 1 + 2 * self.number_of_words_on_each_side:
                occurance_groups.append([occurance])
            else:
                occurance_groups[-1].append(occurance)
        
        for occurance_group in occurance_groups:
            start = max(0, occurance_group[0] - self.number_of_words_on_each_side)
            end = min(len(doc), occurance_group[-1] + self.number_of_words_on_each_side + 1)
            snippet = ' '.join(doc[start:end])
            for occurance in occurance_group:
                query_w = doc[occurance]
                snippet = snippet.replace(query_w, f'***{query_w}***')
            final_snippet += snippet + ' ... '

        return final_snippet, not_exist_words
