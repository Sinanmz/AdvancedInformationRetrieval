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
        stop_words_path = project_root+'/Logic/core/stopwords.txt'
        with open(stop_words_path, 'r') as file:
            additional_stopwords = file.read().splitlines()

        stop_words.update(additional_stopwords)
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

        ##### My comment: The code below highlights every query word also taking into account the capitalization and uppercasing.
        #-------------------------------------------------------------------------------------------
        # doc = doc.split()
        # query = self.remove_stop_words_from_query(query)
        # query = query.split()
        # occurances_dict = {query_word:[] for query_word in query}

        # punctuation_marks = ['-', ',', '.', ' ', ':', ';', '\'', '\"', '!', '?', '(', ')', '[', ']']

        # for query_word in query:
        #     for i, text in enumerate(doc):
        #         # Create a regex pattern that matches any of the punctuation marks
        #         pattern = '|'.join(map(re.escape, punctuation_marks))
        #         # Split the text using the created pattern
        #         words = re.split(pattern, text.lower())
        #         # If the query_word is in the split words list, record the occurrence
        #         if query_word.lower() in words:
        #             occurances_dict[query_word].append(i)
                
                

        # for query_word in query:
        #     if len(occurances_dict[query_word]) == 0:
        #         not_exist_words.append(query_word)
                
        # all_occurances = []
        # for query_word in query:
        #     all_occurances += occurances_dict[query_word]
        # all_occurances.sort()

        # occurance_groups = []

        # for occurance in all_occurances:
        #     if len(occurance_groups) == 0 or occurance - occurance_groups[-1][-1] > 1 + 2 * self.number_of_words_on_each_side:
        #         occurance_groups.append([occurance])
        #     else:
        #         occurance_groups[-1].append(occurance)
        
        # taken = {query_w: False for query_w in query}

        # for occurance_group in occurance_groups:
        #     start = max(0, occurance_group[0] - self.number_of_words_on_each_side)
        #     end = min(len(doc), occurance_group[-1] + self.number_of_words_on_each_side + 1)
        #     snippet = ' '.join(doc[start:end])
        #     for occurance in occurance_group:
        #         for q_w, occurance_w in occurances_dict.items():
        #             if occurance in occurance_w:
        #                 query_w = q_w
        #                 break
        #         if taken[query_w]:
        #             continue
        #         index = doc[occurance].lower().find(query_w.lower())
        #         replacement = doc[occurance][index:index+len(query_w)]
        #         snippet = snippet.replace(replacement, f' ***{replacement}*** ')

        #         taken[query_w] = True
        #     final_snippet += snippet + ' ... '
        #-------------------------------------------------------------------------------------------

        ##### My comment: The code below highloghts best occurance of aquery word also taking into account the capitalization and uppercasing.
        #-------------------------------------------------------------------------------------------
        # doc = doc.split()
        # query = self.remove_stop_words_from_query(query)
        # query = query.split()
        # occurances_dict = {query_word:{} for query_word in query}

        # for i in range(len(doc)):
        #     temp = re.sub(r'[^\w\s]', '', doc[i])
        #     if temp in query or temp.lower() in query or temp.upper() in query or temp.capitalize() in query:
        #         try:
        #             occurances_dict[temp][i] = 0
        #         except:
        #             try:
        #                 occurances_dict[temp.lower()][i] = 0
        #             except:
        #                 try:
        #                     occurances_dict[temp.upper()][i] = 0
        #                 except:
        #                     occurances_dict[temp.capitalize()][i] = 0

        # for query_word in query:
        #     if len(occurances_dict[query_word]) == 0:
        #         not_exist_words.append(query_word)

        # for query_word in occurances_dict:
        #     for i in occurances_dict[query_word]:
        #         for other_query_word in occurances_dict:
        #             if other_query_word != query_word:
        #                 for j in occurances_dict[other_query_word]:
        #                     if abs(i-j) <= self.number_of_words_on_each_side:
        #                         occurances_dict[query_word][i] += 1


        # best_occurrences = {query_word:None for query_word in occurances_dict if query_word not in not_exist_words}

        # for query_word in occurances_dict:
        #     for i in occurances_dict[query_word]:
        #         if best_occurrences[query_word] == None or occurances_dict[query_word][i] > occurances_dict[query_word][best_occurrences[query_word]]:
        #             best_occurrences[query_word] = i
    

        # for query_word, occurance in best_occurrences.items():
        #     start = max(0, occurance - self.number_of_words_on_each_side)
        #     end = min(len(doc), occurance + self.number_of_words_on_each_side + 1)
        #     snippet = ' '.join(doc[start:end])
        #     query_w = doc[occurance]
        #     query_w = re.sub(r'[^\w\s]', '', query_w)
        #     snippet = snippet.replace(query_w, f'***{query_w}***')
        #     final_snippet += snippet + ' ... '

        # print(final_snippet[:-5])
        # final_snippet = final_snippet[:-5]
        #-------------------------------------------------------------------------------------------
        


        ##### My comment: The code below highloghts best occurance of a query word assuming everything is lowercased.
        #-------------------------------------------------------------------------------------------
        # doc = doc.split()
        # query = self.remove_stop_words_from_query(query)
        # query = query.split()
        # query = [word.lower() for word in query]
        # occurances_dict = {query_word:{} for query_word in query}

        # for i in range(len(doc)):
        #     temp = re.sub(r'()",', '', doc[i])
        #     if temp in query:
        #         occurances_dict[temp][i] = 0

        # for query_word in query:
        #     if len(occurances_dict[query_word]) == 0:
        #         not_exist_words.append(query_word)


        # for query_word in occurances_dict:
        #     for i in occurances_dict[query_word]:
        #         for other_query_word in occurances_dict:
        #             if other_query_word != query_word:
        #                 for j in occurances_dict[other_query_word]:
        #                     if abs(i-j) <= self.number_of_words_on_each_side:
        #                         occurances_dict[query_word][i] += 1


        # best_occurrences = {query_word:None for query_word in occurances_dict if query_word not in not_exist_words}

        # for query_word in occurances_dict:
        #     for i in occurances_dict[query_word]:
        #         if best_occurrences[query_word] == None or occurances_dict[query_word][i] > occurances_dict[query_word][best_occurrences[query_word]]:
        #             best_occurrences[query_word] = i
    

        # for query_word, occurance in best_occurrences.items():
        #     start = max(0, occurance - self.number_of_words_on_each_side)
        #     end = min(len(doc), occurance + self.number_of_words_on_each_side + 1)
        #     snippet = ' '.join(doc[start:end])
        #     query_w = doc[occurance]
        #     query_w = re.sub(r'[^\w\s]', '', query_w)
        #     snippet = snippet.replace(query_w, f' ***{query_w}*** ')
        #     final_snippet += snippet + ' ... '
        #-------------------------------------------------------------------------------------------
        
        doc = doc.split()
        query = self.remove_stop_words_from_query(query)
        query = query.split()
        # remove punctuation marks from the query
        query = [re.sub(r'[^\w\s]', '', word) for word in query]
        query = [word.lower() for word in query]
        occurances_dict = {query_word:{} for query_word in query}

        for i in range(len(doc)):
            temp = re.sub(r'[^\w\s]', '', doc[i])
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
                best_occurrences[query_word] = max(occurances_dict[query_word], key=occurances_dict[query_word].get)
        
        all_occurances = []
        for query_word in best_occurrences:
            if best_occurrences[query_word] != None:
                all_occurances.append(best_occurrences[query_word])
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
        final_snippet.replace('  ',' ') 
        
        return final_snippet, not_exist_words
