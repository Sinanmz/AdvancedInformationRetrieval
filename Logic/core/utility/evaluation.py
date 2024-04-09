import os
import sys
current_script_path = os.path.abspath(__file__)
utility_dir = os.path.dirname(current_script_path)
core_dir = os.path.dirname(utility_dir)
Logic_dir = os.path.dirname(core_dir)
project_root = os.path.dirname(Logic_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from Logic import utils

from typing import List
import numpy as np
import wandb




class Evaluation:

    def __init__(self, name: str):
            self.name = name

    def calculate_precision(self, actual: List[List[tuple]], predicted: List[List[str]]) -> float:
        """
        Calculates the precision of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The precision of the predicted results
        """
        precision = 0.0
        all_precisions = []

        # TODO: Calculate precision here
        acual_ids = []
        for i in range(len(actual)):
            acual_ids.append([x[0] for x in actual[i]])

        for i in range(len(predicted)):
            precision += len(set(predicted[i]).intersection(set(acual_ids[i])))/len(predicted[i])
            all_precisions.append(len(set(predicted[i]).intersection(set(acual_ids[i])))/len(predicted[i]))
        precision /= len(predicted)

        return all_precisions, precision
    
    def calculate_recall(self, actual: List[List[tuple]], predicted: List[List[str]]) -> float:
        """
        Calculates the recall of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The recall of the predicted results
        """
        recall = 0.0
        all_recalls = []

        # TODO: Calculate recall here
        acual_ids = []
        for i in range(len(actual)):
            acual_ids.append([x[0] for x in actual[i]])

        for i in range(len(predicted)):
            recall += len(set(predicted[i]).intersection(set(acual_ids[i])))/len(acual_ids[i])
            all_recalls.append(len(set(predicted[i]).intersection(set(acual_ids[i])))/len(acual_ids[i]))
        recall /= len(predicted)

        return all_recalls, recall
    
    def calculate_F1(self, actual: List[List[tuple]], predicted: List[List[str]]) -> float:
        """
        Calculates the F1 score of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The F1 score of the predicted results    
        """
        f1 = 0.0
        all_f1 = []

        # TODO: Calculate F1 here
        acual_ids = []
        for i in range(len(actual)):
            acual_ids.append([x[0] for x in actual[i]])

        for i in range(len(predicted)):
            precision = len(set(predicted[i]).intersection(set(acual_ids[i])))/len(predicted[i])
            recall = len(set(predicted[i]).intersection(set(acual_ids[i])))/len(acual_ids[i])
            f1 += 2*precision*recall/(precision+recall)
            all_f1.append(2*precision*recall/(precision+recall))
        f1 /= len(predicted)

        return all_f1, f1
    
    def calculate_AP(self, actual: List[List[tuple]], predicted: List[List[str]]) -> float:
        """
        Calculates the Mean Average Precision of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The Average Precision of the predicted results
        """
        AP = 0.0
        all_AP = []

        # TODO: Calculate AP here
        acual_ids = []
        for i in range(len(actual)):
            acual_ids.append([x[0] for x in actual[i]])

        for i in range(len(predicted)):
            correct = 0
            precision = 0
            for j in range(len(predicted[i])):
                if predicted[i][j] in acual_ids[i]:
                    correct += 1
                    precision += correct/(j+1)
            AP += precision/correct
            all_AP.append(precision/correct)

        AP /= len(predicted)

        return all_AP, AP
    
    def calculate_MAP(self, actual: List[List[tuple]], predicted: List[List[str]]) -> float:
        """
        Calculates the Mean Average Precision of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The Mean Average Precision of the predicted results
        """
        MAP = 0.0

        # TODO: Calculate MAP here
        # acual_ids = []
        # for i in range(len(actual)):
        #     acual_ids.append([x[0] for x in actual[i]])

        # for i in range(len(predicted)):
        #     correct = 0
        #     precision = 0.0
        #     for j in range(len(predicted[i])):
        #         if predicted[i][j] in acual_ids[i]:
        #             correct += 1
        #             precision += correct/(j+1)
        #     if correct > 0:
        #         MAP += precision/correct
        
        # MAP /= len(predicted)
        AP = self.calculate_AP(actual, predicted)[0]
        MAP = sum(AP)/len(AP)
            
        return MAP
    
    def cacluate_DCG(self, actual: List[List[tuple]], predicted: List[List[str]]) -> float:
        """
        Calculates the Normalized Discounted Cumulative Gain (NDCG) of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The DCG of the predicted results
        """
        DCG = 0.0
        all_DCG = []

        # TODO: Calculate DCG here
        for i in range(len(predicted)):
            temp = 0.0
            for j in range(len(predicted[i])):
                for k in range(len(actual[i])):
                    if predicted[i][j] == actual[i][k][0]:
                        temp += (2**actual[i][k][1])*(1/np.log2(j+2))
                        break
            all_DCG.append(temp)
            DCG += temp

        DCG /= len(predicted)
            
        return all_DCG, DCG
    
    def cacluate_NDCG(self, actual: List[List[tuple]], predicted: List[List[str]]) -> float:
        """
        Calculates the Normalized Discounted Cumulative Gain (NDCG) of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The NDCG of the predicted results
        """
        NDCG = 0.0
        all_NDCG = []

        # TODO: Calculate NDCG here
        valid_queries = 0
        for i in range(len(predicted)):
            DCG = 0.0
            for j in range(len(predicted[i])):
                for k in range(len(actual[i])):
                    if predicted[i][j] == actual[i][k][0]:
                        DCG += (2**actual[i][k][1])*(1/np.log2(j+2))
                        break
        
            sorted_points = sorted(actual[i], key=lambda x: x[1], reverse=True)
            ideal_DCG = 0.0
            for j in range(len(predicted[i])):
                    ideal_DCG += (2**sorted_points[j][1])*(1/np.log2(j+2))
            
            if ideal_DCG > 0:
                NDCG += DCG/ideal_DCG
                all_NDCG.append(DCG/ideal_DCG)
                valid_queries += 1

        NDCG /= valid_queries

        return all_NDCG, NDCG
    
    def cacluate_RR(self, actual: List[List[tuple]], predicted: List[List[str]]) -> float:
        """
        Calculates the Mean Reciprocal Rank of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The Reciprocal Rank of the predicted results
        """
        RR = 0.0
        all_RR = []

        # TODO: Calculate MRR here
        acual_ids = []
        for i in range(len(actual)):
            acual_ids.append([x[0] for x in actual[i]])

        for i in range(len(predicted)):
            for j in range(len(predicted[i])):
                if predicted[i][j] in acual_ids[i]:
                    RR += 1/(j+1)
                    all_RR.append(1/(j+1))
                    break
        RR /= len(predicted)

        return all_RR, RR
    
    def cacluate_MRR(self, actual: List[List[tuple]], predicted: List[List[str]]) -> float:
        """
        Calculates the Mean Reciprocal Rank of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The MRR of the predicted results
        """
        MRR = 0.0

        # TODO: Calculate MRR here
        RR = self.cacluate_RR(actual, predicted)[0]
        MRR = sum(RR)/len(RR)
        
        return MRR
    

    def print_evaluation(self, all_precissions, all_recalls, all_f1s, ap, map_score, all_dcgs, all_ndcgs, rr, mrr, queries=None):
        """
        Prints the evaluation metrics

        parameters
        ----------
        precision : float
            The precision of the predicted results
        recall : float
            The recall of the predicted results
        f1 : float
            The F1 score of the predicted results
        ap : float
            The Average Precision of the predicted results
        map : float
            The Mean Average Precision of the predicted results
        dcg: float
            The Discounted Cumulative Gain of the predicted results
        ndcg : float
            The Normalized Discounted Cumulative Gain of the predicted results
        rr: float
            The Reciprocal Rank of the predicted results
        mrr : float
            The Mean Reciprocal Rank of the predicted results
            
        """
        print(f"Name: {self.name}\n")

        if queries is not None:
            formatted_queries = ' - '.join(queries)
            print(f"Queries: {formatted_queries}\n")

        #TODO: Print the evaluation metrics
        print("Evaluation Metrics:")
        print("-" * 20)

        metrics = {
            "All Precisions": all_precissions,
            "Mean Precision": sum(all_precissions) / len(all_precissions) if all_precissions else 0,
            "All Recalls": all_recalls,
            "Mean Recall": sum(all_recalls) / len(all_recalls) if all_recalls else 0,
            "All F1s": all_f1s,
            "Mean F1": sum(all_f1s) / len(all_f1s) if all_f1s else 0,
            "Average Precision (AP)": ap,
            "Mean Average Precision (MAP)": map_score,
            "All Discounted Cumulative Gains (DCG)": all_dcgs,
            "Mean DCG": sum(all_dcgs) / len(all_dcgs) if all_dcgs else 0,
            "All Normalized Discounted Cumulative Gains (NDCG)": all_ndcgs,
            "Mean NDCG": sum(all_ndcgs) / len(all_ndcgs) if all_ndcgs else 0,
            "All Reciprocal Ranks (RR)": rr,
            "Mean Reciprocal Rank (MRR)": mrr,
        }

        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, list):
                formatted_values = ', '.join(f"{v:.4f}" for v in metric_value)
                print(f"{metric_name}: [{formatted_values}]")
            else:
                print(f"{metric_name}: {metric_value:.4f}")

        print("-" * 20, "\n")




        


    def log_evaluation(self, precision, recall, f1, map, dcg, ndcg, mrr):
        """
        Use Wandb to log the evaluation metrics
      
        parameters
        ----------
        precision : float
            The precision of the predicted results
        recall : float
            The recall of the predicted results
        f1 : float
            The F1 score of the predicted results
        ap : float
            The Average Precision of the predicted results
        map : float
            The Mean Average Precision of the predicted results
        dcg: float
            The Discounted Cumulative Gain of the predicted results
        ndcg : float
            The Normalized Discounted Cumulative Gain of the predicted results
        rr: float
            The Reciprocal Rank of the predicted results
        mrr : float
            The Mean Reciprocal Rank of the predicted results
            
        """
        
        #TODO: Log the evaluation metrics using Wandb
        wandb.login()
        wandb.init(project='MIR_IMDB_Retrieval')
        wandb.log({
            'Mean Precision': precision,
            'Mean Recall': recall,
            'Mean F1': f1,
            'Mean Average Precision': map,
            'Mean Discounted Cumulative Gain': dcg,
            'Mean Normalized Discounted Cumulative Gain': ndcg,
            'Mean Reciprocal Rank': mrr
        })
            


    def calculate_evaluation(self, actual: List[List[tuple]], predicted: List[List[str]], queries=None):
        """
        call all functions to calculate evaluation metrics

        parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results
            
        """

        all_precissions, mean_precision = self.calculate_precision(actual, predicted)
        all_recalls, mean_recall = self.calculate_recall(actual, predicted)
        all_f1s, mean_f1 = self.calculate_F1(actual, predicted)
        ap = self.calculate_AP(actual, predicted)[0]
        map_score = self.calculate_MAP(actual, predicted)
        all_dcgs, mean_dcg= self.cacluate_DCG(actual, predicted)
        all_ndcgs, mean_ndcg = self.cacluate_NDCG(actual, predicted)
        rr = self.cacluate_RR(actual, predicted)[0]
        mrr = self.cacluate_MRR(actual, predicted)

        #call print and viualize functions
        self.print_evaluation(all_precissions, all_recalls, all_f1s, ap, map_score, all_dcgs, all_ndcgs, rr, mrr, queries)
        self.log_evaluation(mean_precision, mean_recall, mean_f1, map_score, mean_dcg, mean_ndcg, mrr)



if __name__ == '__main__':
    queries = ['dune', 'harry potter', 'spiderman', 'matrix', 'batman']



    actual = [[('tt15239678', 20), ('tt0087182', 19), ('tt1160419', 18), ('tt0142032', 17), ('tt31378509', 16), 
               ('tt0287839', 15), ('tt10466872', 14), ('tt1935156', 13), ('tt15331462', 12), ('tt11835714', 11), 
               ('tt12451788', 10), ('tt14450978', 9), ('tt31613341', 8), ('tt0099474', 7), ('tt31613353', 6)], 

               [('tt0241527', 20), ('tt0330373', 19), ('tt0304141', 18), ('tt0295297', 17), ('tt1201607', 16), 
                ('tt0373889', 15), ('tt0417741', 14), ('tt0926084', 13), ('tt13918446', 12), ('tt16116174', 11), 
                ('tt1756545', 10), ('tt15431326', 9), ('tt3731688', 8), ('tt2335590', 7), ('tt7467820', 6)], 

                [("tt0145487", 20), ("tt10872600", 19), ("tt0948470", 18), ("tt1872181", 17), ("tt2705436", 16), 
                 ("tt0112175", 15), ("tt12122034", 14), ("tt0413300", 13), ("tt4633694", 12), ("tt2250912", 11), 
                 ("tt6320628", 10), ("tt9362722", 9), ("tt0316654", 8), ("tt0076975", 7), ("tt16360004",6)],

                 [("tt0133093", 20), ("tt10838180", 19), ("tt0234215", 18), ("tt0242653", 17), ("tt0106062", 16), 
                  ("tt30849138", 15), ("tt0410519", 14), ("tt9847360", 13), ("tt31998838", 12), ("tt30749809", 11), 
                  ("tt0365467", 10), ("tt0364888", 9), ("tt11749868", 8), ("tt0303678", 7), ("tt0274085", 6)], 

                  [("tt0096895", 20), ("tt1877830", 19), ("tt0059968", 18), ("tt0372784", 17), ("tt0103359", 16), 
                   ("tt0118688", 15), ("tt0103776", 14), ("tt0112462", 13), ("tt2975590", 12), ("tt19850008", 11), 
                   ("tt0147746", 10), ("tt0398417", 9), ("tt0035665", 8), ("tt4116284", 7), ("tt0060153", 6)]
                  ]
    
    methods = ['ltn.lnn', 'ltc.lnc', 'OkapiBM25']
    predicted = {method: [] for method in methods}

    for query in queries:
        for method in methods:
            search_term = query
            search_max_num = 10
            search_weights = [1, 1, 1]
            result = utils.search(
                        search_term,
                        search_max_num,
                        method,
                        search_weights,
                    )
            query_predicted = []
            for res in result:
                query_predicted.append(res[0])
            predicted[method].append(query_predicted)

    for method in methods:
        evaluation = Evaluation(method)
        evaluation.calculate_evaluation(actual, predicted[method], queries)