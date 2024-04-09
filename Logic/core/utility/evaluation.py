import os
import sys
current_script_path = os.path.abspath(__file__)
utility_dir = os.path.dirname(current_script_path)
core_dir = os.path.dirname(utility_dir)
Logic_dir = os.path.dirname(core_dir)
project_root = os.path.dirname(Logic_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from typing import List
import numpy as np
import wandb


class Evaluation:

    def __init__(self, name: str):
            self.name = name

    def calculate_precision(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
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

        # TODO: Calculate precision here
        for i in range(len(predicted)):
            precision += len(set(predicted[i]).intersection(set(actual[i])))/len(predicted[i])
        precision /= len(predicted)

        return precision
    
    def calculate_recall(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
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

        # TODO: Calculate recall here
        for i in range(len(predicted)):
            recall += len(set(predicted[i]).intersection(set(actual[i])))/len(actual[i])
        recall /= len(predicted)

        return recall
    
    def calculate_F1(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
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

        # TODO: Calculate F1 here
        for i in range(len(predicted)):
            precision = len(set(predicted[i]).intersection(set(actual[i])))/len(predicted[i])
            recall = len(set(predicted[i]).intersection(set(actual[i])))/len(actual[i])
            f1 += 2*precision*recall/(precision+recall)
        f1 /= len(predicted)

        return f1
    
    def calculate_AP(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
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

        # TODO: Calculate AP here
        for i in range(len(predicted)):
            correct = 0
            precision = 0
            for j in range(len(predicted[i])):
                if predicted[i][j] in actual[i]:
                    correct += 1
                    precision += correct/(j+1)
            AP += precision/correct
        AP /= len(predicted)

        return AP
    
    def calculate_MAP(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
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
        for i in range(len(predicted)):
            correct = 0
            precision = 0.0
            for j in range(len(predicted[i])):
                if predicted[i][j] in actual[i]:
                    correct += 1
                    precision += correct/(j+1)
            if correct > 0:
                MAP += precision/correct
            
        return MAP
    
    def cacluate_DCG(self, actual: List[List[(str, int)]], predicted: List[List[str]]) -> float:
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

        # TODO: Calculate DCG here
        for i in range(len(predicted)):
            for j in range(len(predicted[i])):
                for k in range(len(actual[i])):
                    if predicted[i][j] == actual[i][k]:
                        DCG += (2**actual[i][k][1])*(1/np.log2(j+2))
                        break

        DCG /= len(predicted)
            
        return DCG
    
    def cacluate_NDCG(self, actual: List[List[(str, int)]], predicted: List[List[str]]) -> float:
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

        # TODO: Calculate NDCG here
        valid_queries = 0
        for i in range(len(predicted)):
            DCG = 0.0
            for j in range(len(predicted[i])):
                for k in range(len(actual[i])):
                    if predicted[i][j] == actual[i][k]:
                        DCG += (2**actual[i][k][1])*(1/np.log2(j+2))
                        break

        
            sorted_points = sorted(actual[i], key=lambda x: x[1], reverse=True)
            ideal_DCG = 0.0
            for j in range(len(predicted[i])):
                    ideal_DCG += (2**sorted_points[j][1])*(1/np.log2(j+2))
            
            if ideal_DCG > 0:
                NDCG += DCG/ideal_DCG
                valid_queries += 1

        NDCG /= valid_queries

        return NDCG
    
    def cacluate_RR(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
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

        # TODO: Calculate MRR here
        for i in range(len(predicted)):
            for j in range(len(predicted[i])):
                if predicted[i][j] in actual[i]:
                    RR += 1/(j+1)
                    break
        RR /= len(predicted)

        return RR
    
    def cacluate_MRR(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
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
        for i in range(len(predicted)):
            for j in range(len(predicted[i])):
                if predicted[i][j] in actual[i]:
                    MRR += 1/(j+1)
                    break
        MRR /= len(predicted)

        return MRR
    

    def print_evaluation(self, precision, recall, f1, ap, map, dcg, ndcg, rr, mrr):
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
        print(f"name = {self.name}")

        #TODO: Print the evaluation metrics
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1 Score: {f1}')
        print(f'Average Precision (AP): {ap}')
        print(f'Mean Average Precision (MAP): {map}')
        print(f'Discounted Cumulative Gain (DCG): {dcg}')
        print(f'Normalized Discounted Cumulative Gain (NDCG): {ndcg}')
        print(f'Reciprocal Rank (RR): {rr}')
        print(f'Mean Reciprocal Rank (MRR): {mrr}')


    def log_evaluation(self, precision, recall, f1, ap, map, dcg, ndcg, rr, mrr):
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
        wandb.init(project='MIR_IMDB_Retrieval', entity='sinanmz')
        wandb.log({
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'Average Precision (AP)': ap,
            'Mean Average Precision (MAP)': map,
            'Discounted Cumulative Gain (DCG)': dcg,
            'Normalized Discounted Cumulative Gain (NDCG)': ndcg,
            'Reciprocal Rank (RR)': rr,
            'Mean Reciprocal Rank (MRR)': mrr
        })


    def calculate_evaluation(self, actual: List[List[str]], predicted: List[List[str]]):
        """
        call all functions to calculate evaluation metrics

        parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results
            
        """

        precision = self.calculate_precision(actual, predicted)
        recall = self.calculate_recall(actual, predicted)
        f1 = self.calculate_F1(actual, predicted)
        ap = self.calculate_AP(actual, predicted)
        map_score = self.calculate_MAP(actual, predicted)
        dcg = self.cacluate_DCG(actual, predicted)
        ndcg = self.cacluate_NDCG(actual, predicted)
        rr = self.cacluate_RR(actual, predicted)
        mrr = self.cacluate_MRR(actual, predicted)

        #call print and viualize functions
        self.print_evaluation(precision, recall, f1, ap, map_score, dcg, ndcg, rr, mrr)
        self.log_evaluation(precision, recall, f1, ap, map_score, dcg, ndcg, rr, mrr)



