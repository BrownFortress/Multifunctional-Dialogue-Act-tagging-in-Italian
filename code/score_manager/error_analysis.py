import re
import operator
import numpy as np
import pickle
import os
from collections import Counter
import sys
import json
from dataset_manager.data_preprocessing import DataPreprocessing
from dataset_manager.dataset_analysis import DatasetAnalysis
import plotly.graph_objects as go
import pprint

class ErrorAnalysis():
    def __init__(self, dataset_path=None, representation=None ):
        self.labels = None
        self.dataset_name = None
        self.labels_in_test_set = None
        print(dataset_path)
        if dataset_path != None and representation != None:
            if "/" in dataset_path:
                self.dataset_name = dataset_path.split("/")[-1]
            else:
                self.dataset_name = dataset_path

    def load_data(self, data_path):
        with open(data_path, "r") as f:
            data = json.loads(f.read())
        true_labels = data["ground true"]
        predictions = data["predictions"]
        if type(true_labels[0]) == list:
            tmp_gr_true = []
            tmp_predictions =[]
            for seq_id, seq in enumerate(true_labels):
                tmp_gr_true.extend(["_".join(lab.split("_")[1:]) for lab in seq])
                tmp_predictions.extend(["_".join(lab.split("_")[1:]) for lab in predictions[seq_id]])
            self.labels = list(set(tmp_gr_true + tmp_predictions))
        else:
            self.labels = list(set(data["ground true"] + data["predictions"]))

        return true_labels, predictions

    def get_the_percentage(self, data_to_percentage, all_data, cut_off=0):
        common_divisor = sum(Counter(all_data).values())
        dtp = data_to_percentage
        if type(data_to_percentage) != (dict):
            dtp = Counter(data_to_percentage)
        res = {}
        for k,v in dtp.items():
            if (v / float(common_divisor) * 100) > cut_off:
                res[k] = v / float(common_divisor) * 100
        return res


    def evaluation(self, data_path):
        # This fucntion builds a confusion matrix
        if type(data_path) == str:
            true_labels, predictions = self.load_data(data_path)
        elif type(data_path) == list:
            true_labels = []
            predictions = []
            for element in data_path:
                t, p = element.split(" ")
                true_labels.append(t)
                predictions.append(p)
        else:
            true_labels = data_path["ground_true"]
            predictions = data_path["predictions"]

            #print(len(self.labels))
            used = []
            unique_predictions = [x for x in predictions if x not in used and (used.append(x) or True)]
            #print(len(unique_predictions))
            used = []
            unique_true_labels = [x for x in true_labels if x not in used and (used.append(x) or True)]
            self.labels = unique_true_labels + unique_predictions


        evaluations = {}
        self.labels_in_test_set = self.labels
        labels = self.labels
        for l in labels:
            if l not in evaluations.keys():
                evaluations[l] = {}
                for l2 in labels:
                    if l2 not in evaluations[l]:
                        evaluations[l][l2] = 0
        for id, t_l in enumerate(true_labels):
            if t_l not in evaluations.keys():
                evaluations[t_l] = {}
            if predictions[id] not in evaluations[t_l].keys():
                evaluations[t_l][predictions[id]] = 0
            evaluations[t_l][predictions[id]] += 1
        return evaluations # confusion matrix

    def compute_scores(self, confusion_matrix):
        #Columns are true labels while row are prediction
        matrix_number  = []
        labels = []
        for k, v in confusion_matrix.items():
            matrix_number.append(list(v.values()))
            labels.append(k)
        #print(matrix_number)
        m = np.asarray(matrix_number).transpose()
        accuracy = np.diagonal(m).sum() / float(m.sum())
        precisions = []
        recalls = []
        diagonal = float(np.diagonal(m).sum()) # TP


        for id, row in enumerate(m):
            if row.sum() != 0:
                precisions.append(row[id] / float(row.sum()))
            elif labels[id]:
                precisions.append(0)
        for id, row in enumerate(m.transpose()):
            if row.sum() != 0:
                recalls.append(row[id] / float(row.sum()))
            elif labels[id]:
                recalls.append(0)
        f1s = []
        for id, p in enumerate(precisions):
            if p != 0 or recalls[id] != 0:
                f1 = (2*recalls[id] * p) / float(recalls[id] + p)
                f1s.append(f1)
            else:
                f1s.append(0)
        penalty = 0
        if "segmentation-error" in labels:
            penalty = 1
        f1 = sum(f1s) / (len(f1s) - penalty)
        return accuracy, f1

    def score_by_classes(self, confusion_matrix):
        matrix_number  = []
        xy_labels = []
        for k, v in confusion_matrix.items():
            matrix_number.append(list(v.values()))
            xy_labels.append(k)

        m = np.asarray(matrix_number).transpose()
        accuracy = np.diagonal(m).sum() / float(m.sum())
        precisions = []
        recalls = []
        f1s= []
        precision_classes = {}
        recall_classes = {}
        f1_classes = {}
        diagonal = float(np.diagonal(m).sum())
        for id, row in enumerate(m):
            if row.sum() != 0 :
                precisions.append(row[id] / float(row.sum()))
            else:
                precisions.append(0)
            precision_classes[xy_labels[id]] = precisions[-1]
        for id, row in enumerate(m.transpose()):
            if row.sum() != 0:
                recalls.append(row[id] / float(row.sum()))
            else:
                recalls.append(0)
            recall_classes[xy_labels[id]] = recalls[-1]

        for id, p in enumerate(precisions):
            if p != 0 or recalls[id] != 0:
                f1 = (2*recalls[id] * p) / float(recalls[id] + p)
                f1s.append(f1)
            else:
                f1s.append(0)
            f1_classes[xy_labels[id]] = f1s[-1]

        return f1_classes, recall_classes, precision_classes


    
