import numpy as np
import os
import sys
import pprint
from collections import Counter
import json
from dataset_manager.dataset_analysis import DatasetAnalysis
if sys.version_info[0] == 3:
    from score_manager.error_analysis import ErrorAnalysis

class ScoreManager():
    def __init__(self,):
        self.result_folder = "results/"
        self.datasets_folder = "datasets/"
        self.performances_folder = "performances/"

    def save_results(self, representation, dataset_name, model_name, file_name, true_labels, predictions, fold = None):
        if fold != None:
            path = self.result_folder + model_name + "_" + representation + "_" + dataset_name +  "/" + fold + "/"
        else:
            path = self.result_folder + model_name + "_" + representation + "_" + dataset_name + "/"

        if not os.path.exists(path):
            os.makedirs(path)

        results = {}
        results["ground true"] = true_labels
        results["predictions"] = predictions
        if ".json" not in file_name:
            file_name += ".json"

        with open(path + file_name, "w") as f:
            f.write(json.dumps(results, indent=4))
    def save_performances(self, representation, dataset_name, model_name, file_name, accuracy_train, accuracy_valid, f1s_train, f1s_valid,
        loss_train, loss_valid, fold=None):
        if fold != None:
            path = self.performances_folder + model_name + "_" + representation + "_" + dataset_name +  "/" + fold + "/"
        else:
            path = self.performances_folder + model_name + "_" + representation + "_" + dataset_name + "/"

        if not os.path.exists(path):
            os.makedirs(path)
        results = {}
        results["accuracy_train"] = accuracy_train
        results["accuracy_valid"] = accuracy_valid
        results["f1_train"] = f1s_train
        results["f1_valid"] = f1s_valid
        results["loss_train"] = loss_train
        results["loss_valid"] = loss_valid
        with open(path + file_name, "w") as f:
            f.write(json.dumps(results, indent=4))
    def __process_dataset_name(self, representation, dataset_path, model_name):
        dataset_name = dataset_path.split(".")[0].split("/")[-1]
        data_path = ""
        if ".json" not in dataset_path:
            data_path = dataset_path + ".json"
        else:
            data_path = dataset_path
        if self.datasets_folder not in dataset_path:
            data_path = self.datasets_folder + data_path
        result_name = "_".join([model_name, representation, dataset_name])
        route_path = self.result_folder + result_name
        return dataset_name, route_path, data_path



    def compute_scores(self, representation, dataset_path, model_name):
        dataset_name, route_path, data_path = self.__process_dataset_name(representation, dataset_path, model_name)
        eval = ErrorAnalysis(data_path, representation)
        best_accuracies = []
        best_f1s = []
        results = {}
        best_file_f1 = ""
        best_file_acc = ""
        best_accuracy = 0
        best_f1 = 0
        for folder in sorted(os.listdir(route_path)):
            for file in os.listdir(route_path + "/" + folder):
                print(file.upper())
                code = eval.evaluation(route_path + "/" + folder + "/" + file)
                accuracy, f1 = eval.compute_scores(code)
                print("Micro F1: ", accuracy)
                print("Macro F1: ", f1)
                if accuracy > best_accuracy:
                    path_acc = route_path + "/" + folder + "/" + file
                    best_file1 = file
                    best_accuracy = accuracy
                if f1 > best_f1:
                    path_f1 = route_path + "/" + folder + "/" + file
                    best_file2 = file
                    best_f1 = f1
                    #print(file, " Accuracy: ", accuracy)
                best_accuracies.append(best_accuracy)
                best_f1s.append(best_f1)

        print("Best experiment: ", best_file_acc, " Micro F1: ", best_accuracy)
        print("Best experiment: ", best_file_f1, " MAcro F1: ", best_f1)

        return eval, path_f1, path_acc

    
