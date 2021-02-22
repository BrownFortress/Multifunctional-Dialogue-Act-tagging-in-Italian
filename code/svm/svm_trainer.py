from score_manager.score_manager import ScoreManager
from dataset_manager.data_preprocessing import DataPreprocessing
from dataset_manager.dataset_manager import DatasetManager
import numpy as np
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.model_selection import learning_curve
from sklearn.model_selection import KFold, cross_val_score, cross_validate
from sklearn import preprocessing
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import *
import random
import pickle
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from tqdm import tqdm
from os import path
import pprint
import sys
import os
import warnings
import spacy


if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses
class SVM_trainer():
    def __init__(self):
        '''
            data_represetnation:{
                "utterances":{
                    1: [[Hello], [Hi], [How are you] ...], # each line is a dialogue: the same schema is used for every key of d_rep
                    2: ...,
                },
                "we":{ # "word embedding"
                    1: ...,
                }
                "DA":{}
                "hot_DA":{}
                "prev_DA":{}
                "hot_prev_da":{}
                "pos_tags":{}id
                "hot_pos_tags":{}
                "dep_tags":{}
                "hot_dep_tags":{}
            }
        '''
        self.dataset_folder = "preprocessing/"
        self.standard_split_dataset_folder = "standard_splits/"
    def remove_speaker_turns(self, split, split_ids, dataset_rep, speaker_to_keep):
        print("Removing speaker ",("USER" if speaker_to_keep == "S" else "SYSTEM")," turns")
        id_ba = 0
        to_del = []
        # Remove speaker examples from training set
        for d_id in split_ids["train_set"]:
            for s in dataset_rep["speakers"][d_id]:
                if s != speaker_to_keep:
                    to_del.append(id_ba)
                id_ba += 1
        for i in sorted(to_del, reverse=True):
            del split["train_set"]["examples"][i]
            del split["train_set"]["labels"][i]

        # Remove system examples from test set
        id_ba = 0
        to_del = []
        for d_id in split_ids["test_set"]:
            for s in dataset_rep["speakers"][d_id]:
                if s != speaker_to_keep:
                    to_del.append(id_ba)
                id_ba += 1

        for i in sorted(to_del, reverse=True):
            del split["test_set"]["examples"][i]
            del split["test_set"]["labels"][i]

    def hot_encode(self, to_convert, cipher):
        code = len(cipher.keys()) * [0]
        if type(to_convert) == list:
            for element in to_convert:
                code[cipher[element]] += 1
        else:
            code[cipher[to_convert]] += 1
        return code

    def test_iteration(self, dataset, representation, model_system, model_user,
                       speaker_to_keep="U"):
        dataset_name = dataset.split(".")[0].split("/")[-1]
        norm_selection = [False, True]
        dataset_rep = DataPreprocessing().preprocess(dataset, representation)
        split_ids = DatasetManager().get_official_split_ids(dataset_name, dataset_rep["we"])
        split = DatasetManager().official_split(dataset_name, dataset_rep["we"], dataset_rep["enumerated_DA"])
        true_labels = []
        predictions = []
        user_utterances = []
        for d_id, rep in dataset_rep["we"].items():
            prev_DA = "SOD"
            if d_id in split_ids["test_set"]:
                for turn_id, we in enumerate(rep):
                    feature = {}
                    if dataset_rep["speakers"][d_id][turn_id] == "S":
                        feature["we"] = we
                        feature["utterance"] = dataset_rep["utterances"][d_id][turn_id]
                        feature["pos_tags"] = dataset_rep["hot_pos_tags"][d_id][turn_id]
                        feature["dep_tags"] = dataset_rep["hot_dep_tags"][d_id][turn_id]
                        feature["context"] = dataset_rep["context"][d_id][turn_id]
                        feature["da_tags"] = self.hot_encode(prev_DA, model_user["da_cipher"])
                        feature_ready = self.features_building(feature,
                                            prev_da=model_system["settings"][0],
                                            pos_tag=model_system["settings"][1],
                                            dep_tag=model_system["settings"][2],
                                            context_flag=model_system["settings"][3])

                        if type(model_system["mean"]) != int:
                            feature_ready = self.normalize_features(feature_ready, model_system["mean"],
                                                                    model_system["std"])

                        pred = model_system["model"].predict(feature_ready)
                        prev_DA = model_system["number_to_label"][pred[0]]
                    else:
                        feature["we"] = we
                        user_utterances.append(dataset_rep["utterances"][d_id][turn_id])
                        feature["utterance"] = dataset_rep["utterances"][d_id][turn_id]
                        feature["pos_tags"] = dataset_rep["hot_pos_tags"][d_id][turn_id]
                        feature["dep_tags"] = dataset_rep["hot_dep_tags"][d_id][turn_id]
                        feature["context"] = dataset_rep["context"][d_id][turn_id]
                        feature["da_tags"] = self.hot_encode(prev_DA, model_user["da_cipher"])
                        feature_ready = self.features_building(feature,
                                            prev_da=model_user["settings"][0],
                                            pos_tag=model_user["settings"][1],
                                            dep_tag=model_user["settings"][2],
                                            context_flag=model_user["settings"][3])

                        if type(model_user["mean"]) != int:
                            feature_ready = self.normalize_features(feature_ready, model_user["mean"],
                                                                    model_user["std"])

                        pred = model_user["model"].predict(feature_ready)
                        prediction = model_user["number_to_label"][pred[0]]
                        true_labels.append(dataset_rep["DA"][d_id][turn_id])
                        predictions.append(prediction)
                        prev_DA = prediction
            ScoreManager().save_results(representation, dataset_name, "svm",
                 "final",true_labels, predictions, fold="/")
            new_labels = [(x, predictions[ids]) for ids, x in enumerate(true_labels)]
            '''
            ScoreManager().save_results(representation, dataset_name, "svm",
                 "error",user_utterances, new_labels, fold="Error")
            '''
    def train_iteration(self, dataset, representation, c_values, test_settings, speaker_to_keep="U", split_ids=None):
        dataset_name = dataset.split(".")[0].split("/")[-1]
        norm_selection = [False, True]

        dataset_rep = DataPreprocessing().preprocess(dataset, representation)
        models_f1s = []
        model_names = []
        for norm_flag in norm_selection:
            if norm_flag:
                string1 = "norm"
            else:
                string1 = "not_norm"

            for id, settings in enumerate(test_settings["settings"]):
                # Normalized data
                features = self.features_building(dataset_rep,
                    prev_da=settings[0],
                    pos_tag=settings[1],
                    dep_tag=settings[2],
                    context_flag=settings[3]
                    )

                if norm_flag:
                    features, _, _ = self.normalization(features)

                split = DatasetManager().official_split(dataset_name, features, dataset_rep["enumerated_DA"])
                split_ids = DatasetManager().get_official_split_ids(dataset_name, features)

                print("Features :", test_settings["name"][id])
                print("Features vector dimension :", len(split["train_set"]["examples"][0]))

                print("Selecting C")

                if "ilisten" in dataset_name.lower():
                    self.remove_speaker_turns(split, split_ids, dataset_rep, speaker_to_keep)
                    print("Done")
                micro_f1_score = []
                precision_score = []
                print("Training...")
                for c in tqdm(c_values):
                    micro_f1_score.append(self.cross_validation(c, split["train_set"]["examples"], split["train_set"]["labels"]))

                C = c_values[micro_f1_score.index(max(micro_f1_score))]
                print("C: " + str(C) + ": accuracy " + str(max(micro_f1_score)))
                models_f1s.append(max(micro_f1_score))
                model_names.append((C,id, norm_flag))

        best_candiate = models_f1s.index(max(models_f1s))
        C, id, norm_flag = model_names[best_candiate]
        print("\n")
        print("Best model:", test_settings["name"][id])
        features = self.features_building(dataset_rep,
            prev_da=test_settings["settings"][id][0],
            pos_tag=test_settings["settings"][id][1],
            dep_tag=test_settings["settings"][id][2],
            context_flag=test_settings["settings"][id][3])
        mean = 0
        std = 0
        if norm_flag:
            features, mean, std = self.normalization(features)

        split = DatasetManager().official_split(dataset_name, features, dataset_rep["enumerated_DA"])
        split_ids = DatasetManager().get_official_split_ids(dataset_name, features)

        if "ilisten" in dataset_name.lower():
            self.remove_speaker_turns(split, split_ids, dataset_rep, speaker_to_keep)
            print("Done")
        print("Training...")

        accuracy, precision, best_model = self.train(C, split["train_set"]["examples"], split["train_set"]["labels"] , split["test_set"]["examples"], split["test_set"]["labels"], None)


        true_labels = [dataset_rep["number_to_label"][x] for x in split["test_set"]["labels"]]
        model_state = {}
        model_state["model"] = best_model
        model_state["mean"] = mean
        model_state["std"] = std
        model_state["settings"] = test_settings["settings"][id]
        model_state["pos_cipher"] = dataset_rep["pos_cipher"]
        model_state["dep_cipher"] = dataset_rep["dep_cipher"]
        model_state["da_cipher"] = dataset_rep["da_cipher"]
        model_state["number_to_label"] = dataset_rep["number_to_label"]
        '''
        ScoreManager().save_results(representation, dataset_name, "svm",
             speaker_to_keep + "_" + test_settings["name"][id] + "_" + string1 +"_" + str(C),
            true_labels, predictions, fold = test_settings["name"][id] + "_" + string1)
        '''
        self.save_model("svm_" + speaker_to_keep + "_"+ dataset_name + "_" + test_settings["name"][id].replace("+", "_") + ".npy", model_state)
        #ScoreManager().save_results_inspection(representation, dataset_name, "svm", test_settings["name"][id] + "_" + string1 +"_" + str(C),test_tokens,true_labels, predictions, fold = fold)


    def save_model(self, model_name, model_state):
        with open("models/" + model_name, "wb") as f:
            pickle.dump(model_state, f)

    def load_model(self, model_name):
        with open("models/" + model_name, "rb") as f:
            return pickle.load(f)

    def features_building(self, dataset, prev_da=True,
                            pos_tag=True, dep_tag=True, context_flag=True):

        utterances_rep = dataset["we"]

        if type(dataset["we"]) != dict:
            feature_row = dataset["we"]
            if "?" in dataset["utterance"]: # Add a flag if utterance contains question mark
                feature_row = np.append(feature_row, [1])
            else:
                feature_row = np.append(feature_row, [0])

            if context_flag:
                feature_row = np.concatenate([feature_row,  dataset["context"]], axis=0)
            if pos_tag:
                feature_row = np.concatenate([feature_row, dataset["pos_tags"]], axis=0)
            if prev_da:
                feature_row = np.concatenate([feature_row, dataset["da_tags"]], axis=0)
            if dep_tag:
                feature_row = np.concatenate([feature_row,  dataset["dep_tags"]], axis=0)
            return [feature_row]
        else:
            features_segments = {}
            for k, sentences in tqdm(utterances_rep.items()): # k represents the chunk number i.e. the dialogue id
                feature_subset = []
                for id, s in enumerate(sentences):   # sentences: [[utterance_rep], [utterance_rep]]
                    feature_row = []
                    feature_row = s
                    if "?" in dataset["utterances"][k][id]: # Add a flag if utterance contains question mark
                        feature_row = np.append(feature_row, [1])
                    else:
                        feature_row = np.append(feature_row, [0])

                    if context_flag:
                        feature_row = np.concatenate([feature_row,  dataset["context"][k][id]], axis=0)
                    if pos_tag:
                        feature_row = np.concatenate([feature_row, dataset["hot_pos_tags"][k][id]], axis=0)
                    if prev_da:
                        feature_row = np.concatenate([feature_row, dataset["hot_prev_da"][k][id]], axis=0)
                    if dep_tag:
                        feature_row = np.concatenate([feature_row,  dataset["hot_dep_tags"][k][id]], axis=0)

                    feature_subset.append(feature_row)

                features_segments[k] = feature_subset

        return features_segments
    def normalize_features(self, dataset, mean, std):
        if type(dataset) == list:
            normalized_dataset = []
            for element in dataset:
                sub = np.subtract(element, mean) # (X - mean) / std
                norm = np.divide(sub, std)
                normalized_dataset.append(norm)
        else:
            # Dialogue oriented
            normalized_dataset = {}
            for k, v in dataset.items(): # # v contains the rows of dataset, k is the id of dialogue
                normalized_dataset[k] = []
                for row in v:
                    sub = np.subtract(row, mean) # (X - mean) / std
                    norm = np.divide(sub, std)
                    normalized_dataset[k].append(norm)
        return normalized_dataset

    def normalization(self, dataset): # dataset is the output of features_building
        all_data = []
        for k, v in dataset.items(): # v contains the rows of dataset, k is the id of dialogue
            for row in v:
                all_data.append(np.asarray(row))
        mean = np.asarray(all_data).mean(axis=0)
        std = np.asarray(all_data).std(axis=0)
        #std = np.where(std==0, 1, std)
        normalized_dataset = self.normalize_features(dataset, mean, std)
        return normalized_dataset, mean, std


    def cross_validation(self, C, train_x, train_y):
        clf = OneVsOneClassifier(LinearSVC(C=C, max_iter=1000), n_jobs= -1)
        cv = ShuffleSplit(n_splits=7, test_size=0.3, random_state=0)
        scores = cross_val_score(clf, train_x, train_y, cv=cv, scoring= "f1_micro")
        return scores.mean()

    @ignore_warnings(category=ConvergenceWarning)
    def train(self, C, train_x, train_y, test_x, test_y, model_name):
        njobs  = -1
        clf = OneVsOneClassifier(LinearSVC(C=C, max_iter=1000), n_jobs= njobs)
        clf.fit(train_x, train_y)
        pred_y = clf.predict(test_x)
        # Score computing
        accuracy = accuracy_score(test_y, pred_y)
        precision = precision_score(test_y, pred_y, average="weighted")
        return accuracy, precision, clf
