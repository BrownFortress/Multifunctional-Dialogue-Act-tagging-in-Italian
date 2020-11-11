import numpy as np
import random
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
import json
#import torch
import os
class Batch():
    def __init__(self, source, target):
        self.source = source
        self.target = target
class DatasetManager():
    def __init__(self):
        self.padding_code= 1
        self.sos_code = 0
        self.eos_code = 2
        self.split_folder = "standard_splits/"
    # data coming from preprocessing function
    def dialogue_split_folds(self, split_name, dialogue_list, train_size, validation_size, n_folds):
        if not os.path.exists(self.split_folder+split_name+"/"):
            os.mkdir(self.split_folder + split_name)
        split_rule_train = round(len(dialogue_list) * train_size)
        split_rule_validation = round(len(dialogue_list) * (train_size+validation_size))
        for i in range(0, n_folds):
            path = self.split_folder+split_name+"/" + "fold_" + str(i)
            if not os.path.exists(path):
                os.mkdir(path)
            random.shuffle(dialogue_list)
            split = {}
            split["train_set"] = []
            split["validation_set"] = []
            split["test_set"] = []
            for i, element in enumerate(dialogue_list):
                if i < split_rule_train:
                    split["train_set"].append(element)
                elif i < split_rule_validation:
                    split["validation_set"].append(element)
                else:
                    split["test_set"].append(element)
            for partition, dialogues in split.items():
                with open(path + "/" + partition+".txt", "w") as f:
                    for dial in dialogues:
                        f.write(dial + "\n")
    # Duplicated method
    def split_given_ids_for_nn(self, data, split_ids, dialogue_level = False, labels=None):
        split = {}
        split["train_set"] = {}
        split["test_set"] = {}
        split["validation_set"] = {}
        flag_warning = False
        warning = {}
        warning["train_set"] = 0
        warning["test_set"] = 0
        warning["validation_set"] = 0
        if labels != None:
            for set_name, ids in split_ids.items():
                split[set_name]["examples"] = []
                split[set_name]["labels"] = []
                if dialogue_level:
                    for d_id in ids:
                        if d_id in data.keys():
                            warning[set_name] +=1
                            split[set_name]["examples"].append(data[d_id])
                            split[set_name]["labels"].append(labels[d_id])
                        else:
                            flag_warning = True
                else:
                    for d_id in ids:
                        if d_id in data.keys():
                            warning[set_name] +=1
                            split[set_name]["examples"].extend(data[d_id])
                            split[set_name]["labels"].extend(labels[d_id])
                        else:
                            flag_warning = True
        else:
            for set_name, ids in split_ids.items():
                split[set_name]["examples"] = []
                split[set_name]["labels"] = []
                split[set_name]["pos_tags"] = []
                split[set_name]["speakers"] = []
                if dialogue_level:
                    for d_id in ids:

                        if d_id in data.keys():
                            warning[set_name] +=1
                            split[set_name]["examples"].append(data[d_id]["examples"])
                            split[set_name]["labels"].append(data[d_id]["labels"])
                        else:
                            print(d_id)
                            flag_warning = True
                else:
                    for d_id in ids:
                        if d_id in data.keys():
                            warning[set_name] +=1
                            split[set_name]["examples"].extend(data[d_id]["examples"])
                            split[set_name]["labels"].extend(data[d_id]["labels"])
                            split[set_name]["pos_tags"].extend(data[d_id]["pos_tags"])
                            split[set_name]["speakers"].extend(data[d_id]["speakers"])
                        else:
                            flag_warning = True

        if flag_warning:
            print("Not all dialogues were found in the splits")
        print("Training : ", warning["train_set"], " dialogues")
        print("Test :",warning["test_set"]," dialogues")
        print("Validation :", warning["validation_set"]," dialogues")
        return split

    def split_given_ids(self, data, labels, split_ids, dialogue_level = False):
        split = {}
        split["train_set"] = {}
        split["test_set"] = {}
        split["validation_set"] = {}
        flag_warning = False
        warning = {}
        warning["train_set"] = 0
        warning["test_set"] = 0
        warning["validation_set"] = 0
        if labels != None:
            for set_name, ids in split_ids.items():
                split[set_name]["examples"] = []
                split[set_name]["labels"] = []
                if dialogue_level:
                    for d_id in ids:
                        if d_id in data.keys():
                            warning[set_name] +=1
                            split[set_name]["examples"].append(data[d_id])
                            split[set_name]["labels"].append(labels[d_id])
                        else:
                            flag_warning = True
                else:
                    for d_id in ids:
                        if d_id in data.keys():
                            warning[set_name] +=1
                            split[set_name]["examples"].extend(data[d_id])
                            split[set_name]["labels"].extend(labels[d_id])
                        else:
                            flag_warning = True

        if flag_warning:
            print("Not all dialogues were found in the splits")
        print("Training : ", warning["train_set"], " dialogues")
        print("Test :",warning["test_set"]," dialogues")
        print("Validation :", warning["validation_set"]," dialogues")
        return split

    def load_fold(self, dataset_name, fold_name):
        result = {}
        for file in os.listdir(self.split_folder + dataset_name + "/" + fold_name + "/"):
            result[file.split(".")[0]] = []
            with open(self.split_folder + dataset_name + "/" + fold_name + "/" + file, "r") as f:
                for line in f.readlines():
                    result[file.split(".")[0]].append(line.strip())
        return result

    def load_folds(self, fold_name):
        result = {}
        for dir in os.listdir(self.split_folder + fold_name):
            result[dir] = {}
            for file in os.listdir(self.split_folder + fold_name + "/" + dir + "/"):
                result[dir][file.split(".")[0]] = []
                with open(self.split_folder + fold_name + "/" + dir + "/" + file, "r") as f:
                    for line in f.readlines():
                        result[dir][file.split(".")[0]].append(line.strip())
        return result

    def get_official_split_ids(self, dataset_name, data = None):
        split = {}
        split["train_set"] = []
        split["test_set"] = []
        split["validation_set"] = []
        if "ilisten" in dataset_name.lower():
            split["train_set"] = []
            tmp_train_id = []
            for dialog_id, train in data.items():
                if "training" in dialog_id:
                    tmp_train_id.append(dialog_id)
                else:
                    split["test_set"].append(dialog_id)
            #validation_size = (len(tmp_train_id) - int(len(tmp_train_id) * 0.1)) - 1
            split["validation_set"] = []#tmp_train_id[validation_size:]
            split["train_set"] = tmp_train_id #tmp_train_id[0:validation_size]
        else:
            print("Official split is not supported: ", dataset_name.lower())
            return
        return split

    def get_official_split_ids_for_nn(self, dataset_name, data = None):
        # It contains the validation set
        with open(self.split_folder + "ilisten_split.json") as f:
            split = json.loads(f.read())

        return split


    # If dialogue level is true the it return a list of content for each dialogue
    def official_split(self, dataset_name, data, labels, dialogue_level=False):
        split_ids = self.get_official_split_ids(dataset_name, data)
        split = self.split_given_ids(data, labels, split_ids, dialogue_level = dialogue_level)
        return split

    def split_train_test(self, data_x, data_y, train_size):
        train_size = int(len(data_x) * train_size)
        train_set = {}
        test_set = {}
        train_set["examples"] = data_x[0: train_size]
        train_set["labels"] = data_y[0: train_size]
        test_set["examples"] = data_x[train_size:]
        test_set["labels"] = data_y[train_size:]
        return train_set, test_set
    # NN models
    def padding_batches_of_windows(self, batches, data):
        new_batchs = []
        padding_code = data["number_to_word_embeddings"][data["word_to_number"]["pad"]]
        start_code = data["number_to_word_embeddings"][data["word_to_number"]["sos"]]
        end_code = data["number_to_word_embeddings"][data["word_to_number"]["eos"]]
        special_pad =  data["number_to_word_embeddings"][data["word_to_number"]["special_pad"]]
        pos_padding_code = data["pos_to_number"]["pad"]
        pos_start_code = data["pos_to_number"]["sos"]
        pos_end_code = data["pos_to_number"]["eos"]

        for id_batch, batch in enumerate(batches):
            x_part = [x for window in batch["examples"] for x in window]
            max_len = len(max(x_part, key=len))
            tmp_batches = []
            tmp_lengths = []
            for window in batch["examples"]:
                tmp_win = []
                tmp_len = []
                for seq in window:
                    tmp_len.append(len(seq))
                    len_diff = max_len - len(seq)
                    if (seq[0]==start_code).all():
                        tmp_win.append(seq + [start_code] * len_diff)
                    elif (seq[0]==end_code).all():
                        tmp_win.append(seq + [end_code] * len_diff)
                    elif (seq[0]==special_pad).all():
                        tmp_win.append(seq + [special_pad] * len_diff)
                    else:
                        tmp_win.append(seq + [padding_code] * len_diff)
                tmp_batches.append(tmp_win)
                tmp_lengths.append(tmp_len)

            pos_tags_tmp_batches = []
            for window in batch["pos_tags"]:
                tmp_win = []
                for seq in window:
                    len_diff = max_len - len(seq)
                    if (np.asarray(seq)==pos_start_code).all():
                        tmp_win.append(seq + [pos_start_code]*len_diff)
                    elif (np.asarray(seq)==pos_end_code).all():
                        tmp_win.append(seq + [pos_end_code]*len_diff)
                    else:
                        tmp_win.append(seq + [pos_padding_code]*len_diff)
                pos_tags_tmp_batches.append(tmp_win)
            new_batchs.append({"examples": tmp_batches, "labels": batch["labels"],
                            "speakers": batch["speakers"],"pos_tags": pos_tags_tmp_batches,
                            "lengths":tmp_lengths})
        return new_batchs

    def padding_window(self, window, data, type=None):
        window_tmp = []
        if type == None:
            padding_code = data["number_to_word_embeddings"][data["word_to_number"]["pad"]]
            special_pad = data["number_to_word_embeddings"][data["word_to_number"]["special_pad"]]
            start_code = data["number_to_word_embeddings"][data["word_to_number"]["sos"]]
            end_code = data["number_to_word_embeddings"][data["word_to_number"]["eos"]]
            max_len = len(max(window, key=len))
            for w_id, seg in enumerate(window):
                len_diff = max_len - len(seg)
                if (seg[0]==special_pad).all():
                    window_tmp.append(seg + [special_pad]*len_diff)
                else:
                    window_tmp.append(seg + [padding_code]*len_diff)
            window_tmp.insert(0, [start_code]*max_len)
            window_tmp.append([end_code]*max_len)
            return window_tmp
        elif type == "pos_tags":
            padding_code = data["pos_to_number"]["pad"]
            start_code = data["pos_to_number"]["sos"]
            end_code = data["pos_to_number"]["eos"]
            max_len = len(max(window, key=len))
            for w_id, seg in enumerate(window):
                len_diff = max_len - len(seg)
                window_tmp.append(seg + [padding_code]*len_diff)
            window_tmp.insert(0, [start_code]*max_len)
            window_tmp.append([end_code]*max_len)
            return window_tmp
        else:
            start_id = 0
            bound_found = False
            window_tmp = [x for x in window]
            window_tmp.insert(0, data["label_to_number"]["sos"])
            window_tmp.append(data["label_to_number"]["eos"])

            return window_tmp
    def feature_building(self, data, window_size=5, test_flag=False, MAX_LEN=20):
        dataset = {}
        # Window size is referred to segmenent but it could be referred also to turns
        for d_id, dialogue in data["utterance_encoded"].items():
            dataset[d_id] = {}
            dataset[d_id]["examples"] = []
            dataset[d_id]["labels"] = []
            dataset[d_id]["pos_tags"] = []
            dataset[d_id]["speakers"] = []
            for t_id, turn in dialogue.items():
                 # Get le last words
                #threshold = (len(seq) - MAX_LEN - 1)
                #threshold = MAX_LEN
                # Take the last MAX_LEN tokens
                dataset[d_id]["examples"].extend([[data["number_to_word_embeddings"][x]
                    for id_w, x in enumerate(seq) if id_w > (len(seq) - MAX_LEN - 1)] for turn_id, seq in enumerate(turn["examples"])])
                dataset[d_id]["labels"].extend(turn["labels"])
                dataset[d_id]["speakers"].extend(turn["speaker"])
                dataset[d_id]["pos_tags"].extend([[x for id_w, x in enumerate(seq) if id_w > (len(seq) - MAX_LEN - 1)] for seq in turn["pos_tags"]])
                assert len(dataset[d_id]["examples"]) == len(dataset[d_id]["labels"])
            windows = {}
            window_example = []
            window_label = []
            window_speaker = []
            window_postags = []
            special_pad = data["number_to_word_embeddings"][data["word_to_number"]["special_pad"]]
            lab_pad = [data["label_to_number"]["pad"]]
            # Window building
            for new_d_id, new_dialogue in dataset.items():
                windows[new_d_id] = {}
                windows[new_d_id]["examples"] = []
                windows[new_d_id]["pos_tags"] = []
                windows[new_d_id]["labels"] = []
                windows[new_d_id]["speakers"] = []
                examples_candidate_window = None
                labels_candidate_window = None
                speakers_candidate_window = None
                pos_tags_candidate_window = None
                for seg_id, seg in enumerate(new_dialogue["examples"]):
                    window_example.append(seg)
                    window_label.append(dataset[new_d_id]["labels"][seg_id])
                    window_postags.append(dataset[new_d_id]["pos_tags"][seg_id])
                    window_speaker.append(4 if dataset[new_d_id]["speakers"][seg_id] == "S" else 5)
                    if len(window_example) <= window_size:
                        len_diff = window_size - len(window_example)
                        # Add paddind at the beginning of the window, on the left of the window sequence
                        examples_candidate_window = [[special_pad]]*len_diff + window_example
                        pos_tags_candidate_window = [lab_pad]*len_diff + window_postags
                        labels_candidate_window = lab_pad*len_diff + window_label
                        speakers_candidate_window = lab_pad*len_diff + window_speaker
                    else:
                        # Get rid of the first window element
                        window_example.pop(0)
                        window_postags.pop(0)
                        window_label.pop(0)
                        window_speaker.pop(0)
                        examples_candidate_window = window_example
                        pos_tags_candidate_window = window_postags
                        labels_candidate_window = window_label
                        speakers_candidate_window = window_speaker
                    # It adds the window elements sos and eos. It adds padding at token level
                    examples_candidate_window = self.padding_window(examples_candidate_window, data)
                    labels_candidate_window = self.padding_window(labels_candidate_window, data, type="labels")
                    speakers_candidate_window = self.padding_window(speakers_candidate_window, data, type="speaker")
                    pos_tags_candidate_window = self.padding_window(pos_tags_candidate_window, data, type="pos_tags")


                    if not test_flag or not (dataset[new_d_id]["speakers"][seg_id] == "S"):
                        windows[new_d_id]["examples"].append(examples_candidate_window)
                        windows[new_d_id]["pos_tags"].append(pos_tags_candidate_window)
                        windows[new_d_id]["labels"].append(labels_candidate_window)
                        windows[new_d_id]["speakers"].append(speakers_candidate_window)
                        assert len(windows[new_d_id]["examples"][-1]) == window_size + 2
                        assert len(windows[new_d_id]["speakers"][-1]) == window_size + 2
                        assert len(windows[new_d_id]["labels"][-1]) == window_size + 2
        return windows
