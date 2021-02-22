from svm.svm_trainer import SVM_trainer
import os
import sys
import pprint
import argparse
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("-dataset_name", help="specify the dataset name")
parser.add_argument("-speaker_to_keep", help="System (S) or User (U)", default="None")

args = parser.parse_args()
dataset = args.dataset_name

if ".json" not in dataset:
    dataset += ".json"
speakers = []
if args.speaker_to_keep == "None":
    speakers = ["U", "S"]
elif args.speaker_to_keep == "S":
    speakers = ["S"]
else:
    speakers = ["U"]

svm = SVM_trainer()

for speaker in speakers:
    rep = ["fast_text"]
    c_values = [0.01, 0.1, 0.2, 0.005]
    test_settings = {}
    #test_settings["settings"] = [[PREV_DA, POS_tags, DEP_tags, Context]]
    if speaker == "S":
        test_settings["settings"] = [[False, True, False, False], # POS
                                     [False, False, True, False], # DEP
                                     [False, True, True, False], # POS+DEP
                                     [False, True, True, True], # ALL
                                     [False, False, False, True], # Context
                                     [False, False, False, False] # WE
                                     ]
        test_settings["name"] = [
        "POS",
        "DEP",
        "POS+DEP",
        "All",
        "Context",
        "Only Word Embedding"
        ]
    else:
        test_settings["settings"] = [[False, True, False, False], # POS
                                     [False, False, True, False], # DEP
                                     [False, True, True, False], # POS+DEP
                                     [True, True, True, False], # PREV_DA+POS+DEP
                                     [True, True, False, False], # PREV_DA+POS
                                     [True, False, True, False], # PREV_DA+DEP
                                     [True, False, False, True], # PREV_DA+Context
                                     [True, True, True, True], # ALL
                                     [True, False, False, False], # PREV_DA
                                     [False, False, False, True], # Context
                                     [False, False, False, False] # WE
                                     ]
        test_settings["name"] = [
        "POS",
        "DEP",
        "POS+DEP",
        "PREV_DA+POS+DEP",
        "PREV_DA+POS",
        "PREV_DA+DEP",
        "PREV_DA+Context",
        "All",
        "PREV_DA",
        "Context",
        "Only Word Embedding"
        ]

    svm.train_iteration(dataset, "fast_text", c_values, test_settings, speaker_to_keep=speaker)
    print("Model for speaker: ", speaker, " trained !")
