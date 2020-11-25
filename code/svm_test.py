from svm.svm_trainer import SVM_trainer
import os
import sys
import pprint
import argparse
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("-dataset_name", help="specify the dataset name")
parser.add_argument("-model_system", help="model for system turns")
parser.add_argument("-model_user", help="model for user turns")


args = parser.parse_args()
dataset = args.dataset_name

if ".json" not in dataset:
    dataset += ".json"



svm = SVM_trainer()
model_1 = svm.load_model(args.model_system)
model_2 = svm.load_model(args.model_user)
svm.test_iteration(dataset, "fast_text", model_1, model_2)
