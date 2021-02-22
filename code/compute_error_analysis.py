from score_manager.error_analysis import ErrorAnalysis
from score_manager.score_manager import ScoreManager
import sys
import os
import pprint
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-dataset_name", help="specify the dataset name")
parser.add_argument("-model", help="specify the model svm or cnn")
args = parser.parse_args()
mode = args.model

eval, best_f1, best_acc = ScoreManager().compute_scores(
    "fast_text", args.dataset_name, args.model)

code = eval.evaluation(best_f1)
to_eval = code
accuracy, f1 = eval.compute_scores(to_eval)

f1_classes, recall_classes, precision_classes = eval.score_by_classes(to_eval)
