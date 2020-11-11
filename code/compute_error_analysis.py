from score_manager.error_analysis import ErrorAnalysis
from score_manager.score_manager import ScoreManager
from dataset_manager.dataset_analysis import DatasetAnalysis
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

pprint.pprint(f1_classes)

f1_classes = DatasetAnalysis().order_dict(f1_classes, reverse = True)
for k, v in f1_classes.items():
    f1_classes[k] = round(v*100,2)
DatasetAnalysis().bar_chart("results_" + args.dataset_name, "F1 by classes " +
    args.dataset_name +" " + args.model.upper(), "Dialgoue Acts", "F1 (%)",
    list(f1_classes.keys()), list(f1_classes.values()),
    random.randint(0,6), text_flag=True)
new_code = {}
for gt, values in code.items():
    new_code[gt] = DatasetAnalysis().percentage_style(values, list(values.keys()))

DatasetAnalysis().print_heatmap(new_code, args.model + " DA tagging " + args.dataset_name)

for gt, values in code.items():
    print(gt.upper(), ": ")
    pprint.pprint(DatasetAnalysis().percentage_style(values, list(values.keys())))
