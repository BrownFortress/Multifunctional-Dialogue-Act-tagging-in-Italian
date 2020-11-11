import time
import torch
import torch.nn as nn
import random
import numpy as np
from collections import Counter
import xml.etree.ElementTree as ET
import string
import spacy
import re
import sys
import os
import json

def to_json(dataset):
    nlp = spacy.load("it_core_news_sm", disable = ["tagger", "parser", "ner", "textcat",  "entity_linker", "sentecizer"])
    data_to_json = {}
    for file_id, data in dataset.items():
        data_to_json[file_id] = {}
        for t_id, utt in enumerate(data["examples"]):
            key = "turn_" + str(t_id)
            data_to_json[file_id][key] = []
            tokened_utt = " ".join(splitter(utt, nlp)).strip() # Every token is separated by space
            data_to_json[file_id][key].append({"id":0,
                                               "FU": tokened_utt,
                                               "DA": dataset[file_id]["labels"][t_id],
                                               "speaker": dataset[file_id]["speaker"][t_id]})
    with open("datasets/iLISTEN.json", "w") as f:
        f.write(json.dumps(data_to_json, ensure_ascii=False, indent=4))
def build_dataset():
    corpus_location = "datasets/"
    nlp = spacy.load("it_core_news_sm", disable = ["tagger", "parser", "ner", "textcat",  "entity_linker", "sentecizer"])
    try:
        assert os.path.exists(corpus_location)
        assert os.path.exists(corpus_location + "ILISTEN")
        assert os.path.exists(corpus_location + "ILISTEN/test_gold.xml")
        assert os.path.exists(corpus_location + "ILISTEN/test.xml")
        assert os.path.exists(corpus_location + "ILISTEN/training.xml")
    except AssertionError:
        logging.warning("The folder " + corpus_location + " does not contain some important files or folders ")
    dataset = {}
    pars = re.compile("(S|U)")
    for file in os.listdir(corpus_location + "ILISTEN/"):
        if "training" in file or "gold" in file:
            root = ET.parse(corpus_location + "ILISTEN/" + file).getroot()
            for id, dialogue in enumerate(root.findall(".//dialogue")):
                key = str(id) + "_" + file.split(".")[0]
                if key not in dataset.keys():
                    dataset[key] = {}
                    dataset[key]["examples"] = []
                    dataset[key]["labels"] = []
                    dataset[key]["prev_da"] = []
                    dataset[key]["speaker"] = []
                    dataset[key]["segments"] = []

                for utterance in dialogue.findall(".//speechAct"):
                    #tmp_utt = [doc.text  for doc in  nlp(utterance.text.lower()) if not doc.is_punct or doc.text == "?"]
                    #tmp_utt = [doc.text  for doc in  nlp(utterance.text.lower())]
                    tmp_utt = [doc.text  for doc in  nlp(utterance.text)]
                    #dataset["examples"][key].append(" ".join(tmp_utt).replace(",", "-"))
                    dataset[key]["examples"].append(" ".join(tmp_utt))
                    if len(dataset[key]["labels"]) > 0:
                        dataset[key]["prev_da"].append(dataset[key]["labels"][-1])
                    else:
                        dataset[key]["prev_da"].append(utterance.attrib["act"].lower())
                    dataset[key]["labels"].append(utterance.attrib["act"].lower())
                    segment = int(pars.split(utterance.attrib["id"])[-1])
                    if  pars.split(utterance.attrib["id"])[-2] != "S" or segment != 1:
                        segment = dataset[key]["segments"][-1] + 1
                    if "U" in utterance.attrib["id"]:
                        dataset[key]["speaker"].append("U")
                    else:
                        dataset[key]["speaker"].append("S")
                    dataset[key]["segments"].append(segment)

    return dataset
def splitter(sentence, nlp):
    first_split = sentence.split(" ")
    pars = re.compile("(\?|\.|\!|\,)")
    doc = nlp(sentence)
    chunks = [""]
    for d in doc:
        if d.is_punct:
            chunks[-1] = chunks[-1] + d.text
        elif pars.findall(d.text):
                sub_split = pars.split(d.text)
                all_toks = pars.findall(d.text)
                str_tmp = ""
                for ids, sub_s in enumerate(sub_split):
                    if sub_s not in all_toks:
                        str_tmp = sub_s
                    else:
                        chunks.append(str_tmp+sub_s)
                        str_tmp = ""
                if str_tmp != "":
                    chunks.append(str_tmp)
        else:
            if not d.is_space:
                chunks.append(d.text)
    if chunks[0] == "":
        chunks.pop(0)
    return chunks

if not os.path.exists("datasets/ILISTEN/"):
	print("datasets/ILISTEN/ does not exist")
	sys.exit(1)

nlp1 = spacy.load("it_core_news_sm", disable = ["tagger", "parser", "ner", "textcat",  "entity_linker", "sentecizer"])
#dataset = {}
'''
pars = re.compile("(S|U)")
for file in os.listdir("datasets/ILISTEN/"):
    if "training" in file or "gold" in file:
        root = ET.parse("datasets/" + "ILISTEN/" + file).getroot()
        for id, dialogue in enumerate(root.findall(".//dialogue")):
            key = str(id) + "_" + file.split(".")[0]
            if key not in dataset.keys():
                dataset[key] = {}
                dataset[key]["examples"] = []
                dataset[key]["labels"] = []
                dataset[key]["prev_da"] = []
                dataset[key]["speaker"] = []
                dataset[key]["segments"] = []
            for utterance in dialogue.findall(".//speechAct"):
                #tmp_utt = [doc.text  for doc in  nlp(utterance.text.lower()) if not doc.is_punct or doc.text == "?"
                dataset[key]["examples"].append(utterance.text)
                if len(dataset[key]["labels"]) > 0:
                    dataset[key]["prev_da"].append(dataset[key]["labels"][-1])
                else:
                    dataset[key]["prev_da"].append(utterance.attrib["act"].lower())
                dataset[key]["labels"].append(utterance.attrib["act"].lower())
                dataset[key]["speaker"].append(pars.split(utterance.attrib["id"])[-2])
                segment = int(pars.split(utterance.attrib["id"])[-1])
                if  pars.split(utterance.attrib["id"])[-2] != "S" or segment != 1:
                    segment = dataset[key]["segments"][-1] + 1
                dataset[key]["segments"].append(segment)
'''
dataset = build_dataset()
to_json(dataset)
with open("datasets/mapping.json") as f:
    mapping = json.loads(f.read())

ilisten = {}

segmentation = {}
for da_id, dialogue in dataset.items():
    if da_id not in segmentation.keys():
        segmentation[da_id] = {}
    turns = []
    for id_s, utt in enumerate(dialogue["examples"]):
        key = "turn_" + str(id_s)
        segmentation[da_id][key] = []
        turns.append(utt)
        utterance = splitter(utt, nlp1)
        segments = mapping[da_id][key]["segmentation"]
        das = mapping[da_id][key]["DAs"]

        if segments != "0":
            splits = segments.replace(" ", "").split(",")
            dialogue_acts = das.replace(" ", "").split(",")
            for seg_id, s in enumerate(splits):
                if "-" in s:
                    b, e = s.split("-")
                    b = int(b)
                    if e == "end":
                        e = len(utterance) - 1
                    else:
                        e = int(e)
                else:
                    b = int(s)
                    e = b
                segmentation[da_id][key].append({"id": seg_id,
                                     "FU": " ".join(utterance[b:e+1]),
                                     "speaker": dialogue["speaker"][id_s],
                                     "DA":dialogue_acts[seg_id]})

        else:
            segmentation[da_id][key].append({"id": 0,
                                 "FU": " ".join(utterance),
                                 "speaker": dialogue["speaker"][id_s],
                                 "DA":das})

with open("datasets/iLISTEN2ISO.json", "w") as f:
    f.write(json.dumps(segmentation, ensure_ascii=False, indent=4))
