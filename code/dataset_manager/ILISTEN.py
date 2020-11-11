import os
import xml.etree.ElementTree as ET
import pandas as pd
from dataset_manager.data_preprocessing import DataPreprocessing
import re
import logging
import spacy
import json
class ILISTEN():
    def __init__(self, location):
            self.corpus_location = location
            if self.corpus_location[-1] != "/":
                self.corpus_location += "/"
            self.dataset = self.build_dataset()
            self.save_csv()
    def save_csv(self):
        with open(self.corpus_location + "/ilisten.csv", "w") as f:
            for file_id , data in self.dataset["examples"].items():
                for row_id, utt in enumerate(data):
                    f.write(utt +","+ self.dataset["labels"][file_id][row_id] + "," +
                    self.dataset["prev_da"][file_id][row_id] + "," +
                    str(self.dataset["segments"][file_id][row_id])+ "," +
                    file_id + "\n")
    def splitter(self, sentence, nlp):
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
    def to_json(self, dataset):
        nlp = spacy.load("it_core_news_sm", disable = ["tagger", "parser", "ner", "textcat",  "entity_linker", "sentecizer"])
        data_to_json = {}
        for file_id, data in dataset["examples"].items():
            data_to_json[file_id] = {}
            for t_id, utt in enumerate(data):
                key = "turn_" + str(t_id)
                data_to_json[file_id][key] = []
                tokened_utt = " ".join(self.splitter(utt, nlp)).strip() # Every token is separated by space
                data_to_json[file_id][key].append({"id":0,
                                                   "FU": tokened_utt,
                                                   "DA": dataset["labels"][file_id][t_id],
                                                   "speaker": dataset["speaker"][file_id][t_id]})
        with open("datasets/ilisten.json", "w") as f:
            f.write(json.dumps(data_to_json, ensure_ascii=False, indent=4))
    def csv_for_segmentation(self):
        nlp = spacy.load("it_core_news_sm", disable = ["tagger", "parser", "ner", "textcat",  "entity_linker", "sentecizer"])
        dataset = {}
        dataset["examples"] = {}
        dataset["labels"] = {}
        dataset["labels_s"] = {}
        dataset["prev_utt"] = {}
        dataset["segments"] = {}
        pars = re.compile("(S|U)")
        for file in os.listdir(self.corpus_location + "ILISTEN/"):
            if "training" in file or "gold" in file:
                root = ET.parse(self.corpus_location + "ILISTEN/" + file).getroot()
                for id, dialogue in enumerate(root.findall(".//dialogue")):
                    key = str(id) + "_" + file.split(".")[0]
                    dataset["examples"][key] = []
                    dataset["labels"][key] = []
                    dataset["labels_s"][key] = []
                    dataset["prev_utt"][key] = []
                    dataset["segments"][key] = []
                    for utterance in dialogue.findall(".//speechAct"):
                        #tmp_utt = [doc.text  for doc in  nlp(utterance.text.lower()) if not doc.is_punct or doc.text == "?"]
                        if pars.split(utterance.attrib["id"])[-2] == "S":
                            dataset["prev_utt"][key].append(utterance.text)
                            dataset["labels_s"][key].append(utterance.attrib["act"].lower())
                        else:
                            dataset["examples"][key].append(utterance.text)
                            dataset["labels"][key].append(utterance.attrib["act"].lower())
                        segment = int(pars.split(utterance.attrib["id"])[-1])
                        if  pars.split(utterance.attrib["id"])[-2] != "S" or segment != 1:
                            segment = dataset["segments"][key][-1] + 1
                        dataset["segments"][key].append(segment)
                with open(self.corpus_location + "/ilisten_for_seg_user_side.csv", "w") as f:
                    for file_id , data in dataset["examples"].items():
                        for row_id, utt in enumerate(data):
                            f.write(file_id + "\t" + dataset["prev_utt"][file_id][row_id]  + "\t" + dataset["labels_s"][file_id][row_id] + "\t" + utt + "\t" +  dataset["labels"][file_id][row_id] +"\n")
                with open(self.corpus_location + "/ilisten_for_seg_system_side.csv", "w") as f:
                    for file_id , data in dataset["prev_utt"].items():
                        for row_id, utt in enumerate(data):
                            if row_id != 0:
                                f.write(file_id + "\t" + dataset["examples"][file_id][row_id-1] + "\t" +  dataset["labels"][file_id][row_id-1] + "\t"+ utt  + "\t" + dataset["labels_s"][file_id][row_id] + "\n")
                            else:
                                f.write(file_id + "\t" + "None" + "\t" +  "None" + "\t"+ utt  + "\t" + dataset["labels_s"][file_id][row_id] + "\n")
        return dataset
    def build_dataset(self):
        nlp = spacy.load("it_core_news_sm", disable = ["tagger", "parser", "ner", "textcat",  "entity_linker", "sentecizer"])
        try:
            assert os.path.exists(self.corpus_location)
            assert os.path.exists(self.corpus_location + "ILISTEN")
            assert os.path.exists(self.corpus_location + "ILISTEN/test_gold.xml")
            assert os.path.exists(self.corpus_location + "ILISTEN/test.xml")
            assert os.path.exists(self.corpus_location + "ILISTEN/training.xml")
        except AssertionError:
            logging.warning("The folder " + self.corpus_location + " does not contain some important files or folders ")
        dataset = {}
        dataset["examples"] = {}
        dataset["labels"] = {}
        dataset["prev_da"] = {}
        dataset["segments"] = {}
        dataset["speaker"] = {}
        pars = re.compile("(S|U)")
        for file in os.listdir(self.corpus_location + "ILISTEN/"):
            if "training" in file or "gold" in file:
                root = ET.parse(self.corpus_location + "ILISTEN/" + file).getroot()
                for id, dialogue in enumerate(root.findall(".//dialogue")):
                    key = str(id) + "_" + file.split(".")[0]
                    dataset["examples"][key] = []
                    dataset["labels"][key] = []
                    dataset["prev_da"][key] = []
                    dataset["segments"][key] = []
                    dataset["speaker"][key] = []
                    for utterance in dialogue.findall(".//speechAct"):
                        #tmp_utt = [doc.text  for doc in  nlp(utterance.text.lower()) if not doc.is_punct or doc.text == "?"]
                        #tmp_utt = [doc.text  for doc in  nlp(utterance.text.lower())]
                        tmp_utt = [doc.text  for doc in  nlp(utterance.text)]
                        #dataset["examples"][key].append(" ".join(tmp_utt).replace(",", "-"))
                        dataset["examples"][key].append(" ".join(tmp_utt))
                        if len(dataset["labels"][key]) > 0:
                            dataset["prev_da"][key].append(dataset["labels"][key][-1])
                        else:
                            dataset["prev_da"][key].append(utterance.attrib["act"].lower())
                        dataset["labels"][key].append(utterance.attrib["act"].lower())
                        segment = int(pars.split(utterance.attrib["id"])[-1])
                        if  pars.split(utterance.attrib["id"])[-2] != "S" or segment != 1:
                            segment = dataset["segments"][key][-1] + 1
                        if "U" in utterance.attrib["id"]:
                            dataset["speaker"][key].append("U")
                        else:
                            dataset["speaker"][key].append("S")
                        dataset["segments"][key].append(segment)

        return dataset
