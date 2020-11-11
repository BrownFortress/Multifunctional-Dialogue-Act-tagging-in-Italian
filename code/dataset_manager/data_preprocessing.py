import pandas as pd
import spacy
from dataset_manager.wordrepresentation import WordRepresentation
import numpy as np
from collections import Counter
from multiprocessing import Process
from multiprocessing import Manager
from os import path
import os
from tqdm import tqdm
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import pprint
import math
import shutil
import re
import json
import sys

class DataPreprocessing():
    def __init__(self):
        self.pos_tags = {}
        self.dep_tags = {}
        self.word_r =  WordRepresentation()
        self.folder = "preprocessing/"
        self.dataset_folder = "datasets/"
        self.decay = False

    def __sub_preprocess_for_crf(self, data, output, representation_name, punctuation_flag=True):
        pars = re.compile("'")
        nlp = spacy.load("it_core_news_sm", disable = ["ner", "entity_linker", "sentecizer"])

        for file_id, dialog in data["examples"].items():
            output[file_id] = {}
            output[file_id]["utterances"] = []
            output[file_id]["labels"] = []
            output[file_id]["pos"] = []
            output[file_id]["dep"] = []
            output[file_id]["we"] = []
            output[file_id]["is_stop"] = []
            output[file_id]["lemma"] = []
            output[file_id]["is_alpha"] = []
            output[file_id]["is_punct"] = []
            for id, utterance in enumerate(dialog):
                labels = data["labels"][file_id][id]
                docs = nlp(" ".join(utterance).lower())
                tmp_utterance = []
                tmp_labels = []
                tmp_pos = []
                tmp_dep = []
                tmp_we = []
                tmp_lemma = []
                tmp_is_stop = []
                tmp_is_alpha = []
                tmp_is_punct = []
                word_embeddings = self.word_r.get_word_embeddings(utterance, representation_name)
                word_id = 0
                for token in docs:
                    if (not token.is_punct or punctuation_flag) and token.text != "'":
                    #if not token.is_punct:
                        tmp_utterance.append(token.text.lower())
                        tmp_we.append(word_embeddings[word_id])
                        tmp_labels.append(labels[word_id])
                        tmp_pos.append(token.pos_)
                        tmp_dep.append(token.dep_)
                        tmp_lemma.append(token.lemma_)
                        tmp_is_stop.append(token.is_stop)
                        tmp_is_alpha.append(token.is_alpha)
                        tmp_is_punct.append(token.is_punct)
                        word_id += 1
                    elif token.text == "'":
                        tmp_utterance[-1] += "'"
                    #else:
                        #tmp_utterance[-1] += token.text
                output[file_id]["utterances"].append(tmp_utterance)
                output[file_id]["labels"].append(tmp_labels)
                output[file_id]["pos"].append(tmp_pos)
                output[file_id]["dep"].append(tmp_dep)
                output[file_id]["we"].append(tmp_we)
                output[file_id]["is_stop"].append(tmp_is_stop)
                output[file_id]["lemma"].append(tmp_lemma)
                output[file_id]["is_alpha"].append(tmp_is_alpha)
                output[file_id]["is_punct"].append(tmp_is_punct)

    def segmentation_preprocessing(self, dataset_path, representation_name="fast_text"):
        output = {}
        output["examples"] = {}
        output["labels"] = {}
        dataset_name = dataset_path.split("/")[-1].split(".")[0]
        with open(dataset_path, "r") as f:
            dataset = json.loads(f.read())

        nlp = spacy.load("it_core_news_sm", disable=["tagger", "parser", "ner","textcat",  "entity_linker", "sentecizer"])

        most_da = self.most_common_dialogue_acts(dataset, 80)
        for da_id, turns in dataset.items():
            output["examples"][da_id] = []
            output["labels"][da_id] = []
            for turn_id, turn in turns.items():
                utterance = []
                labels = []
                for segment in turn:
                    if segment["speaker"] != "S":
                        seq = nlp(segment["FU"])
                        lenght = 0
                        for w in seq:
                            if w.text != "'":
                                utterance.append(w.text)
                                lenght += 1
                            else:
                                utterance[-1] = utterance[-1] + "'"

                        da = ""
                        if segment["DA"] in most_da or "ilisten" == dataset_name:
                            da = segment["DA"]
                        else:
                            da = "other"
                        labels.extend(["B_" + da] + (lenght-1) * ["I_" + da])

                if len(utterance) > 0:
                    output["examples"][da_id].append(utterance)
                    output["labels"][da_id].append(labels)

        return self.preprocess_for_crf(output, dataset_name, representation_name=representation_name)

    def load_data(self, dataset_name):
        with open(self.dataset_folder+dataset_name, "r") as f:
            return json.loads(f.read())

    def segment(self, labels, utterance):
        indexes = [id_l  for id_l, lab in enumerate(labels) if "B_" in lab ]
        segments = []
        das = []
        if len(indexes) == 0:
            segments = [" ".join(utterance)]
            das = ["_".join(labels[0].split("_")[1:])]
        else:
            for pos, index in enumerate(indexes):
                if pos == len(indexes)-1:
                    segments.append(" ".join(utterance[index:len(utterance)]))
                else:
                    segments.append(" ".join(utterance[index:indexes[pos+1]]))
                das.append("_".join(labels[index].split("_")[1:]))
        return segments, das
    # Preprocess a well formed file containing a list of utterances
    def preprocess_for_crf_file(self, dataset_path, we_flag=False):
        data = {}
        with open(dataset_path, "r") as f:
            data = json.loads(f.read())
        output = {}
        nlp = spacy.load("it_core_news_sm", disable = ["ner", "entity_linker", "sentecizer"])
        for da_id, dialog in data.items():
            output[da_id] = {}
            output[da_id]["utterances"] = []
            output[da_id]["labels"] = []
            output[da_id]["pos"] = []
            output[da_id]["dep"] = []
            output[da_id]["we"] = []
            output[da_id]["is_stop"] = []
            output[da_id]["lemma"] = []
            output[da_id]["is_alpha"] = []
            output[da_id]["is_punct"] = []
            output[da_id]["turns_id"] = []
            for turn_id, turn in dialog.items():
                docs = nlp(turn["FU"])
                tmp_utterance = []
                tmp_pos = []
                tmp_dep = []
                tmp_we = []
                tmp_lemma = []
                tmp_is_stop = []
                tmp_is_alpha = []
                tmp_is_punct = []
                if we_flag:
                    word_embeddings = self.word_r.get_word_embeddings([token.text for token in docs], "fast_text")
                word_id = 0
                for token in docs:
                    if token.text != "'":
                        tmp_utterance.append(token.text.lower())
                        if we_flag:
                            tmp_we.append(word_embeddings[word_id])
                        tmp_pos.append(token.pos_)
                        tmp_dep.append(token.dep_)
                        tmp_lemma.append(token.lemma_)
                        tmp_is_stop.append(token.is_stop)
                        tmp_is_alpha.append(token.is_alpha)
                        tmp_is_punct.append(token.is_punct)
                        word_id += 1
                    elif token.text == "'":
                        tmp_utterance[-1] += "'"

                output[da_id]["utterances"].append(tmp_utterance)
                output[da_id]["pos"].append(tmp_pos)
                output[da_id]["dep"].append(tmp_dep)
                output[da_id]["we"].append(tmp_we)
                output[da_id]["is_stop"].append(tmp_is_stop)
                output[da_id]["lemma"].append(tmp_lemma)
                output[da_id]["is_alpha"].append(tmp_is_alpha)
                output[da_id]["is_punct"].append(tmp_is_punct)
                output[da_id]["turns_id"].append(turn_id)
        return output
    def preprocess_for_crf(self, corpus, data_name, representation_name = "None", punctuation_flag=True):
        # Lowering text
        # Extracting POS, DEP
        if path.exists(self.folder + representation_name + "_prep_" + data_name + "_for_crf"+ ".npy"):
            print("Loading : " + self.folder + representation_name + "_prep_" + data_name + "_for_crf" + ".npy" + " ...")
            return self.load(data_name + "_for_crf", representation_name)

        clean_corpus = {}
        self.__sub_preprocess_for_crf(corpus, clean_corpus, representation_name, punctuation_flag)

        self.save(data_name + "_for_crf", clean_corpus, representation_name)
        return clean_corpus

    def get_utterance_for_crf(self, utterance,representation_name= "fast_text"):
        nlp = spacy.load("it_core_news_sm")
        pars = re.compile("'")
        result = {}
        result["utterance"] = []
        result["pos"] = []
        result["dep"] = []
        result["we"] = []
        result["is_stop"] = []
        result["lemma"] = []
        result["is_alpha"] = []
        docs = nlp(utterance)
        cleaned_utterance = [token.text.lower() for token in docs]
        word_embeddings = self.word_r.get_word_embeddings(cleaned_utterance, representation_name)
        for id_w, token in enumerate(docs):
            if not token.is_punct:
                result["utterance"].append(token.text.lower())
                result["pos"].append(token.tag_)
                result["dep"].append(token.dep_)
                result["we"].append(word_embeddings[id_w])
                result["is_stop"].append(token.is_stop)
                result["lemma"].append(token.lemma_)
                result["is_alpha"].append(token.is_alpha)
        return result

    def load_dataset(self, filename):
        utterances = []
        dialogue_acts = []
        prev_dialogue_acts = []
        segments= []
        dialogue_id = []
        with open(filename) as fp:
            for line in fp:
                tokens = line.split(",")
                utterances.append(tokens[0].lower())
                dialogue_acts.append(tokens[1])
                prev_dialogue_acts.append(tokens[2])
                segments.append(int(tokens[3]))
                #if tokens[4] not in dialogue_id:
                dialogue_id.append(tokens[4].strip())
        return utterances, dialogue_acts, prev_dialogue_acts, segments, dialogue_id


    def order_dict(self, dictionary, reverse =True):
        return {k: v for k, v in sorted(dictionary.items(),
            key=lambda item: item[1], reverse = reverse)}

    def get_the_percentage(self, data_to_percentage, all_data, cut_off = 0):
        common_divisor = sum(Counter(all_data).values())
        dtp = data_to_percentage
        if type(data_to_percentage) != (dict):
            dtp = Counter(data_to_percentage)
        res = {}
        for k,v in dtp.items():
            if (v / float(common_divisor) * 100) > cut_off:
                res[k] = round(v /common_divisor, 3) * 100
        return res
    def most_common_dialogue_acts(self, dataset, coverage):
        dialogue_acts = []
        for dia_id, dialogue in dataset.items():
            for turn_id, turn in dialogue.items():
                for seg in turn:
                    if seg["speaker"] != "S" or coverage == 1:
                        dialogue_acts.append(seg["DA"])
        count = self.order_dict(self.get_the_percentage(dialogue_acts, dialogue_acts))
        threshold = 0
        if coverage <= 1:
            threshold = coverage*100
        else:
            threshold = coverage
        reached = False
        result = []
        id = 0
        amount = 0
        while not reached:
            if amount < threshold:
                da = list(count.keys())[id]
                result.append(da)
                amount += count[da]
                id +=1
            else:
                reached = True
        return result

    def preprocess(self, filename_1, representation="fast_text", window_size=1):
        filename = ""
        if "json" in filename_1:
            if self.dataset_folder not in filename_1:
                filename = self.dataset_folder + filename_1
            else:
                filename = filename_1
            with open(filename, "r") as f:
                dataset = json.loads(f.read())
        else:
            print(filename_1)
            print("The file must be JSON")
            sys.exit(1)

        if not path.exists(filename):
            print("The file " + filename + " does not exist.")
            sys.exit(1)

        fname = filename.split("/")[-1].split(".")[0]
        representation_name = representation

        if path.exists(self.folder + representation_name + "_prep_" + fname + ".npy"):
            print("Loading : " + self.folder + representation_name + "_prep_" + fname + ".npy" + " ...")
            return self.load(filename, representation_name)

        wr = WordRepresentation()
        nlp_inst = spacy.load("it_core_news_sm",  disable = ["ner", "textcat",  "entity_linker", "sentecizer"])
        tokenizer = spacy.load("it_core_news_sm", disable = ["ner", "textcat",  "entity_linker", "sentecizer", "parser", "tagger"])
        most_common_da = self.most_common_dialogue_acts(dataset, 80)
        result = {}
        result["utterances"] = {}
        result["grams"] = {}
        result["context"] = {}
        result["we"] = {}
        result["DA"] = {}
        result["prev_DA"] = {}
        result["pos_tags"] = {}
        result["dep_tags"] = {}
        result["speakers"] = {}
        das = []
        pos = []
        deps = []
        for dia_id, dialogue in dataset.items():
            result["utterances"][dia_id] = []
            result["context"][dia_id] = []
            result["we"][dia_id] = []
            result["DA"][dia_id] = []
            result["prev_DA"][dia_id] = []
            result["pos_tags"][dia_id] = []
            result["dep_tags"][dia_id] = []
            result["speakers"][dia_id] = []
            for turn_id, turn in dialogue.items():
                doc = nlp_inst(" ".join([x["FU"] for x in turn]))
                # Remove punctuation
                #doc_tmp = [x  for x in doc if not x.is_punct]
                #doc = doc_tmp
                threshold = {}
                prev_length = 0
                length = 0

                for x_id, x in enumerate(turn):
                    if x_id == 0:
                        length = len(tokenizer(x["FU"]))
                    else:
                        length += len(tokenizer(x["FU"]))
                    threshold[x_id] = (prev_length, length)
                    prev_length = length

                for seg_id, seg in enumerate(turn):
                    lower_bound, upper_bound = threshold[seg_id]
                    #print(lower_bound, upper_bound)
                    #print([w.text for w in doc[lower_bound:upper_bound]])
                    da = ""
                    if len(doc[lower_bound:upper_bound]) == 0:
                        print(doc)
                        print(lower_bound, upper_bound)
                        print(turn_id, "  ", seg_id)
                        a = 0/0

                    # Cut the tail
                    if seg["DA"] in most_common_da or "ilisten" == fname:
                        da = seg["DA"]
                    else:
                        da = "other2"
                    #doc = nlp_inst(seg["FU"])
                    result["utterances"][dia_id].append(seg["FU"].lower())
                    # Add the mean of word embeddings
                    result["we"][dia_id].append(
                        wr.get_sentence_embedding([w.text.lower() for w in doc[lower_bound:upper_bound]],
                            representation))
                    # Window size is the number of FU or turns that it looks back

                    # Add the mean of previus FU representations
                    result["context"][dia_id].append(self.get_context(result["we"][dia_id]))
                    result["DA"][dia_id].append(da)
                    result["prev_DA"][dia_id].append(self.get_prev_da(result["DA"][dia_id])) # Add previus dialogue acts

                    result["pos_tags"][dia_id].append([w.pos_ for w in doc[lower_bound:upper_bound]]) # Add pos tags
                    result["dep_tags"][dia_id].append([w.dep_ for w in doc[lower_bound:upper_bound]]) # Add dependecies tags
                    result["speakers"][dia_id].append(seg["speaker"])
                    das.append(da)
                    pos.extend(result["pos_tags"][dia_id][-1])
                    deps.extend(result["dep_tags"][dia_id][-1])

        dialogue_act_chipher = self.get_cipher(das)
        pos_chipher = self.get_cipher(pos)
        dep_chipher = self.get_cipher(deps)
        result["pos_cipher"] = pos_chipher
        result["dep_cipher"] = dep_chipher
        self.hot_encode("DA", "hot_DA", result, dialogue_act_chipher)
        dialogue_act_chipher["SOD"] = max(list(dialogue_act_chipher.values()))+1
        result["da_cipher"] = {k:v for k, v in dialogue_act_chipher.items()}    # Deep copy

        self.hot_encode("prev_DA", "hot_prev_da", result, dialogue_act_chipher)
        self.hot_encode("pos_tags", "hot_pos_tags", result, pos_chipher)
        self.hot_encode("dep_tags", "hot_dep_tags", result, dep_chipher)
        del dialogue_act_chipher["SOD"] # SOD: Start of Dialogue
        number_to_label = self.enumerate_labels_dialogue_oriented(dialogue_act_chipher, result)
        result["number_to_label"] = number_to_label
        self.save(filename, result, representation_name)

        return result

    def enumerate_labels_dialogue_oriented(self, da_cipher, result):
        # It keeps dialogue id
        reverse_cipher = {}
        for k, v in da_cipher.items():
            reverse_cipher[v] = k
        result["enumerated_DA"] = {}
        for dia_id, values in result["DA"].items():
            result["enumerated_DA"][dia_id] = [da_cipher[l] for l in values]
        return  reverse_cipher

    def hot_encode(self, key_to_encode, result_key, dictionary, cipher):
        dictionary[result_key] = {}
        for dia_id, values in dictionary[key_to_encode].items():
            dictionary[result_key][dia_id] = []
            for row in values:
                code = len(cipher.keys()) * [0]
                if type(row) == list:
                    for element in row:
                        code[cipher[element]] += 1
                else:
                    code[cipher[row]] += 1
                dictionary[result_key][dia_id].append(code)

    def get_context(self, word_embeddings):
        if len(word_embeddings)>1:
            return word_embeddings[-2]
        else:
            return len(word_embeddings[0]) * [0]



    def get_prev_da(self,  da_list):
        if len(da_list)>1:
            return [da_list[-2]]
        else:
            return ["SOD"]

    def enumerate_labels(self, labels):
        cipher = self.get_cipher(labels)
        reverse_cipher = {}
        for k, v in cipher.items():
            reverse_cipher[v] = k
        result = [cipher[l] for l in labels]
        return  result, reverse_cipher



    def context_window(self, n, utterance_representations):
        context = {}
        print("Building contex")
        for n_dial, representations in tqdm(utterance_representations.items()):
            context[n_dial] = []
            for id, rep in enumerate(representations):
                window = []
                if id != 0:
                    previous_id = id - 1
                    if id < n:
                        window = id
                    else:
                        window = n
                    max = len(list(range(previous_id, previous_id-window, -1)))
                    # Dmg is a penalty, that decreases the influence of farest examples
                    if self.decay:
                        context[n_dial].append(np.sum([np.divide(representations[win], (1 + dmg) )
                            for dmg, win in  enumerate(range(previous_id, previous_id-window, -1))], axis=0))
                    else:
                        context[n_dial].append(np.sum([np.divide(representations[win], 1 )
                            for dmg, win in  enumerate(range(previous_id, previous_id-window, -1))], axis=0))

                else:
                    context[n_dial].append(rep)
        return context

    def get_cipher(self, data):
        # Data to number
        cipher = {}
        used = []
        if type(data[0]) == list:
            all_data = [y for x in data for y in x] # Flatting the list of list called data [ [TAG1, TAG2], [TAG1]..]
        else:
            all_data = data
        unique = [x for x in all_data if x not in used and (used.append(x) or True)]
        for id_e, element in enumerate(unique):
            cipher[element] = id_e

        return cipher
    # NN models
    def representation_for_nn(self, dataset, representation, lang, fname):
        number_to_word_embeddings = {}
        word_to_number = {}
        label_to_number = {}
        pos_to_number = {}
        result = {}
        nlp = spacy.load("it_core_news_sm", disable = ["parser", "ner", "textcat",  "entity_linker", "sentecizer"])
        word_to_number["pad"] = 0
        word_to_number["eos"] = 1
        word_to_number["sos"] = 2
        word_to_number["special_pad"] = -1
        pos_to_number["pad"] = 0
        pos_to_number["eos"] = 1
        pos_to_number["sos"] = 2
        label_to_number["pad"] = 0
        label_to_number["eos"] = 1
        label_to_number["sos"] = 2
        if "iso" in fname.lower():
            label_to_number["other"] = 3
            da_id = 4
        else:
            da_id = 3
        word_id = 3
        pos_id = 3
        if "iso" not in fname.lower():
            m_da = self.most_common_dialogue_acts(dataset, 1)
        else:
            m_da = self.most_common_dialogue_acts(dataset, 0.8)
        label_owner = {}
        label_owner["other"] = "U"
        for dialogue_id, dialogue in dataset.items():
            result[dialogue_id] = {}
            for turn_id, turn in dialogue.items():
                result[dialogue_id][turn_id] = {}
                result[dialogue_id][turn_id]["examples"] = []
                result[dialogue_id][turn_id]["pos_tags"] = []
                result[dialogue_id][turn_id]["labels"] = []
                result[dialogue_id][turn_id]["speaker"] = []
                for seg in turn:
                    # seg["FU"] is already split using spacy and it was saved with a space as separator
                    if seg["DA"] not in label_owner.keys():
                        label_owner[seg["DA"]] = []
                    if seg["speaker"] not in label_owner[seg["DA"]]:
                        label_owner[seg["DA"]].append(seg["speaker"])

                    if seg["DA"] not in label_to_number.keys() and seg["DA"] in m_da:
                        label_to_number[seg["DA"]] = da_id
                        da_id += 1
                    seq_tmp = [x.text.lower() for x in nlp(seg["FU"].replace("…", "").replace("...", ""))]
                    pos_tags = [x.pos_ for x in nlp(seg["FU"].replace("…", "").replace("...", ""))]
                    seq = []
                    for x in seq_tmp:
                        if x == "'":
                            seq[-1] += x
                        else:
                            seq.append(x)
                    rep = []
                    pos_rep = []
                    for w_id, word in enumerate(seq):
                        if word not in word_to_number.keys():
                            word_to_number[word] = word_id
                            word_id += 1
                        if pos_tags[w_id] not in pos_to_number.keys():
                            pos_to_number[pos_tags[w_id]] = pos_id
                            pos_id += 1

                        rep.append(word_to_number[word])
                        pos_rep.append(pos_to_number[pos_tags[w_id]])

                        if word_to_number[word] not in number_to_word_embeddings.keys():
                            number_to_word_embeddings[word_to_number[word]] = self.word_r.get_word_embedding(word, representation)
                    result[dialogue_id][turn_id]["examples"].append(rep)
                    result[dialogue_id][turn_id]["pos_tags"].append(pos_rep)
                    if seg["DA"] in m_da:
                        result[dialogue_id][turn_id]["labels"].append(label_to_number[seg["DA"]])
                    else:
                        result[dialogue_id][turn_id]["labels"].append(label_to_number["other"])
                    result[dialogue_id][turn_id]["speaker"].append(seg["speaker"])
        emb_dim = len(list(number_to_word_embeddings.values())[-1])
        number_to_word_embeddings[-1] = np.repeat(0.2, emb_dim)
        number_to_word_embeddings[0] = np.repeat(0, emb_dim)
        number_to_word_embeddings[1] = np.repeat(0.3, emb_dim)
        number_to_word_embeddings[2] = np.repeat(0.7, emb_dim)
        to_return = {}
        to_return["utterance_encoded"] = result
        to_return["number_to_word_embeddings"] = number_to_word_embeddings
        to_return["word_to_number"] = word_to_number
        to_return["label_to_number"] = label_to_number
        to_return["label_owner"] = label_owner
        to_return["pos_to_number"] = pos_to_number
        return to_return

    def sequences_preprocessing(self, filename_1, representation_name="fast_text", lang="it"):
            fname = filename_1.split("/")[-1].split(".")[0]
            if path.exists(self.folder + representation_name + "_prep_"  + fname + "_for_nn" + ".npy"):
                print("Loading : " + self.folder + representation_name + "_prep_"+ fname + "_for_nn" + ".npy" + " ...")
                return self.load(fname+"_for_nn", representation_name)
            filename = ""
            if self.dataset_folder not in filename_1:
                filename = self.dataset_folder + filename_1
            else:
                filename = filename_1
            if "json" not in filename:
                filename += ".json"
            if not os.path.exists(filename):
                print("File : ", filename, " does not exist")
                sys.exit(1)

            with open(filename, "r") as f:
                dataset = json.loads(f.read())
            # Words representations
            # Labales
            result  = self.representation_for_nn(dataset, representation_name, lang, fname)
            self.save(fname+"_for_nn", result, representation_name)
            return result


    def save(self, filename, representation, rep_name):
        fname = filename.split("/")[-1].split(".")[0]
        with open(self.folder + rep_name +"_prep_" + fname + ".npy", "wb") as f:
            pickle.dump(representation, f)

    def load(self, filename, rep_name):
        fname = filename.split("/")[-1].split(".")[0]
        with open(self.folder + rep_name + "_prep_" + fname + ".npy", "rb") as f:
            return pickle.load(f)