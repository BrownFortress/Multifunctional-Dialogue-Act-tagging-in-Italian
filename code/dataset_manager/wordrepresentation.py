import numpy as np
import pandas as pd
import itertools
import pickle
import random
import sys
import os
from tqdm import tqdm
import fasttext
from itertools import groupby
from os import path


class WordRepresentation():
    def __init__(
        self, file_names = []
    ):

        self.w2v_model = None
        self.folder = "preprocessing/"


    def select_w2v_model(self, representation_name, model_name = None):
        if representation_name == "fast_text":
            if model_name != None:
                self.w2v_model = fasttext.load_model(model_name)
            else:
                print("This feature implemented yet")
        else:
            print("Representation <", representation_name, " currently is not supported")


    def get_word_embeddings(self, sentence, representation_name):
        if representation_name == "fast_text":
            if "fasttext" not in str(type(self.w2v_model)):
                self.select_w2v_model(representation_name, "preprocessing/wiki.it.bin")

            return [self.w2v_model.get_word_vector(word) for word in sentence]

        else:
            print("Representation <", representation_name , "> currently is not supported!")
            return
    def get_word_embedding(self, word, representation_name):
        if representation_name == "fast_text":
            if "fasttext" not in str(type(self.w2v_model)):
                self.select_w2v_model(representation_name, "preprocessing/wiki.it.bin")
            return self.w2v_model.get_word_vector(word)

        else:
            print("Representation <", representation_name , "> currently is not supported!")
            return
    def get_sentence_embedding(self, sentence, representation_name):
        if representation_name == "fast_text":
            if "fasttext" not in str(type(self.w2v_model)):
                self.select_w2v_model(representation_name, "preprocessing/wiki.it.bin")
            return np.asarray([self.w2v_model.get_word_vector(word) for word in sentence]).mean(axis=0)

        elif representation_name == "glove":
            if self.w2v_model == None:
                print("Not implemented yet")
                return
                #self.select_w2v_model(representation_name, )
        else:
            print("Representation <", representation_name , "> currently is not supported!")
            return

    def utterance_representation(self, sentence):
        np_sentence = np.asarray(sentence)
        utterance = np_sentence.mean(axis=0)
        return utterance
