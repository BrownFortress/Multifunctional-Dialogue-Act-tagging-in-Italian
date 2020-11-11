from dataset_manager.data_preprocessing import DataPreprocessing
from dataset_manager.wordrepresentation import WordRepresentation
import spacy
from svm.svm_trainer import SVM_trainer
import json
import argparse

class bcolor:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def hot_encode(to_convert, cipher):
    code = len(cipher.keys()) * [0]
    if type(to_convert) == list:
        for element in to_convert:
            code[cipher[element]] += 1
    else:
        code[cipher[to_convert]] += 1
    return code



parser = argparse.ArgumentParser()
parser.add_argument("-input_file", help="specify the input file")
parser.add_argument("-model_name", help="specify the model name to use")
args = parser.parse_args()
print(args.model_name)
svm = SVM_trainer()
model_state = svm.load_model("svm_ILISTEN_ISO_PREV_DA_POS.npy")


# Pos tag cypher
# Dep tag cypher
# Prev DA cypher
# Context
# window_size
# Norm and STD
dataset = {}
with open(args.input_file, "r") as f:
    dataset = json.loads(f.read())

wr = WordRepresentation()
nlp_inst = spacy.load("it_core_news_sm",  disable = ["ner", "textcat",  "entity_linker", "sentecizer"])
results= {}
if "SOD" not in model_state["da_cipher"].keys():
    model_state["da_cipher"]["SOD"] = max(list(model_state["da_cipher"].values()))+1
print(bcolor.OKGREEN + "Preprocessing and tagging are running... " + bcolor.ENDC)
for dialogue_id, dialogue in dataset.items():
    prev_DA = "SOD" # Start of Dialogue
    prev_contex = []
    results[dialogue_id] = {}
    for turn_id, turn in dialogue.items():
        results[dialogue_id][turn_id] = []
        for seg in turn:
            doc = nlp_inst(seg["FU"].lower())
            # Add the mean of word embeddings
            feature_raw = {}
            feature_raw["utterance"] = seg["FU"]
            feature_raw["we"] = wr.get_sentence_embedding([w.text.lower() for w in doc], "fast_text")
            feature_raw["contex"] = feature_raw["we"] if len(prev_contex) == 0 else prev_contex
            feature_raw["pos_tags"] = hot_encode([w.pos_ for w in doc], model_state["pos_cipher"])
            feature_raw["dep_tags"] = hot_encode([w.dep_ for w in doc], model_state["dep_cipher"])
            feature_raw["da_tags"] = hot_encode(prev_DA, model_state["da_cipher"])
            feature_cooked = svm.features_building(feature_raw,
                                prev_da=model_state["settings"][0],
                                pos_tag=model_state["settings"][1],
                                dep_tag=model_state["settings"][2],
                                context_flag=model_state["settings"][3])
            if type(model_system["mean"]) != int:
                # Mean and std should be computed on the entier input file,
                # but mean and std shouldn't be so different from training set
                feature_cooked = svm.normalize_features(feature_cooked, model_state["mean"],
                                                        model_state["std"])

            pred = model_state["model"].predict(feature_cooked)
            prediction = model_state["number_to_label"][pred[0]]
            results[dialogue_id][turn_id].append({"id": seg["id"],
                                                  "FU": seg["FU"],
                                                  "DA": prediction})
            prev_DA = prediction
            prev_contex = [x for x in feature_raw["we"]]


with open(args.input_file + " (SVM dialogue acts)", "w") as f:
    f.write(json.dumps(results, indent=4, ensure_ascii=False))

print(bcolor.OKGREEN + "Complete!" + bcolor.ENDC)
