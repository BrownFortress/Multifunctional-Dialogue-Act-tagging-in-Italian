from dataset_manager.data_preprocessing import DataPreprocessing
from dataset_manager.dataset_manager import DatasetManager

from neural_models.cnn_model import DialogueActModelSC
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-dataset_name", help="specify the dataset name")
parser.add_argument("-model_name", help="specify the model name")
args = parser.parse_args()
dataset_name = args.dataset_name
model_name = args.model_name
#model_name = "ilisten_first__enc_128_clip5_epochs_400_tc-model_0.001.pt"
# Load or compute the preprocessing on the specified dataset
data = DataPreprocessing().sequences_preprocessing(dataset_name)

# Window building
# test_flag removes system examples
dataset = DatasetManager().feature_building(data, window_size=5, MAX_LEN=50, test_flag=True)
print("Data loaded")

split_ids = DatasetManager().get_official_split_ids_for_nn(dataset_name, dataset)
split = DatasetManager().split_given_ids_for_nn(dataset, split_ids)

print("Let's test begins")

model_manger = DialogueActModelSC()

number_of_labels = len(data["label_to_number"].keys())
#specify model parameters
#model, checkpoint = model_manger.load_model("models/"+model_name, 128, 128, 300, number_of_labels) # seq2seq case
#model_manger.testing(split["test_set"], model, data, dataset_name, exp_name="soft_attention_w6_best_macro_")
model, checkpoint = model_manger.load_model("models/"+model_name, 200, [1,2,3,4], 128, 300) # cnn_model case
model_manger.testing(split["test_set"], model, data, checkpoint, dataset_name, exp_name="")
