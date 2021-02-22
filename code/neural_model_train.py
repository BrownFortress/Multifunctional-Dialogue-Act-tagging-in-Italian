from dataset_manager.data_preprocessing import DataPreprocessing
from dataset_manager.dataset_manager import DatasetManager
from neural_models.cnn_model import DialogueActModelSC
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-dataset_name", help="specify the dataset name")
parser.add_argument("-device", help="specify the device on which you want to run the NN", default="cuda:0")
args = parser.parse_args()
dataset_name = args.dataset_name
# specify the dataset name
#dataset_name = "ILISTEN_ISO"
# Load or compute the preprocessing on the specified dataset
data = DataPreprocessing().sequences_preprocessing(dataset_name)
# Window building, test flag true means that only the windows for which the last window element
# belongs to the user are kept.
dataset = DatasetManager().feature_building(data, window_size=5, MAX_LEN=50, test_flag=True)
print("Data loaded")

split_ids = DatasetManager().get_official_split_ids_for_nn(dataset_name, dataset)
split = DatasetManager().split_given_ids_for_nn(dataset, split_ids)

print("Let's train begins")


cnn_model = DialogueActModelSC()


cnn_model.manage(split, data, 128, 200, [1,2,3,4], 128, 300, 0.001, n_epochs=300, dataset_name=dataset_name, device=args.device, exp_name="cnn")
