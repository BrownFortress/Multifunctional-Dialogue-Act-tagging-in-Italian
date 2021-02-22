# Dialogue act tagger

This repository contains a dialogue act tagger based on SVM and CNN model.

### Requirements
- python >= 3.6
- sklearn >= 0.22
- spacy >= 2.2.4
 - spacy Italian model (python -m spacy download it_core_news_sm)
- torch >= 1.5.0
- FastText
  - FastText pretrained model (https://fasttext.cc/docs/en/pretrained-vectors.html)
  - Place the *wiki.it.bin* file inside the **preprocessing** folder
- Plotly for charts

### Getting Started
#### Get easily paper results  
The models used to produce the results showed in the paper are saved in *models*. Also the preprocessing of the datasets are provided, thus for these experiments you do not need FastText *wiki.it.bin*. To run them and get the results, firstly get **iLISTEN** and **iLISTEN2ISO** datasets and copy them into the *datasets* folder. Then secondly run the .sh files (or copy&paste on a terminal their content):
- **CNN models**
  - cnn_iLISTEN.sh
  - cnn_iLISTEN2ISO.sh
- **SVM models**
  - svm_iLISTEN.sh
  - svm_iLISTEN2ISO.sh

#### Set up datasets
The first step is to copy in the "datasets/" folder the "iLISTEN.json" and "iLISTEN2ISO.json" datasets that can be obtained in the "resource/" folder.
#### SVM model
To train an SVM model run **svm_train.py**:
```bash
python svm_train.py -dataset_name={dataset name such as iLISTEN2ISO} -speaker_to_keep= {U for user; S for system}
```
If **-speaker_to_keep** is not specified the program will train two models for both U and S speakers.
To test the trained models run **svm_test.py**:
```bash
python svm_test.py -dataset_name={dataset name such as iLISTEN2ISO} -model_system=path/to/model/trained/on/system/turns -model_user=path/to/model/trained/on/user/turns
```
The outcoming results are saved in /results folder, while the relative models are saved in /models folder. To analyze these results run:
```bash
python compute_error_analysis.py -model={svm or cnn_model} -dataset_name={for instance iLISTEN2ISO}
```
#### CNN model
To train the CNN model run:
```bash
python neural_model_train.py -dataset_name={ilisten or iLISTEN2ISO}
```
To test the CNN mode run:
```bash
python neural_model_test.py -dataset_name={ilisten or iLISTEN2ISO} -model_name=path/to/model/
```
The outcoming results are saved in /results folder, while the relative models are saved in /models folder. To analyze these results run:
```bash
python compute_error_analysis.py -model={svm or cnn_model} -dataset_name={for instance iLISTEN2ISO}
```
### Dataset
The training datasets must be stored in /datasets folder. The files must be in JSON format, using the
following schema:
```json
{
    "dialogue_id": {
        "turn_id": [
            {
                "id": 0,
                "FU": "ciao , il mio nome e' valentina .",
                "DA": "other",
                "speaker": "S"
            },
            {
                "id": 1,
                "FU": "sono qui per darti dei suggerimenti su come migliorare la tua dieta .",
                "DA": "inform",
                "speaker": "S"
            }
        ],
        "turn_1": []
      }
}
```

*dialogue_id* identifies uniquely a dialogue. A dialogue is composed of a set of turns identified with an id that must be unique in the dialogue. A turn is composed of a list of functional units(1 or more). The attributes of a functional unit are:
- id: functional unit id.
- FU: the portion of utterance that characterizes that FU.
- DA: the relative dialogue act.
- speaker: the speaker that uttered that turn.

### Ready to use scripts
The scripts used to run training models are: **svm_get_da.py** . The input parameters are:
```bash
python svm_get_da.py -input_file= -model_name=
```

Regarding **svm_get_da.py** the input file must be written in the same format of dataset file, where the "DA" field is ignored in this case.
