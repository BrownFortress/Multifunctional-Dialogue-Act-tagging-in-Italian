import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
from tqdm import tqdm
from dataset_manager.dataset_manager import DatasetManager
from dataset_manager.dataset_analysis import DatasetAnalysis
from score_manager.score_manager import ScoreManager
from score_manager.error_analysis import ErrorAnalysis
import pickle
import os
import random
import sys
import time
import json

# Sentence encoder using CNN
class EncoderCNN(nn.Module):
    def __init__(self, embedding_dim, n_filters, filter_sizes, output_dim,
                 dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1,
                                              out_channels = n_filters,
                                              kernel_size = (fs, embedding_dim))
                                    for fs in filter_sizes
                                    ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        #self.output_dim = output_dim
        self.output_dim = len(filter_sizes) * n_filters
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        input = input.permute(1,0,2)
        #input = [batch size, sent len, emb dim]
        input = input.unsqueeze(1)
        #embedded = [batch size, 1, sent len, emb dim]
        #conved = [F.relu(conv(input)).squeeze(3) for conv in self.convs]
        # The convolution is performed
        conved = [torch.relu(conv(input)).squeeze(3) for conv in self.convs]
        #conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
        # The pooling is applied
        pooled = [torch.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        #pooled_n = [batch size, n_filters]
        cat = self.dropout(torch.cat(pooled, dim = 1))
        #cat = [batch size, n_filters * len(filter_sizes)]
        #return self.fc(cat)
        return cat


class BiGRUFusion(nn.Module):
    def __init__(self, first_layer_encoder, input_encoder, output_size, system_output_size, device):
        super(BiGRUFusion, self).__init__()
        self.encoder_layer1 = first_layer_encoder
        self.input_encoder = input_encoder
        self.device = device
        self.output_size = output_size

        self.linear_layer = nn.Linear(first_layer_encoder.output_dim , first_layer_encoder.output_dim)
        #self.out_layer = nn.Linear(first_layer_encoder.output_dim, output_size)
        self.out_layer = nn.Linear(input_encoder.output_dim + 30, output_size)
        self.system_predictor = nn.Linear(self.encoder_layer1.output_dim, system_output_size)
        self.embedding = nn.Embedding(20, 30)
        self.softmax = nn.Softmax(dim=1)
        #self.softmax =
        self.dropout = nn.Dropout(0.5)

    def forward(self, src, trg, speakers, lengths, teacher_forcing_ratio=0.5):
        # Src  [window size, max setentence lenght, batch_size, embedding dim]
        # trg: [window size, batch_size]
        batch_size = src.shape[2]
        trg_len = trg.shape[0]
        trg_vocab_size = self.output_size
        #tensor to store decoder outputs

        # Window encoding
        to_classify = self.input_encoder(src[-2])
        system_enc = self.encoder_layer1(src[-3])
        sys_da_prediction = torch.softmax(self.system_predictor(system_enc), dim=1)
        pred_sys_da = sys_da_prediction.argmax(1)
        prev_da = self.dropout(self.embedding(pred_sys_da))
        features = torch.cat((to_classify, prev_da), dim=1)
        out = self.out_layer(self.dropout(features))
        #output = out
        output = self.softmax(out)
        return output, sys_da_prediction

def init_weights(mat):
    for m in mat.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.xavier_uniform_(param[idx*mul:(idx+1)*mul])
                elif 'weight_hh' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.orthogonal_(param[idx*mul:(idx+1)*mul])
                elif 'bias' in name:
                    param.data.fill_(0)
        else:
            if type(m) in [nn.Linear]:
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias != None:
                    m.bias.data.fill_(0.01)

class DialogueActModelSC():
    def __init__(self):
        pass

    def manage(self, dataset, data, batch_size, number_of_filters, filter_sizes, output_size,
        input_size, lr, n_epochs=200, dataset_name="", model_to_load=None, device="cuda:1", exp_name=""):
        N_EPOCHS = n_epochs
        #number_of_labels = len(data["label_to_number"].keys())
        number_to_label = {v:k for k, v in data["label_to_number"].items()}
        mapping = {}
        sys_mapping = {}
        id_label = 0
        id_label_sys = 0
        for k, v in data["label_to_number"].items():
            if k not in ["pad", "sos", "eos"]:
                if "U" in data["label_owner"][k]:
                    mapping[v] = id_label
                    id_label += 1
                if "S" in data["label_owner"][k] or "iso" in dataset_name.lower():
                    sys_mapping[v] = id_label_sys
                    id_label_sys += 1
        number_of_labels = len(mapping.keys())
        sys_number_of_labels = len(sys_mapping.keys())


        encoder = EncoderCNN(input_size, number_of_filters, filter_sizes, output_size*2)
        input_encoder = EncoderCNN(input_size, number_of_filters, filter_sizes, output_size*2)
        model = BiGRUFusion(encoder, input_encoder, number_of_labels, sys_number_of_labels, device).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        if model_to_load == None:
            model.apply(init_weights)
        else:
            checkpoint = torch.load(model_to_load, map_location=device)
            N_EPOCHS = N_EPOCHS - checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])

        best_valid_accuracy = 0
        best_valid_F1 = 0
        print("Train is started")

        all_losses = []
        accuracies_train = []
        accuracies_test = []
        f1s_test = []
        f1s_train = []
        #print(model)
        data_name = dataset_name.split(".")[0]

        exp_name +=  "_epochs_" + str(n_epochs)
        mean_losses_train = []
        mean_losses_test = []
        for epoch in tqdm(range(N_EPOCHS)):
            # Gets batches return a list of batches, while padding batches add pads to windows in that batch. The padding in this phased is applied at token level only.
            train_iterator = DatasetManager().padding_batches_of_windows(self.get_batches(dataset["train_set"], batch_size), data)
            test_iterator = DatasetManager().padding_batches_of_windows(self.get_test_batches(dataset["validation_set"], batch_size), data)
            accuracy_train, f1_train, losses_train, ground_true_train, predictions_train = self.train(model, train_iterator, optimizer, criterion, device, mapping, sys_mapping)
            accuracy_test, f1_test, losses_test, ground_true_test, predictions_test = self.evaluate(model, test_iterator, criterion,  device, mapping, sys_mapping)
            '''
            if accuracy_test > best_valid_accuracy: # If the model achieved the best f1 score than it is saved.
                best_valid_accuracy = accuracy_test
                model.train()
                state = {
                        'epoch': epoch + 1, 'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'mapping': mapping,
                        'sys_mapping': sys_mapping,
                        'number_to_label': number_to_label
                        }
                torch.save(state, "models/"+dataset_name+ "_" + exp_name +"_cnn_model-model_"+str(lr)+".pt")
            '''
            if f1_train > best_valid_F1: # If the model achieved the best f1 score than it is saved.
                best_valid_F1 = f1_train
                model.train()
                state = {
                        'epoch': epoch + 1, 'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'mapping': mapping,
                        'sys_mapping': sys_mapping,
                        'number_to_label': number_to_label
                        }

                torch.save(state, "models/" + data_name + "_" + exp_name +"_cnn_"+str(lr)+".pt")
            #all_losses.append(np.asarray(losses).mean())

            accuracies_train.append(accuracy_train)
            accuracies_test.append(accuracy_test)
            f1s_test.append(f1_test)
            f1s_train.append(f1_train)
            print("\nAccuracy train: ", accuracy_train * 100 , " F1:", f1_train * 100, "\n")
            print("Accuracy valid: ", accuracy_test * 100, " F1:", f1_test * 100, "\n")

            mean_losses_train.append(sum(losses_train)/len(losses_train))
            mean_losses_test.append(sum(losses_test)/len(losses_test))
            ScoreManager().save_performances("fast_text", dataset_name, "cnn_model", exp_name + "_" + str(lr),
                accuracies_train, accuracies_test, f1s_train, f1s_test, mean_losses_train, mean_losses_test)




    def get_batches(self, dataset, batch_size):
        indexes = random.sample(range(0,len(dataset["examples"])), len(dataset["examples"]))
        batches = []
        start = 0
        end = batch_size
        for x in range(int(len(dataset["examples"])/batch_size)):
            src = [dataset["examples"][ids] for ids in indexes[start:end]]
            trg = [dataset["labels"][ids] for ids in indexes[start:end]]
            speakers = [dataset["speakers"][ids] for ids in indexes[start:end]]
            pos_tags = [dataset["pos_tags"][ids] for ids in indexes[start:end]]
            start = end
            end += batch_size
            batches.append({"examples":src, "labels":trg, "speakers":speakers, "pos_tags":pos_tags})
        if len(dataset["examples"]) % batch_size > 0:
            src = [dataset["examples"][ids] for ids in indexes[start:end]]
            trg = [dataset["labels"][ids] for ids in indexes[start:end]]
            speakers = [dataset["speakers"][ids] for ids in indexes[start:end]]
            pos_tags = [dataset["pos_tags"][ids] for ids in indexes[start:end]]
            missing = batch_size - (len(dataset["examples"]) % batch_size)
            for i in range(0, missing):
                random_pos = random.randint(0, len(dataset["examples"])-1)
                src.append(dataset["examples"][random_pos])
                trg.append(dataset["labels"][random_pos])
                speakers.append(dataset["speakers"][random_pos])
                pos_tags.append(dataset["pos_tags"][random_pos])

            batches.append({"examples":src, "labels":trg, "speakers":speakers, "pos_tags": pos_tags})
        return batches
        #DatasetAnalysis().line_chart("loss", "Loss total 2" + str(epoch), "epochs", "loss", list(range(1, len(all_losses))), all_losses, 2)
    def get_test_batches(self, dataset, batch_size):
        batches = []
        start = 0
        end = batch_size
        for x in range(int(len(dataset["examples"])/batch_size)):
            src = dataset["examples"][start:end]
            trg = dataset["labels"][start:end]
            speakers = [dataset["speakers"][ids] for ids in range(start,end)]
            pos_tags = dataset["pos_tags"][start:end]
            start = end
            end += batch_size
            batches.append({"examples":src, "labels":trg, "speakers":speakers, "pos_tags": pos_tags})
        src = []
        trg = []
        speakers = []
        if len(dataset["examples"]) % batch_size > 0:
            src = [dataset["examples"][ids] for ids in range(start, len(dataset["examples"]))]
            trg = [dataset["labels"][ids] for ids in range(start, len(dataset["examples"]))]
            speakers = [dataset["speakers"][ids] for ids in range(start, len(dataset["examples"]))]
            pos_tags = [dataset["pos_tags"][ids] for ids in range(start, len(dataset["examples"]))]
            batches.append({"examples":src, "labels":trg, "speakers":speakers, "pos_tags": pos_tags})
        return batches
    def train(self, model, iterator, optimizer, criterion,device, mapping, sys_mapping):
        model.train()
        epoch_loss = 0
        n_samples = len(iterator) * len(iterator[0]["examples"])
        correct = 0
        losses = []
        predictions = []
        ground_true = []
        for i, batch in enumerate(iterator):
            '''
                Window: sos, S_DA1, S_DA2, S_DAn, eos
                n = window length
            '''
            src = torch.tensor(batch["examples"], dtype=torch.float).permute(1,2,0,3).to(device)
            # Src  [window size, max setentence lenght, batch_size, embedding dim]
            trg = torch.tensor(batch["labels"], dtype=torch.long).permute(1,0).to(device)
            # Trg [window size, batch_size]
            speakers = torch.tensor(batch["speakers"], dtype=torch.long).permute(1,0).to(device)
            # speakers [window size, batch_size]
            lengths = torch.tensor(batch["lengths"], dtype=torch.long).permute(1,0).to(device)
            # Utterance lengths [window size, batch_size]
            optimizer.zero_grad()
            output, sys_prediction = model(src, trg, speakers, lengths)

            #trg = [trg len, batch size]
            #output = [trg len, batch size, output dim]

            output_dim = output.shape[-1]
            output_tmp = output.view(-1, output_dim)
            sys_output_dim = sys_prediction.shape[-1]
            sys_out_tmp =  sys_prediction.view(-1, sys_output_dim)
            sys_trg = trg[-3].reshape(-1)
            usr_trg = trg[-2].reshape(-1)

            trg_tmp = torch.zeros(trg.shape[1], dtype=torch.long).to(device)
            for id_t, t in enumerate(usr_trg):
                trg_tmp[id_t] = mapping[t.item()]

            sys_trg_tmp = torch.zeros(trg.shape[1], dtype=torch.long).to(device)
            for id_t, t in enumerate(sys_trg):
                sys_trg_tmp[id_t] = sys_mapping[t.item()]
            #trg = [(trg len - 1) * batch size]
            #output = [(trg len - 1) * batch size, output dim]
            # Accuracy is measured only on the last example, so outputs[-2] beacuse at -1 there is the eos token

            loss1 = criterion(output_tmp, trg_tmp)
            loss2 = criterion(sys_out_tmp, sys_trg_tmp)
            loss = (loss1 + loss2)/2

            losses.append(loss.item())
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), clip) possible bug clip was =2
            optimizer.step()
            # Metrics are computed only on the last window element.
            predicted = output.argmax(1)

            predictions.extend(predicted.cpu().detach().numpy().tolist())
            ground_true.extend(trg_tmp.cpu().detach().numpy().tolist())
            correct += float((predicted == trg_tmp).sum())


        eval = ErrorAnalysis()
        code = eval.evaluation({"ground_true": ground_true, "predictions": predictions})
        accuracy, f1 = eval.compute_scores(code)
        return accuracy, f1, losses, ground_true, predictions

    def load_model(self, model_path, filter_dim, filter_sizes, output_size,
        input_size):
        checkpoint = torch.load(model_path, map_location="cuda:0")
        number_of_labels = len(checkpoint["mapping"].keys())
        sys_number_of_labels = len(checkpoint["sys_mapping"].keys())
        encoder = EncoderCNN(input_size, filter_dim, filter_sizes, output_size*2)
        input_encoder = EncoderCNN(input_size, filter_dim, filter_sizes, output_size*2)
        model = BiGRUFusion(encoder, input_encoder, number_of_labels, sys_number_of_labels, "cuda:0").to("cuda:0")
        to_load = {}
        for name, prm in checkpoint['state_dict'].items():
            if name in model.state_dict():
                to_load[name] = prm

        model.load_state_dict(to_load)
        return model, checkpoint

    def testing(self, test_set, model, data, checkpoint, dataset_name, exp_name=None ):
        iterator =  DatasetManager().padding_batches_of_windows(self.get_test_batches(test_set, 128), data)
        number_to_label = checkpoint["number_to_label"]
        reversed_mapping = {v:k for k, v in checkpoint["mapping"].items()}
        model.eval()
        predictions = []
        ground_true = []
        device = "cuda:0"
        with torch.no_grad():
            for i, batch in enumerate(iterator):
                src = torch.tensor(batch["examples"], dtype=torch.float).permute(1,2,0,3).to(device)
                trg = torch.tensor(batch["labels"], dtype=torch.long).permute(1,0).to(device)
                speakers = torch.tensor(batch["speakers"], dtype=torch.long).permute(1,0).to(device)
                lengths = torch.tensor(batch["lengths"], dtype=torch.long).permute(1,0).to(device)
                output, sys_output = model(src, trg, speakers, lengths) #turn off teacher forcing
                output_dim = output.shape[-1]
                output_tmp = output.view(-1, output_dim)
                trg_tmp = trg[-2].reshape(-1)
                predicted = output.argmax(1)
                predicted_list = [reversed_mapping[p] for p in predicted.cpu().detach().numpy().tolist()]
                predictions.extend(predicted_list)
                ground_true.extend(trg[-2].cpu().detach().numpy().tolist())
        assert len(predictions) == len(ground_true)
        predictions = [number_to_label[l] for l in predictions]
        ground_true = [number_to_label[l] for l in ground_true]
        ScoreManager().save_results("fast_text", dataset_name, "cnn", "test_set_"+ exp_name + ".json", ground_true, predictions)

    def evaluate(self, model, iterator, criterion, device, mapping, sys_mapping):
        model.eval()
        epoch_loss = 0
        n_samples = len(iterator) * len(iterator[0]["examples"])
        correct = 0
        predictions = []
        ground_true = []
        losses = []
        with torch.no_grad():
            for i, batch in enumerate(iterator):
                src = torch.tensor(batch["examples"], dtype=torch.float).permute(1,2,0,3).to(device)
                trg = torch.tensor(batch["labels"], dtype=torch.long).permute(1,0).to(device)
                speakers = torch.tensor(batch["speakers"], dtype=torch.long).permute(1,0).to(device)
                lengths = torch.tensor(batch["lengths"], dtype=torch.long).permute(1,0).to(device)
                output, sys_prediction = model(src, trg, speakers, lengths)

                #trg = [trg len, batch size]
                #output = [trg len, batch size, output dim]

                output_dim = output.shape[-1]
                output_tmp = output.view(-1, output_dim)
                sys_output_dim = sys_prediction.shape[-1]
                sys_out_tmp =  sys_prediction.view(-1, sys_output_dim)
                sys_trg = trg[-3].reshape(-1)
                usr_trg = trg[-2].reshape(-1)

                trg_tmp = torch.zeros(trg.shape[1], dtype=torch.long).to(device)
                for id_t, t in enumerate(usr_trg):
                    trg_tmp[id_t] = mapping[t.item()]

                sys_trg_tmp = torch.zeros(trg.shape[1], dtype=torch.long).to(device)
                for id_t, t in enumerate(sys_trg):
                    sys_trg_tmp[id_t] = sys_mapping[t.item()]
                #trg = [(trg len - 1) * batch size]
                #output = [(trg len - 1) * batch size, output dim]
                # Accuracy is measured only on the last example, so outputs[-2] beacuse at -1 there is the eos token

                loss1 = criterion(output_tmp, trg_tmp)
                loss2 = criterion(sys_out_tmp, sys_trg_tmp)
                loss = (loss1 + loss2)/2
                losses.append(loss.item())
                predicted = output.argmax(1)
                predictions.extend(predicted.cpu().detach().numpy().tolist())
                ground_true.extend(trg_tmp.cpu().detach().numpy().tolist())
                correct += float((predicted == trg_tmp).sum())
        eval = ErrorAnalysis()
        code = eval.evaluation({"ground_true": ground_true, "predictions": predictions})
        accuracy, f1 = eval.compute_scores(code)
        return accuracy, f1, losses, ground_true, predictions
