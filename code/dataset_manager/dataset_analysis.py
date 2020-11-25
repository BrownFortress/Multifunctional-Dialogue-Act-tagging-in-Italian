from collections import Counter
import plotly.graph_objects as go
import os
import spacy
from itertools import groupby
import numpy as np
from dataset_manager.data_preprocessing import DataPreprocessing

class DatasetAnalysis():
    def __init__(self):
        pass
    # It returns an dictionary where the keys are words and values are frequencies
    def word_count(self, utterances):
        nlp = spacy.load("it_core_news_sm", disable = ["tagger", "parser", "ner", "textcat",  "entity_linker", "sentecizer"])
        list_utterances = []
        st_list_utterances = []
        for utterance in utterances:
            seq = []
            for doc in nlp(utterance):
                if not doc.is_stop and not doc.is_punct and len(doc.text) > 2 and "filler" not in doc.text.lower() and "fil" != doc.text and "sil" != doc.text:
                    seq.append(doc.text.lower())
            st_list_utterances.append(list(filter(None, utterance.split(" "))))
            list_utterances.append(seq)

        count = Counter(sum(list_utterances,[]))
        st_words_count = Counter(sum(st_list_utterances, [])) # Word count with stop words

        return count, st_words_count
    def dialogue_length(self, dialogue_ids, segments):
        used = []
        unique_dialogue_ids = [x for x in dialogue_ids if x not in used and (used.append(x) or True)]
        number_of_dialogues = len(unique_dialogue_ids)
        prev_d = dialogue_ids[0]
        trace = {}
        n_turns = 0
        for id_d, d in enumerate(dialogue_ids):
            if d not in trace.keys():
                trace[d] = []
            if segments[id_d] not in trace[d]:
                trace[d].append(segments[id_d])
        for v in trace.values():
            n_turns += len(v)
        mean_dialogue_length = n_turns / len(unique_dialogue_ids)
        return number_of_dialogues, mean_dialogue_length, n_turns
    def most_common_dialogue_acts(self, dataset_name, coverage):
        dataset = DataPreprocessing().load_data(dataset_name)
        dialogue_acts = []
        for dia_id, dialogue in dataset.items():
            for turn_id, turn in dialogue.items():
                for seg in turn:
                    if seg["speaker"] != "S":
                        dialogue_acts.append(seg["DA"])
        count = self.order_dict(self.get_the_percentage(dialogue_acts, dialogue_acts))
        threshold = 0
        if coverage < 1:
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

    def bar_chart(self, name_serie, title, x_caption, y_caption, tags, values, color, text_flag= False):
        colors_a = ['#46e500', '#eb7070','#263859', '#1089ff']
        colors_b = ['#f7be16', '#64e291','#ff6768','#ff6363', '#eb7070', '#1768bf', '#27c408']
        text_to_disaplay = None
        if text_flag:
            text_to_disaplay = values
        fig = go.Figure(data=[
            go.Bar(name=name_serie, x=tags, y=values, text=text_to_disaplay,
                textposition='auto', marker_color=colors_b[color]),
        ])
        # Change the bar mode
        fig.update_layout(
            title=title,
            xaxis_title=x_caption,
            yaxis_title=y_caption,

            font=dict(
                family="Droid Sans",
                size=30,
                color="#7f7f7f"
            )
            )
        #fig.show()
        if not os.path.exists("charts/"):
            os.mkdir("charts")
        print("Chart is saved in charts/ folder")
        fig.write_image("charts/"+ title + ".png", format="png", width=1920, height=1080, scale=1)

    def line_chart(self, name_serie, title, x_caption, y_caption, tags, values, color):
        colors_a = ['#46e500', '#eb7070','#263859', '#1089ff']
        colors_b = ['#f7be16', '#64e291','#ff6768','#ff6363', '#eb7070', '#1768bf', '#27c408']

        fig = go.Figure(data=[
            go.Scatter(name=name_serie, x=tags, y=values, line = dict(color=colors_b[color], width=6), marker=dict(size=12),mode='lines+markers'),
        ])
        # Change the bar mode
        fig.update_layout(
            title=title,
            xaxis_title=x_caption,
            yaxis_title=y_caption,
            font=dict(
                family="Droid Sans",
                size=36,
                color="#7f7f7f"
            )
            )
        #fig.show()
        if not os.path.exists("charts/"):
            os.mkdir("charts")
        print("Chart is saved in charts/ folder")
        fig.write_image("charts/"+ title + ".png", format="png", width=1920, height=1080, scale=1)
    def pie_chart(self, title, tags, values):
        fig = go.Figure(data=[go.Pie(labels=tags, values=values, hole=.3)])
        # Change the bar mode
        fig.update_layout(
            title=title,
            font=dict(
                family="Droid Sans",
                size=36,
                color="#7f7f7f"
            )
            )
        #fig.show()
        if not os.path.exists("charts/"):
            os.mkdir("charts")
        print("Chart is saved in charts/ folder")
        fig.write_image("charts/"+ title + ".png", format="png", width=1920, height=1080, scale=1)

    def order_dict(self, dictionary, reverse =True):
        return {k: v for k, v in sorted(dictionary.items(), key=lambda item: item[1], reverse = reverse)}
    def order_dict_by_key(self, dictionary, reverse =True):
        return {k: v for k, v in sorted(dictionary.items(), key=lambda item: item[0], reverse = reverse)}

    def count(self):
        pass
    # Counter is a dictionary key: frequency of key
    # partition is a subset or a proper set of counter's keys
    # for which the percentage is computed
    def percentage_style(self, counter, partition):
        common_divisor = sum(counter.values())
        result = {}
        for p in set(partition):
            result[p] = (counter[p] / common_divisor) * 100
        return result

    def get_the_percentage(self, data_to_percentage, all_data, cut_off=0):
        common_divisor = sum(Counter(all_data).values())
        dtp = data_to_percentage
        if type(data_to_percentage) != (dict):
            dtp = Counter(data_to_percentage)
        res = {}
        for k,v in dtp.items():
            if (v / float(common_divisor) * 100) > cut_off:
                res[k] = round(v /common_divisor, 3) * 100
        return res
    def print_heatmap(self,confusion_matrix, name):
        matrix_number  = []
        xy_labels = []
        keys = list(confusion_matrix.keys())

        for k, v in confusion_matrix.items():
            if len(keys) != len(v.values()):
                sub_list = [0]*len(keys)
                for s_k, s_v in v.items():
                    if s_k in keys:
                        sub_list[keys.index(s_k)] = s_v
                matrix_number.append(sub_list)
            else:
                matrix_number.append(list(v.values()))
            xy_labels.append(k)
        print(xy_labels)
        m = np.asarray(matrix_number).transpose()
        fig = go.Figure(data=go.Heatmap(
                           z=m,
                           x=xy_labels,
                           y=xy_labels
                           ))
        fig.update_layout(
            font=dict(
                size=28,
            ))
        fig.write_image("charts/heatmap-" + name +".png", format="png", width=1920, height=1920, scale=1)
