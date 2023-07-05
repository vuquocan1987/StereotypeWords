import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
# import logging
import config as cf
# from progressbar import *
from tqdm import tqdm
from collections import defaultdict, Counter

import random
import math
import copy
import scipy.stats as stats
import json
import shap
import os
from transformers import RobertaTokenizerFast
from dataclasses import dataclass, field


@dataclass
class Example:
    text: str
    label: str
    fully_counterfactual_text: list[str] = field(default_factory=list)
    partial_counterfactual_text: list[str] = field(default_factory=list)


class TextDataset():
    def __init__(self, dataset_name):
        cf.random_setting()
        self.dataset_name = dataset_name
        self.train_examples = []
        self.dev_examples = []
        self.test_examples = []
        self.Read_Data()

    # read data
    def Read_from_Datapath(self, data_path):
        """
        Read data from a jsonl file.
        Args:
            data_path: the path of the jsonl file.
        Returns:
            examples: a list of Example objects (The data).
        """
        examples = []
        for line in open(data_path).read().split('\n'):
            if '{' in line:
                linemap = json.loads(line.lower())
                if len(linemap['text'].strip()) > 0 and len(linemap['label'].strip()) > 0:
                    examples.append(Example(linemap['text'], linemap['label']))
        return examples

    # Conform
    def Conform_Dev_Test(self, dev_examples, test_examples):
        """
        Balanced out the classes in dev set and test set.
        Args:
            dev_examples (list): a list of Example objects (The dev set).
            test_examples (list): a list of Example objects (The test set).
        Returns:
            dev_examples_ (list): a list of Example objects for the balance dev set.
            test_examples_ (list): a list of Example objects for the balanced test set.
        """
        label2examples = defaultdict(list)
        for example in dev_examples + test_examples:
            label2examples[example.label].append(example)
        dev_examples_, test_examples_ = [], []
        for subexamples in label2examples.values():
            random.shuffle(subexamples)
            seperator = int(len(subexamples) / 2)
            dev_examples_.extend(subexamples[:seperator])
            test_examples_.extend(subexamples[seperator:])
        return dev_examples_, test_examples_

    # initialize
    def Init_Public(self, train_examples, dev_examples, test_examples):
        examples = train_examples + dev_examples + test_examples

        for i, example in enumerate(examples):
            example.text = example.text.split(' ')
            example.text = [word.strip()
                            for word in example.text if len(word.strip()) > 0]
            example.text = ' '.join(example.text)
            for j in range(len(example.text)):
                example.fully_counterfactual_text.append(cf.Mask_Token)
                example.partial_counterfactual_text.append(cf.Mask_Token)

        cf.XMaxLen = max(examples,key = lambda x:len(x.text))
        cf.XMaxLen = len(cf.XMaxLen.text)
        cf.XMaxLen = min(cf.XMaxLen, cf.XMaxLenLimit)
        self.train_examples = train_examples
        self.dev_examples = dev_examples
        self.test_examples = test_examples

    # initialize
    def Public(self, train):
        model = train.model_x
        stereotype_words = generate_stereotype_words(self.train_examples, model, cf.LOAD_STEREO_TYPE_WORDS_FROM_FILE)
        for i, example in enumerate(self.train_examples + self.dev_examples + self.test_examples):
            example.fully_counterfactual_text = []
            example.partial_counterfactual_text = []
            text = example.text.split(' ')
            for j in range(len(text)):
                word = text[j].strip()
                example.fully_counterfactual_text.append(cf.Mask_Token)
                if word in stereotype_words:
                    example.partial_counterfactual_text.append(word)
                else:
                    example.partial_counterfactual_text.append(cf.Mask_Token)

    

    def Read_Data(self, init_train=False, train=None):

        if init_train == False:
            # output dataset's name
            print('Dataset:{}'.format(self.dataset_name))
            train_datapath = './data/' + self.dataset_name + '.train.jsonl'
            dev_datapath = './data/' + self.dataset_name + '.dev.jsonl'
            test_datapath = './data/' + self.dataset_name + '.test.jsonl'

            train_examples = self.Read_from_Datapath(train_datapath)
            dev_examples = self.Read_from_Datapath(dev_datapath)
            test_examples = self.Read_from_Datapath(test_datapath)

            cf.YList = sorted(
                set(example.label for example in train_examples + dev_examples + test_examples))

            for example in train_examples + dev_examples + test_examples:
                example.label = cf.YList.index(example.label)

            cf.YList = list(range(len(cf.YList)))

            dev_examples, test_examples = self.Conform_Dev_Test(
                dev_examples, test_examples)

            # analysis
            random.shuffle(train_examples)
            random.shuffle(dev_examples)
            random.shuffle(test_examples)
            trLen, deLen, teLen = len(train_examples), len(
                dev_examples), len(test_examples)
            train_examples = train_examples[:min(
                len(train_examples), cf.Train_Example_Num_Control)]
            dev_examples = dev_examples[:min(
                len(dev_examples), int(len(train_examples)*1.0/trLen*deLen))]
            test_examples = test_examples[:min(
                len(test_examples), int(len(train_examples)*1.0/trLen*teLen))]
            trLen, deLen, teLen = len(train_examples), len(
                dev_examples), len(test_examples)
            alLen = trLen + deLen + teLen
            print('#train_examples: {}({:.2%})'.format(
                trLen, trLen * 1.0 / alLen))
            print('#dev_examples: {}({:.2%})'.format(
                deLen, deLen * 1.0 / alLen))
            print('#test_examples: {}({:.2%})'.format(
                teLen, teLen * 1.0 / alLen))

            self.Init_Public(train_examples, dev_examples, test_examples)

            print('cf.XMaxLen={}'.format(cf.XMaxLen))
            print('cf.YList={} {}'.format(len(cf.YList), cf.YList))

            # probability distributions
            train_distribution = cf.Train_Distribution = [
                0 for _ in range(len(cf.YList))]
            dev_distribution = [0 for _ in range(len(cf.YList))]
            test_distribution = [0 for _ in range(len(cf.YList))]
            for e in self.train_examples:
                train_distribution[cf.YList.index(e.label)] += 1
            for e in self.dev_examples:
                dev_distribution[cf.YList.index(e.label)] += 1
            for e in self.test_examples:
                test_distribution[cf.YList.index(e.label)] += 1
            train_distribution = [
                x * 1.0 / sum(train_distribution) for x in train_distribution]
            dev_distribution = [
                x * 1.0 / sum(dev_distribution) for x in dev_distribution]
            test_distribution = [
                x * 1.0 / sum(test_distribution) for x in test_distribution]
            print('train_distribution: [', end='')
            for v in train_distribution:
                print('{:.2%}'.format(v), end=' ')
            print('dev_distribution:   [', end='')
            for v in dev_distribution:
                print('{:.2%}'.format(v), end=' ')
            print(']')
            print('test_distribution:  [', end='')
            for v in test_distribution:
                print('{:.2%}'.format(v), end=' ')
            print(']')

        else:
            self.Public(train)

            # MASK ratio
            examples = self.train_examples + self.dev_examples + self.test_examples
            Ratio = 0.0
            for example in examples:
                up = len(
                    [word for word in example.partial_counterfactual_text if word == cf.Mask_Token])
                down = len(example.text)
                Ratio += up * 1.0 / down
            Ratio = Ratio * 1.0 / len(examples)
            print('{:.2%} MASKed ({:.2%} is context)'.format(Ratio, 1.0 - Ratio))




class TrainDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        index %= self.__len__()

        x = self.examples[index].text
        fcx = self.examples[index].fully_counterfactual_text
        pcx = self.examples[index].partial_counterfactual_text
        fcx = " ".join(fcx)
        pcx = " ".join(pcx)

        y = self.examples[index].label
        y_tensor = self.Generate_Y_Tensor(y)
        return x, fcx, pcx, y,  y_tensor

    def Generate_Y_Tensor(self, label):
        tensor = torch.zeros(len(cf.YList))
        tensor[cf.YList.index(label)] = 1
        tensor = torch.argmax(tensor)
        if cf.Use_GPU == True:
            tensor = tensor.cuda()
        return tensor


def generate_stereotype_words(train_examples=None,model=None, load_from_file=False):


    if cf.Stereotype == 'Imbword':
        return get_positive_shap_words(train_examples, model, load_from_file)
    if cf.Stereotype == 'Keyword':
        return get_low_entropy_words(train_examples, load_from_file)
    if cf.Stereotype == 'Normal':
        # The normal case is just the getting the positive shap words and use them as candidates to find low entropy words among them
        # The assumption is that those words are bias because they have high contribution to the model and having low entropy
        stereotype_words = get_low_entropy_words(train_examples) & get_positive_shap_words(train_examples)
        np.save(cf.Base_Model+cf.Dataset_Name+cf.Stereotype+"low_entropy_positive_shap.npy",list(stereotype_words))
        return stereotype_words
    
def get_low_entropy_words(train_examples,load_from_file=False):
    key_word_file_path = cf.Base_Model+cf.Dataset_Name+cf.Stereotype+"low_entropy.npy"
    if load_from_file:
        return np.load(key_word_file_path, allow_pickle=True)
    classes = set(example.label for example in train_examples)
    
    keyword_entropy = {}
    key_words_class_count = defaultdict(lambda: defaultdict(int))
    for example in train_examples:
        for word in example.text.split(' '):
            key_words_class_count[word][example.label] += 1

    for word in key_words_class_count.keys():
        entropy = 0
        word_class_count = key_words_class_count[word]
        word_sum = sum(list(word_class_count.values()))
        for _class in classes:
            word_class_percentage = word_class_count.get(_class, 0) / word_sum
            entropy -= word_class_percentage * np.log(word_class_percentage + 1e-8)
        keyword_entropy[word] = entropy
    sorted_keyword_entropy = sorted(keyword_entropy.items(), key=lambda x: x[1])
    stereotype_words = [word for word, _ in sorted_keyword_entropy[:int(cf.Alpha*len(sorted_keyword_entropy))+1]]

    print('stereotype_words lengths: ', len(stereotype_words))
    np.save(key_word_file_path, stereotype_words)
    return set(stereotype_words)

def get_positive_shap_words(train_examples = None, model = None,load_from_file=False):
    positive_path_file_path = cf.Base_Model+cf.Dataset_Name+cf.Stereotype+"positive_shap.npy"
    if load_from_file:
        return np.load(positive_path_file_path)
    model.cuda()
    model.eval()
    train_dataset = TrainDataset(train_examples)
    train_loader = DataLoader(train_dataset, batch_size=cf.Train_Batch_Size, shuffle=False)
    important_words = []
    if cf.Base_Model == 'RoBERTa' or cf.Base_Model == 'GPT2':
        roberta_tokenizer = RobertaTokenizerFast.from_pretrained(
                        'roberta-base')
        masker = shap.maskers.Text(roberta_tokenizer)
    elif cf.Base_Model == 'TextCNN' or cf.Base_Model == 'TextRCNN':
        masker = shap.maskers.Text(r"\W")
        cf.Explain = True
    explainer = shap.Explainer(
                    model, masker, output_names=cf.YList, seed=1)
    model.eval()
    for batch_idx, (x, _, _, y, _) in enumerate(train_loader):
        print(batch_idx)
        shap_values = explainer(x)
        for i in range(len(x)):
            shap_value = cf.normalization(
                            shap_values.values[i][:, int(y[i])])
            important_words += list(set([shap_values.data[i][idx].strip() for idx in range(len(shap_value))
                                                    if shap_value[idx] > 0 and len(shap_values.data[i][idx].strip()) > 0]))
    np.save(positive_path_file_path, important_words)
    return set(important_words)