import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

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
    def Public(self, train, train_loader):


        classes = [example.label for example in self.train_examples]
        class_count = Counter(classes)

        model = train.model_x
        important_words = []
        if not cf.Pretrained:
            if cf.Stereotype != 'Imbword':
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
                word_shap = {}

                for batch_idx, (x, fcx, pcx, y, y_tensor) in enumerate(train_loader):
                    # if batch_idx > 0:
                    #     break
                    print(batch_idx)
                    shap_values = explainer(x)
                    for i in range(len(x)):
                        shap_value = cf.normalization(
                            shap_values.values[i][:, int(y[i])])
                        important_words += list(set([shap_values.data[i][idx].strip() for idx in range(len(shap_value))
                                                    if shap_value[idx] > 0 and len(shap_values.data[i][idx].strip()) > 0]))
                cf.Explain = False

            model = model.cuda()

            # important_words = list(word_shap.keys())

            stereotype_words = []
            keyword_entropy = {}
            if cf.Stereotype != 'Keyword':
                
                for keyword in important_words:
                    keyword_class_count = {}
                    for i, example in enumerate(self.train_examples):
                        if keyword not in example.text:
                            pass
                        else:
                            if keyword_class_count.get(example.label) == None:
                                keyword_class_count[example.label] = 1
                            else:
                                keyword_class_count[example.label] += 1
                    keyword_class_percentage = {}
                    keyword_sum = sum(list(keyword_class_count.values()))
                    entropy = 0
                    for _class in class_count.keys():
                        keyword_class_percentage[_class] = keyword_class_count.get(
                            _class, 0) / keyword_sum
                        entropy -= keyword_class_percentage[_class] * np.log(
                            keyword_class_percentage[_class] + 1e-8)

                    keyword_entropy[keyword] = entropy

                stereotype_word_list = list(keyword_entropy.values())
                stereotype_word_list.sort(reverse=True)
                boundary = stereotype_word_list[int(
                    len(stereotype_word_list) * cf.Alpha)]

                for keyword in list(keyword_entropy.keys()):

                    if keyword_entropy[keyword] >= boundary:
                        print(keyword)
                        stereotype_words.append(keyword)
                print('stereotype_words lengths: ', len(stereotype_words))
        else:
            important_words = np.load(
                cf.Base_Model + cf.Dataset_Name + cf.Stereotype + 'keyword.npy')
            stereotype_words = np.load(
                cf.Base_Model + cf.Dataset_Name + cf.Stereotype + 'imbword.npy')

        for i, example in enumerate(self.train_examples + self.dev_examples + self.test_examples):
            example.fully_counterfactual_text = []
            example.partial_counterfactual_text = []
            text = example.text.split(' ')
            for j in range(len(text)):
                word = text[j].strip()
                example.fully_counterfactual_text.append(cf.Mask_Token)
                if word not in important_words or word not in stereotype_words:
                    example.partial_counterfactual_text.append(cf.Mask_Token)
                else:
                    example.partial_counterfactual_text.append(word)

        if not cf.Pretrained:
            np.save(cf.Base_Model + cf.Dataset_Name +
                    cf.Stereotype + 'keyword.npy', important_words)
            np.save(cf.Base_Model + cf.Dataset_Name +
                    cf.Stereotype + 'imbword.npy', stereotype_words)

    def Read_Data(self, init_train=False, train=None, train_loader=None):

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
            self.Public(train, train_loader)

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

    def Word_Detection(self, train_loader, train):

        self.Read_Data(True, train, train_loader)


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
