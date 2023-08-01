from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
# import logging
import config as cf
from config import StereoType
# from progressbar import *
from tqdm import tqdm
from collections import defaultdict, Counter
import re
import spacy
from idiomatch import Idiomatcher
import nltk
from nltk.corpus import wordnet
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('words')
nltk.download('omw-1.4')

import random
import math
import copy
import scipy.stats as stats
import json
import shap
import os
from transformers import RobertaTokenizerFast
from dataclasses import dataclass, field, asdict



@dataclass
class Example:
    text: str
    label: str
    fully_counterfactual_text: str = ""
    partial_counterfactual_text: str = ""

class ExampleEncoder(json.JSONEncoder):
    def __init__(self, subset,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.subset = subset
    def default(self, o: Any) -> Any:
        d = asdict(o)
        d['subset'] = self.subset
        return d

class TextDataset():
    def __init__(self, dataset_name):
        cf.random_setting()
        self.dataset_name = dataset_name
        self.train_examples = []
        self.dev_examples = []
        self.test_examples = []
        self.masked_ratio = -1.0
        self.prepare_data()
        

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

        cf.XMaxLen = max(examples,key = lambda x:len(x.text))
        cf.XMaxLen = len(cf.XMaxLen.text)
        cf.XMaxLen = min(cf.XMaxLen, cf.XMaxLenLimit)
        self.train_examples = train_examples
        self.dev_examples = dev_examples
        self.test_examples = test_examples

    # initialize dataset i.e. masking words in the text
    def Public(self, train=None, n_gram=cf.N_GRAM):
        if cf.Stereotype == StereoType.Idiom:
            self.mask_idiom()
            return
        if train is not None:
            model = train.model_x
        else:
            model = None
        
        stereotype_words = generate_stereotype_words(self.train_examples, model, cf.LOAD_STEREO_TYPE_WORDS_FROM_FILE, n_gram = n_gram)
        for i, example in enumerate(self.train_examples + self.dev_examples + self.test_examples):
            # example.fully_counterfactual_text = []
            # example.partial_counterfactual_text = []
            len_text = len(example.text.split())
            example.fully_counterfactual_text = ' '.join([cf.Mask_Token] * len_text)
            partial_counterfactual_text = [cf.Mask_Token] * len_text
            for index, ngram in enumerate(split_into_ngrams(example.text, n_gram)):
                if ngram in stereotype_words:
                    partial_counterfactual_text[index:index+n_gram] = ngram.split()
            example.partial_counterfactual_text = ' '.join(partial_counterfactual_text)
    def mask_idiom(self):
        nlp = spacy.load('en_core_web_sm')
        idiomatcher = Idiomatcher.from_pretrained(nlp)
        for i, example in tqdm(enumerate(self.train_examples + self.dev_examples + self.test_examples)):
            len_text = len(example.text.split())
            example.fully_counterfactual_text = ' '.join([cf.Mask_Token] * len_text)
            partial_counterfactual_text = [cf.Mask_Token] * len_text
            doc = nlp(example.text)
            for idiom in idiomatcher.identify(doc):
                partial_counterfactual_text[idiom['meta'][1]:idiom['meta'][2]] = idiom['span'].split()
            example.partial_counterfactual_text = ' '.join(partial_counterfactual_text)
            
    def prepare_data(self, init_train=False, train=None,n_gram = cf.N_GRAM):

        if init_train == False:
            # output dataset's name
            print('Dataset:{}'.format(self.dataset_name))
            train_datapath = cf.DATA_PATH + self.dataset_name + '.train.jsonl'
            dev_datapath = cf.DATA_PATH + self.dataset_name + '.dev.jsonl'
            test_datapath = cf.DATA_PATH + self.dataset_name + '.test.jsonl'

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
            self.Public(train,n_gram)

            # MASK ratio
            examples = self.train_examples + self.dev_examples + self.test_examples
            Ratio = 0.0
            for example in examples:
                up = len(
                    [word for word in example.partial_counterfactual_text.split() if word == cf.Mask_Token])
                down = len(example.text)
                Ratio += up * 1.0 / down
            Ratio = Ratio * 1.0 / len(examples)
            self.masked_ratio = Ratio
            print('{:.2%} MASKed ({:.2%} is context)'.format(Ratio, 1.0 - Ratio))
    def write_masked_data_to_disk(self):
        #dumping examples to json file, with a new column indicate which set it belongs to
        path = cf.DATA_PATH + "/masked_dataset/" + self.dataset_name + 'processed.json'

        with open(path, 'w') as f:
            
            write_example_data_to_disk(self.train_examples,"train",self.masked_ratio)
            write_example_data_to_disk(self.dev_examples,"dev",self.masked_ratio)
            write_example_data_to_disk(self.test_examples,"test",self.masked_ratio)



def write_example_data_to_disk(examples, subset, masked_ratio):
    path = cf.DATA_PATH + "/masked_dataset/maksed_data.json"
    with open(path, 'a') as f:
        for example in examples:
            d = asdict(example)
            d["Dataset"] = cf.Dataset_Name
            d["Base_Model"] = cf.Base_Model
            d["Mask_Ratio"] = masked_ratio
            if cf.N_GRAM == 2:
                d["Stereo_Type"] = "Bi-gram"
            else:
                d["Stereo_Type"] = cf.Stereotype.name
            d["Subset"] = subset
            json.dump(d, f)
            f.write('\n')

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
        # fcx = " ".join(fcx)
        # pcx = " ".join(pcx)

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


def generate_stereotype_words(train_examples,model=None, load_from_file=False,n_gram = 1):
    if cf.Stereotype == StereoType.Noun:
        words = {word for example in train_examples for word in example.text.split()}
        nouns = set(nltk.corpus.words.words())
        # Filter for nouns
        nouns = [word for word in nouns if wordnet.synsets(word) and wordnet.synsets(word)[0].pos() == 'n']
        return set(nouns) & words
    if cf.Stereotype == StereoType.RandomMask:
        words = {word for example in train_examples for word in example.text.split()}
        words = random.sample(sorted(words), int(cf.RANDOM_MASK_RATE * len(words)))
        return set(words)
    if cf.Stereotype == StereoType.Imbword:
        return get_positive_shap_words(train_examples, model, load_from_file, n_gram=n_gram)
    if cf.Stereotype == StereoType.Keyword:
        return get_low_entropy_words(train_examples, load_from_file, n_gram=n_gram)
    if cf.Stereotype == StereoType.Normal:
        # The normal case is just the getting the positive shap words and use them as candidates to find low entropy words among them
        # The assumption is that those words are bias because they have high contribution to the model and having low entropy
        stereotype_words = get_low_entropy_words(train_examples, load_from_file, n_gram=n_gram) & get_positive_shap_words(train_examples, model, load_from_file, n_gram=n_gram)
        np.save(cf.get_file_prefix()+"low_entropy_positive_shap.npy",list(stereotype_words))
        return stereotype_words

def get_low_entropy_words(train_examples,load_from_file=False, n_gram = 1):
    key_word_file_path = cf.get_file_prefix()+"low_entropy.npy"
    if load_from_file:
        return set(np.load(key_word_file_path, allow_pickle=True))
    classes = set(example.label for example in train_examples)
    
    keyword_entropy = {}
    key_words_class_count = defaultdict(lambda: defaultdict(int))
    for example in train_examples:
        for word in split_into_ngrams(example.text, n=n_gram):
            key_words_class_count[word][example.label] += 1

    for word in key_words_class_count.keys():
        entropy = 0
        word_class_count = key_words_class_count[word]
        word_sum = sum(list(word_class_count.values()))
        for _class in classes:
            word_class_percentage = word_class_count.get(_class, 0) / word_sum
            entropy -= word_class_percentage * np.log(word_class_percentage + 1e-8)

        keyword_entropy[word] = entropy, word_sum
    sorted_keyword_entropy = sorted(keyword_entropy.items(), key=lambda x: x[1][0])
    sorted_keyword_entropy = [item for item in sorted_keyword_entropy if item[1][1] >= cf.MIN_FREQUENCY and item[1][0] < cf.ENTROPY_THRESHOLD]

    stereotype_words = [word for word, _ in sorted_keyword_entropy[:int(cf.Alpha*len(sorted_keyword_entropy))+1]]
    print('stereotype_words lengths: ', len(stereotype_words))
    np.save(key_word_file_path, stereotype_words)
    return set(stereotype_words)

def get_positive_shap_words(train_examples = None, model = None,load_from_file=False, n_gram = 1):
    positive_path_file_path = cf.get_file_prefix()+"positive_shap.npy"
    if os.path.exists(positive_path_file_path):
        return set(np.load(positive_path_file_path, allow_pickle=True))
    # if load_from_file:
    #     return np.load(positive_path_file_path)
    model.cuda()
    model.eval()
    train_dataset = TrainDataset(train_examples)
    train_loader = DataLoader(train_dataset, batch_size=cf.Train_Batch_Size, shuffle=False)
    important_words = []

    masker_tokenizers = get_masker_or_tokenizer(n_gram)
    for masker_tokenizer in masker_tokenizers:
        important_words += get_positive_shapwords_with_masker(train_loader, model, masker_tokenizer)
    np.save(positive_path_file_path, important_words)
    return set(important_words)

def get_positive_shapwords_with_masker(train_loader, model, masker_tokenizer):
    explainer = shap.Explainer(
                    model, masker_tokenizer,  seed=1)
    model.eval()
    high_shap_words = []
    for batch_idx, (x, _, _, y, _) in enumerate(train_loader):
        print(batch_idx)
        shap_values = explainer(x)
        for i in range(len(x)):
            shap_value = cf.normalization(
                            shap_values.values[i][:, int(y[i])])
            high_shap_words += list(set([shap_values.data[i][idx].strip() for idx in range(len(shap_value))
                                                    if shap_value[idx] > 0 and len(shap_values.data[i][idx].strip()) > 0]))
    return high_shap_words

def split_into_ngrams(text, n):
    # Split the text into words
    words = text.split()

    # Create a list to store the n-grams
    ngrams = []

    # Loop through the words and create n-grams
    for i in range(len(words) - n + 1):
        ngram = ' '.join(words[i:i+n])
        ngrams.append(ngram)
    return ngrams

def get_masker_or_tokenizer(ngram):
    if ngram == 1:
        if cf.Base_Model == 'RoBERTa' or cf.Base_Model == 'GPT2':
            # roberta_tokenizer = RobertaTokenizerFast.from_pretrained(
            #                 'roberta-base')
            # masker = shap.maskers.Text(roberta_tokenizer)
            masker = shap.maskers.Text(r"\W")
        elif cf.Base_Model == 'TextCNN' or cf.Base_Model == 'TextRCNN':
            masker = shap.maskers.Text(r"\W")
            cf.Explain = True
        else:
            raise NotImplementedError
        return [masker]
    elif ngram == 2:
        return [shap.maskers.Text(lambda s,return_offsets_mapping=True: skip_n_bigram_tokenizer(s,return_offsets_mapping,0), mask_token =cf.Mask_Token),
                shap.maskers.Text(lambda s,return_offsets_mapping=True: skip_n_bigram_tokenizer(s,return_offsets_mapping,1), mask_token =cf.Mask_Token)]
    raise NotImplementedError


def skip_n_bigram_tokenizer(s, return_offsets_mapping=True, n=0):
    """ Custom non-overlapping bigram tokenizers that treat the first n words as a single token, conform to a subset of the transformers API.
    """
    if not s.strip():
        return {"input_ids": [], "offset_mapping": []}
    pos = 0
    offset_ranges = []
    input_ids = []
    words = re.split(r'\s', s)  # extract words

    # if n > 0 and n words exist, group them as a single token
    if n > 0 and len(words) >= n:
        first_n_words = " ".join(words[:n])
        start = pos
        end = pos + len(first_n_words)
        offset_ranges.append((start, end))
        input_ids.append(first_n_words)
        pos = end + 1  # increment the position by the length of the first n words plus the following space

    # continue with bigram tokenization
    for i in range(n, len(words) - 1, 2):  # start from n to skip the first n words
        if i + 1 < len(words):  # make sure i+1 is within index range
            bigram = words[i] + " " + words[i+1]
            start = s.find(bigram, pos)
            if start != -1:  # find returns -1 if not found
                end = start + len(bigram)
                offset_ranges.append((start, end))
                input_ids.append(bigram)
                pos = end + 1  # increment the position by the length of the bigram plus the following space

    # handle the case where there is a single word left
    if n < len(words) and (len(words) - n) % 2 != 0:
        start = s.find(words[-1], pos)
        if start != -1:
            end = start + len(words[-1])
            offset_ranges.append((start, end))
            input_ids.append(words[-1])

    out = {"input_ids": input_ids}
    if return_offsets_mapping:
        out["offset_mapping"] = offset_ranges
    return out


def custom_tokenizer(s, return_offsets_mapping=True):
        """ Custom tokenizers conform to a subset of the transformers API.
        """
        pos = 0
        offset_ranges = []
        input_ids = []
        for m in re.finditer(r"\W", s):
            start, end = m.span(0)
            offset_ranges.append((pos, start))
            input_ids.append(s[pos:start])
            pos = end
        if pos != len(s):
            offset_ranges.append((pos, len(s)))
            input_ids.append(s[pos:])
        out = {}
        out["input_ids"] = input_ids
        if return_offsets_mapping:
            out["offset_mapping"] = offset_ranges
        return out