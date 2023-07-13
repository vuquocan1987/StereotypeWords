import pytest
from collections import Counter, defaultdict
from src import data_process
from src import model
import config as cf
import torch
from collections import namedtuple

@pytest.fixture
def amazon_dataset():
    return data_process.TextDataset('Amazon')
@pytest.fixture
def train_examples():
    # Replace with your own train examples
    TextDataset = data_process.TextDataset("Amazon")
    return TextDataset.train_examples

@pytest.fixture
def model_fixture():
    # Replace with your own model
    m = model.TextCNN()
    m = m.cuda()
    cf.Base_Model = 'TextCNN'
    cf.Stereotype = cf.StereoType.Normal
    cf.Dataset_Name = 'Amazon'
    cf.embedding, cf.word2id = cf.Pickle_Read('./w2v/glove.300d.en.txt.pickle')
    cf.embedding.weight.requires_grad = False
    m.load_state_dict(torch.load(cf.get_file_prefix() + 'xinit.pt'))
    return m

@pytest.fixture
def sample_amazon_dataset():
    return data_process.TextDataset('AmazonTest')


def test_conform(amazon_dataset):
    test_count, dev_count = defaultdict(int), defaultdict(int)
    for example in amazon_dataset.dev_examples:
        dev_count[example.label] += 1
    for example in amazon_dataset.test_examples:
        test_count[example.label] += 1
    for key in dev_count.keys():
        assert dev_count[key] in range(test_count[key]-1,test_count[key]+2)
    assert True
def test_generate_stereotype_words(train_examples, model_fixture):
    stereotype_words = data_process.generate_stereotype_words(train_examples, model_fixture)
    assert isinstance(stereotype_words, list)
    assert all(isinstance(word, str) for word in stereotype_words)

def test_get_important_words(train_examples, model_fixture):
    important_words = data_process.get_positive_shap_words(train_examples, model_fixture)
    assert isinstance(important_words, list)
    assert all(isinstance(word, str) for word in important_words)

def test_get_low_entropy_words():
    Example = namedtuple('Example', ['text', 'label'])
    train_examples = [
        Example('this is a test', 'class1'),
        Example('this is another test', 'class1'),
        Example('this is a third test', 'class2'),
        Example('this is a third fourth test', 'class2'),
        Example('this is a fifth test', 'class2')
    ]
    result = data_process.get_low_entropy_words(train_examples)
    assert result == {"another"}

data_test = [
"getting [MASK] curtain extender compatible with [MASK] kirsch rod was [MASK] [MASK] [MASK] [MASK] extender [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] with [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK]",
"[MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK]",
"[MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK]",
"[MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] with [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK]"
]

def test_dataset_Public(sample_amazon_dataset):
    cf.Stereotype = cf.StereoType.Keyword
    sample_amazon_dataset.Public()
    print('hello word')
    for i, example_train in enumerate(sample_amazon_dataset.train_examples):
        print(example_train.partial_counterfactual_text)
        assert data_test[i] == example_train.partial_counterfactual_text
        assert len(example_train.text.split(' ')) == len(example_train.fully_counterfactual_text.split(' '))
        assert len(example_train.text.split(' ')) == len(example_train.partial_counterfactual_text.split(' '))


def test_dataset_Public_BiGram(sample_amazon_dataset):
    cf.Stereotype = cf.StereoType.Keyword
    sample_amazon_dataset.Public(n_gram=2)
    print('hello word')
    

def test_amazon_dataset_Public_BiGram(amazon_dataset):
    cf.Alpha = 1
    cf.Stereotype = cf.StereoType.Keyword
    cf.MIN_FREQUENCY = 2
    cf.ENTROPY_THRESHOLD = .5
    amazon_dataset.Public(n_gram=2)
    print('hello word')

def test_split_into_ngrams():
    text = "the quick brown fox jumps over the lazy dog"
    trigrams = data_process.split_into_ngrams(text, 3)
    expected_trigrams = [('the', 'quick', 'brown'), ('quick', 'brown', 'fox'), ('brown', 'fox', 'jumps'), ('fox', 'jumps', 'over'), ('jumps', 'over', 'the'), ('over', 'the', 'lazy'), ('the', 'lazy', 'dog')]
    assert trigrams == expected_trigrams