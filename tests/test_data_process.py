import pytest
from collections import Counter, defaultdict
from src import data_process

@pytest.fixture
def amazon_dataset():
    return data_process.TextDataset('Amazon')
def test_conform(amazon_dataset):
    test_count, dev_count = defaultdict(int), defaultdict(int)
    for example in amazon_dataset.dev_examples:
        dev_count[example.label] += 1
    for example in amazon_dataset.test_examples:
        test_count[example.label] += 1
    for key in dev_count.keys():
        assert dev_count[key] in range(test_count[key]-1,test_count[key]+2)

    assert True