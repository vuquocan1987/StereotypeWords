from src import my_main,data_process,model
from unittest.mock import patch
import os
import sys
import pytest
from collections import Counter, defaultdict
import config as cf
@patch.object(sys, 'argv', ['my_main.py', '--Dataset_Name', 'Amazon', '--Base_Model', 'TextCNN'])
def test_amazon():
    my_main.main()
    assert True

def test_arc():
    sys.argv = ['my_main.py', '--Dataset_Name', 'ARC', '--Base_Model', 'TextCNN']
    my_main.main()
    assert True

def test_RoBERTa():
    sys.argv = ['my_main.py', '--Dataset_Name', 'Amazon', '--Base_Model', 'RoBERTa']
    my_main.main()
    assert True

@patch.object(sys, 'argv', ['my_main.py', '--Dataset_Name', 'Amazon', '--Base_Model', 'TextCNN'])
def test_amazon_low_init():
    cf.Init_epoch = 2
    my_main.main()
    assert True