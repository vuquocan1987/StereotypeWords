from src import my_main,data_process,model
from unittest.mock import patch
import os
import sys
import pytest
from collections import Counter, defaultdict
import config as cf
# @patch.object(sys, 'argv', ['my_main.py', '--Dataset_Name', 'Amazon', '--Base_Model', 'TextCNN'])
# def test_amazon():
#     my_main.main()
#     assert True


@patch.object(sys, 'argv', ['my_main.py', '--Dataset_Name', 'Amazon', '--Base_Model', 'TextCNN'])
def test_amazon():
    cf.Pretrained = True
    my_main.main()
    assert True



@patch.object(sys, 'argv', ['my_main.py', '--Dataset_Name', 'Amazon', '--Base_Model', 'TextCNN'])
def test_random_mask_amazon():
    cf.Pretrained = True
    cf.Stereotype = cf.StereoType.RandomMask
    cf.RANDOM_MASK_RATE = .2
    my_main.main()
    assert True


@patch.object(sys, 'argv', ['my_main.py', '--Dataset_Name', 'Amazon', '--Base_Model', 'TextCNN'])
def test_amazon_bigram_normal():
    cf.Pretrained = True
    cf.N_GRAM = 2
    my_main.main()
    assert True

@patch.object(sys, 'argv', ['my_main.py', '--Dataset_Name', 'Amazon', '--Base_Model', 'TextCNN'])
def test_amazon_bigram():
    cf.N_GRAM = 2
    cf.Stereotype = cf.StereoType.Keyword
    cf.Alpha = 1
    cf.MIN_FREQUENCY = 2
    cf.ENTROPY_THRESHOLD = .5
    cf.Pretrained = True
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


@patch.object(sys, 'argv', ['my_main.py', '--Dataset_Name', 'Amazon', '--Base_Model', 'TextCNN'])
def test_amazon_pretrained():
    cf.Pretrained = True
    cf.LOAD_STEREO_TYPE_WORDS_FROM_FILE = False
    cf.Round = 1
    my_main.main()
    assert True


