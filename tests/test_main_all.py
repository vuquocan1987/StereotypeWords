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
def test_amazon_normal():
    cf.Pretrained = True
    my_main.main()
    assert True

@patch.object(sys, 'argv', ['my_main.py', '--Dataset_Name', 'Amazon', '--Base_Model', 'TextCNN'])
def test_amazon_high_shap():
    cf.Pretrained = True
    cf.Stereotype = "Imbword"
    my_main.main()
    assert True

@patch.object(sys, 'argv', ['my_main.py', '--Dataset_Name', 'Amazon', '--Base_Model', 'TextCNN'])
def test_amazon_low_entropy():
    cf.Pretrained = True
    cf.Stereotype = "Keyword"
    my_main.main()
    assert True

@patch.object(sys, 'argv', ['my_main.py', '--Dataset_Name', 'Amazon', '--Base_Model', 'TextCNN'])
def test_amazon_nouns():
    cf.Pretrained = True
    cf.Stereotype = "Noun"
    my_main.main()
    assert True

@patch.object(sys, 'argv', ['my_main.py', '--Dataset_Name', 'Amazon', '--Base_Model', 'TextCNN'])
def test_random_mask_amazon():
    cf.Pretrained = True
    cf.Stereotype = "RandomMask"
    cf.RANDOM_MASK_RATE = .2
    my_main.main()
    assert True


@patch.object(sys, 'argv', ['my_main.py', '--Dataset_Name', 'Amazon', '--Base_Model', 'TextCNN'])
def test_amazon_bigram():
    cf.N_GRAM = 2
    cf.Stereotype = 'Keyword'
    cf.Alpha = 1
    cf.Pretrained = True
    my_main.main()
    assert True



'Keyword' 'Imbword' 'Normal' 'RandomMask' 'Noun'