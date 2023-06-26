from src import my_main
from unittest.mock import patch
import os
import sys
@patch.object(sys, 'argv', ['my_main.py', '--Dataset_Name', 'Amazon', '--Base_Model', 'TextCNN'])
def test_main():
    my_main.main()
    assert True