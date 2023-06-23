from my_main import main
from unittest.mock import patch
import os
import sys
@patch.object(sys, 'argv', ['my_main.py', '--Dataset_Name', 'Amazon', '--Base_Model', 'TextCNN'])
def test_main():
    main()
    assert True