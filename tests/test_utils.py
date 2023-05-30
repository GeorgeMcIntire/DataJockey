import numpy as np
import pytest
import os
from project_tools import utils

print("Current dir>>>>", os.getcwd())

@pytest.mark.parametrize("number_form, letter_form", [
	("60", "sixty"),
	("73", "seventy-three"),
	("80s", "eighties"),
	("90s", "nineties"),
])
def test_digit2letters(number_form, letter_form):
	assert utils.digit2letters(number_form) == letter_form
	

def test_jsonopener():
#	fname = os.path.join(os.path.dirname(__file__), "test_example.json")
	fname = os.path.join(os.getcwd(), "tests/test_example.json")
	fname = "tests/test_example.json"
	assert type(utils.json_opener(fname)) == dict
	
@pytest.mark.parametrize("in_data, out_data", [
	(120, 120),
	("disco", "disco"),
	(["Kiki Gyan"], "Kiki Gyan"),
	([110, 110], 110)
])
def test_tagcleaner(in_data, out_data):
	assert utils.tag_cleaner(in_data) == out_data
	
	
# create test data
test_array = np.arange(5)

def test_adapt_array():
	# test if the function returns a memoryview object
	assert isinstance(utils.adapt_array(test_array), memoryview)
	
def test_convert_array():
	# test if the function returns a numpy array
	assert isinstance(utils.convert_array(memoryview(utils.adapt_array(test_array))), np.ndarray)
	
	# test if the function returns the original array after conversion
	assert np.array_equal(utils.convert_array(memoryview(utils.adapt_array(test_array))), test_array)

@pytest.mark.parametrize("key1, key2, thresh, result", [
	("6A", "7A", 2, True),
	("5A", "5B", 2, True),
	("12A", "1A", 2, True),
	("4A","6A", 2, False),
	("2A","8B", 2, False),
	("9A","4A", 2, False),
])
def test_key_matcher(key1, key2,thresh, result):
    assert utils.key_matcher(key1, key2, thresh) == result
    
@pytest.mark.parametrize("key, camelot_key", [
	("Gminor", "6A"),
	("Aminor", "8A"),
	("Amajor", "11B"),
])  
def test_camelot_converter(key, camelot_key):
    assert utils.camelot_convert(key) == camelot_key