import numpy as np
import pytest
from project_tools import utils

@pytest.mark.parametrize("number_form, letter_form", [
	("60", "sixty"),
	("73", "seventy-three"),
	("80s", "eighties"),
	("90s", "nineties"),
])
def test_digit2letters(number_form, letter_form):
	assert utils.digit2letters(number_form) == letter_form
	

def test_jsonopener():
	assert type(utils.json_opener("/Users/georgemcintire/projects/djing/project_tools/tests/test_example.json")) == dict
	
@pytest.mark.parametrize("in_data, out_data", [
	(120, 120),
	("disco", "disco"),
	(["Kiki Gyan"], "Kiki Gyan"),
	([110, 110], 110),
	(["Folamour", "folamour"], "Folamour"),
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
	