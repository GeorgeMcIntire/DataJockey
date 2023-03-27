import numpy as np
import os
import io
import json
import inflect
from typing import Union
import sqlite3


p = inflect.engine()

def adapt_array(arr:np.ndarray) -> memoryview:
	out = io.BytesIO()
	np.save(out, arr)
	out.seek(0)
	return sqlite3.Binary(out.read())

def convert_array(text:memoryview) -> np.ndarray:
	out = io.BytesIO(text)
	out.seek(0)
	return np.load(out, allow_pickle=True)

def json_opener(jay:str) -> str:
	with open(jay) as f:
		output = json.load(f)
	return output


def tag_cleaner(x) -> Union[str, int, float]:
	if type(x) != list:
		return x
	l = list(set(x))
	if len(l) == 1:
		return l[0]
	elif len(l) ==2:
		return l[1]
	else:
		return np.nan
	
def digit2letters(digits:str) -> str:
	if not digits[0].isdigit():
		return digits
	num = p.number_to_words(digits)
	if digits.endswith("s"):
		num = p.plural(num)
	return num
	
	
	