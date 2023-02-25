import numpy as np
from tqdm import tqdm
import onnx
import onnxruntime as ort
import librosa
import os
from time import time
import matplotlib.pyplot as plt
import io
import pandas as pd
import json
import inflect

from glob import glob
import sqlite3

effnet_config = dict(FFT_HOP = 256, FFT_SIZE = 512, N_MELS = 96)

p = inflect.engine()

def adapt_array(arr):
	out = io.BytesIO()
	np.save(out, arr)
	out.seek(0)
	return sqlite3.Binary(out.read())

def convert_array(text):
	out = io.BytesIO(text)
	out.seek(0)
	return np.load(out)

def json_opener(jay):
	with open(jay) as f:
		output = json.load(f)
	return output


def tag_cleaner(x):
	if type(x) != list:
		return x
	l = list(set(x))
	if len(l) == 1:
		return l[0]
	elif len(l) ==2:
		return l[1]
	else:
		return np.nan
	
def digit2letters(digits):
	if not digits[0].isdigit():
		return digits
	num = p.number_to_words(digits)
	if digits.endswith("s"):
		num = p.plural(num)
	return num
	
	
	