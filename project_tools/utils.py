import numpy as np
import os
import io
import json
import inflect
from typing import Union
import sqlite3
from inspect import getsourcefile
import pandas as pd


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
	l = sorted(set(x))
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

def camelot_convert(key:str) -> str:
    module_file_path = os.path.abspath(getsourcefile(lambda: 0))
    tonal_keys_full_filepath = os.path.join(os.path.dirname(module_file_path), "tonal_keys_dict.json")
    key_dict = json_opener(tonal_keys_full_filepath)
    return key_dict[key]


def load_genre_cols():
    module_file_path = os.path.abspath(getsourcefile(lambda: 0))
    genre_cols_full_filepath = os.path.join(os.path.dirname(module_file_path), "../keep_genre_cols.pkl")
    return np.load(genre_cols_full_filepath, allow_pickle=True)

    
def key_matcher(key1:str, key2:str, thresh:int = 2):
    let_diff = key1[-1] != key2[-1]

    num_diff = abs(int(key1[:-1]) - int(key2[:-1]))
    if num_diff == 11:
        num_diff = 1
    output = let_diff + num_diff
    return output < thresh


def table_loader(conn, *queries, apply_function = None):
    
    if len(queries) == 1:
        query = queries[0]
        if apply_function is not None:
            output_table = pd.read_sql_query(query, con = conn).set_index("sid").applymap(apply_function)
        else:
            output_table = pd.read_sql_query(query, con = conn).set_index("sid")
            
    else:
        output_table = []
        
        for query in queries:
            if apply_function is not None:
                output = pd.read_sql_query(query, con = conn).set_index("sid").applymap(apply_function)
            else:
                output = pd.read_sql_query(query, con = conn).set_index("sid")
            output_table.append(output)
            
        output_table = pd.concat(output_table, axis = 1)
        
    return output_table