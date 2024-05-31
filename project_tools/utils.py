import numpy as np
import os
import io
import json
import inflect
from typing import Union
import sqlite3
from inspect import getsourcefile
import pandas as pd
import boto3
from tqdm import tqdm

p = inflect.engine()


def clear_directory(directory_path):
    try:
        # List all files in the directory
        files = os.listdir(directory_path)

        # Iterate over each file and delete it
        for file_name in files:
            file_path = os.path.join(directory_path, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
#                 print(f"Deleted: {file_path}")

        print(f"All files in {directory_path} have been deleted.")

    except Exception as e:
        print(f"An error occurred: {e}")

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

def fix_gcols(x):
    return x.replace("-", "_").replace(" ", "_").replace("&", "").replace("'", "").replace(",", "").replace("/", "").lower()

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

def song_recommender(ref_id:str, dist_matrix:pd.DataFrame, tags:pd.DataFrame, bpm_window:int= 5, excludes:Union[None, list] = None) -> str:
    
    
    """
    Returns the most similar to a reference song assuming to query stipulations about bpm and key.
    
    Params:
    
    ref_id: Short for reference id, a str representing a song's unique id that's used to query a distance matrix
    
    dist_matrix: Precomputed distance matrix — one of the four embeddings&distance function combinations
    
    bpm_window: An integer used to filtered the pool of possible recommendations so that the function 
    only returns a song of a bpm within n points of the reference track
    
    
    excludes: Acts as a way to manually excludes songs from the recommendation pool. Useful for the setlist_generator
    function when I don't want a song already on the setlist to be re-recommended.
    
    
    Returns:
    
    The id of the most similar song.
    
    """
    
    #Grab bpm and key
    song_bpm = tags.loc[ref_id, "bpm"]
    song_key = tags.loc[ref_id, "tonal_keys"]
    
    #Calculate upper and lower bpm window
    lower_bpm = song_bpm - bpm_window
    upper_bpm = song_bpm + bpm_window
    
    #Ensures that the distance matrix is in line with the metadata
    intersection_ids = dist_matrix.index.intersection(tags.index)
    dist_matrix = dist_matrix.loc[intersection_ids]
    
    #Initialize songs I'm going to drop from the song pool, starting with the reference song
    drops = [ref_id]
    
    #Add excludes song to drops if they are not None
    if excludes is not None:
        drops.extend(excludes)
    
    #Song_pool is the collection of ids of the songs that fall within the bpm and key windows
    song_pool = tags[(tags.tonal_keys.apply(lambda x:key_matcher(song_key, x))) & 
                     (tags.bpm.between(lower_bpm, upper_bpm))].drop(drops, errors = "ignore").index.tolist()
    
    #Query the distance matrix using the reference song and song pool and grab the arg min id
    most_similar_song = dist_matrix.loc[ref_id, song_pool].idxmin()
    
    #Return most similar id
    return most_similar_song


def setlist_generator(ref_id:str, dist_matrix:pd.DataFrame,tags:pd.DataFrame, bpm_window:int= 5, n_songs = 10) -> pd.DataFrame:
    
    
    """
    Generates a setlist of songs based on a starting point song.
    
    Params:
    
    ref_id: Short for reference id, a str representing a song's unique id that's used to query a distance matrix
    
    dist_matrix: Precomputed distance matrix — one of the four embeddings&distance function combinations
    
    bpm_window: An integer used to filtered the pool of possible recommendations so that the function 
    only returns a song of a bpm within n points of the reference track
    
    n_songs: The number of songs for the setlist
    
    Returns
    
    A Dataframe of n_songs in the playlist.
    
    """
    
    #Initializ setlist with the id of the reference track
    setlist = [ref_id]
    
    #excludes is initialized as None but will change after the first iteration
    excludes = None
    
    #Conduct n_songs iterations
    for i in range(n_songs): 
        #For each iteration find the most similar song to the reference song
        
        most_similar_song = song_recommender(ref_id, dist_matrix, tags,
                                             bpm_window=bpm_window, excludes = excludes)
            
        #add most similar song to the setlist
        setlist.append(most_similar_song)
        
        #Overwrites excludes to be a copy of setlist so that in the next iteration 
        #the song_recommender won't rerecommend the same tracks
        excludes = setlist[:]
        
        #The upcoming song becomes the now playing song.
        ref_id = most_similar_song
        
    #Query tags dataframe using the setlist (list of ids) and the title and artist columns
    return tags.loc[setlist, ["title", "artist"]].assign(order = range(1, len(setlist) + 1))


def s3_uploader(files_dict):

    aws_access_key_id = ''
    aws_secret_access_key = ''
    aws_region = ''
    bucket_name = ''
    s3_key = 'song_files/{}{}'

# Create a Lambda client
    client = boto3.client('s3', region_name=aws_region, 
                          aws_access_key_id=aws_access_key_id, 
                          aws_secret_access_key=aws_secret_access_key)
    
    for idd, fil in tqdm(files_dict.items()):
        ext = os.path.splitext(fil)[1]
        key = s3_key.format(idd, ext)
        client.upload_file(fil, bucket_name, key)
        
    print(f"All {len(files_dict)} songs uploaded to S3 bucket")
    