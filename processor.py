#!/usr/bin/env python
# coding: utf-8

# # Music ETL Process Notebook
# 
# This code is the notebook version of the `processor.py` file which is the ETL script I use whenever I have a new batch of songs that need to processed, analyzed, and uploaded to my database.
# 
# Here I annotate each step of the process with explanations of what this code does.

# In[1]:


#Imports
import sqlite3
import pandas as pd
import numpy as np
import matchering as mg
import json
from glob import glob
from tqdm import tqdm
from io import StringIO
import sys
import pathlib

import taglib
from datetime import datetime
import shutil
import os
from essentia.standard import MusicExtractor, YamlOutput,MetadataReader, PCA, YamlInput
import warnings
from zipfile import ZipFile
warnings.filterwarnings('ignore')
pd.set_option('max_colwidth', 100)


from project_tools.utils import json_opener, adapt_array, convert_array, tag_cleaner, digit2letters
from project_tools.models import Activator, Classifier

sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("array", convert_array)

conn = sqlite3.connect("jaage.db", detect_types= sqlite3.PARSE_DECLTYPES)
cur = conn.cursor()

load_path = "/Volumes/LaCie/Loading Dock/"
dj_hub  = "/Volumes/LaCie/DJ Hub/"
zip_files = glob(load_path+"*.zip")

if len(zip_files) > 0:
    for z in zip_files:
        zf = ZipFile(z)
        zf.extractall(path=load_path)
        shutil.move(z,dj_hub)

loading_files = pathlib.Path(load_path).glob("*[.wav, .mp3, .aiff]")

len_loading_files = len(list(loading_files))
print("There are {} files for the ETL pipeline".format(len_loading_files))

loading_files = pathlib.Path(load_path).glob("*[.wav, .mp3, .aiff]")


ref_file = '/Volumes/LaCie/DJ Hub/Rayko - Magnetized (Rayko rework).wav'
#The directory where all my music is stored.
collection = "Collection"

new_file_paths = []
for f in tqdm(loading_files):
    out_stem = f.stem
    out_path = f.parent.parent/collection/f.stem
    out_path = out_path.as_posix() +".wav"
    
    mg.process(target= f.as_posix(),
              reference=ref_file, 
              results = [mg.pcm24(out_path)])
    
    load_tags = taglib.File(f.as_posix())
    mastered_tags = taglib.File(out_path)
    mastered_tags.tags = load_tags.tags
    mastered_tags.save()
    
    new_file_paths.append(out_path)
    
    try:
        shutil.move(f.as_posix(), dj_hub)
    except:
        print(f, "already exists")
        os.remove(f.as_posix())
    

copied_paths = new_file_paths[:]
new_file_paths = []
for i in copied_paths:
    if os.path.exists(i):
        new_file_paths.append(i)
        
len(new_file_paths)


tag_check = input("Dump new tracks in RekordBox/Serato and tag them, press enter when you're done")

music_ext = MusicExtractor(lowlevelStats=['mean', 'stdev'],
                                    rhythmStats=['mean', 'stdev', "max", "min", "median"],
                                    tonalStats=['mean', 'stdev'],
                           mfccStats = ["mean", "cov"],
                           gfccStats = ["mean", "cov"])



out_dir = 'temp_features/'
extracted_files = []
id_2_paths = {}

for fil in tqdm(new_file_paths, total = len(new_file_paths)):
    try:
        features, _ = music_ext(fil)
        idd = features['metadata.audio_properties.md5_encoded']
        YamlOutput(filename= out_dir+"features.json", format="json")(features)
        json_data = json_opener(out_dir+"features.json")
        id_2_paths[idd] = fil
        extracted_files.append(json_data)
    except Exception as e:
        print(e)


extracted = pd.json_normalize(extracted_files)
extracted.columns = extracted.columns.str.replace(".", "_")
extracted.rename(columns={"metadata_audio_properties_md5_encoded":"sid"}, inplace=True)

drop_cols = np.load("drop_cols.pkl", allow_pickle=True).tolist()
extracted.drop(drop_cols, axis = 1, inplace=True,errors="ignore")
extracted.set_index("sid", inplace=True)
extracted.shape


cols = extracted.columns
meta_cols = cols[cols.str.startswith("meta")]
non_meta_cols = cols[~cols.str.startswith("meta")]
meta_df = extracted[meta_cols].copy()
extracted.drop(meta_cols, axis = 1, inplace=True)


list_cols = extracted.columns[extracted.iloc[0].apply(lambda x:type(x)) == list]
no_list_cols = extracted.columns[extracted.iloc[0].apply(lambda x:type(x)) != list]
list_data = extracted[list_cols]
no_list_data = extracted[no_list_cols]

meta_df = meta_df.applymap(tag_cleaner)
meta_df.columns = meta_df.columns.str.split("_").map(lambda x:x[-1])
meta_df.rename(columns={"name":"file_name"}, inplace=True)

tags_cols = pd.read_sql("SELECT * FROM tags LIMIT 1", con = conn).set_index('sid').columns.tolist()

meta_cols = [i for i in meta_df.columns if i in tags_cols]
meta_df[meta_cols].to_sql("tags", con=conn, if_exists = "append")

files = pd.DataFrame(id_2_paths.items(), columns=["sid", "file_path"])
files.to_sql("files", con = conn, if_exists="append", index = False)

cols = no_list_data.columns
tonal_cols = cols[cols.str.startswith("tonal")]
lowlevel_cols = cols[cols.str.startswith("lowlevel")]
rhythm_cols = cols[cols.str.startswith("rhyt")]

tonal_df = no_list_data[tonal_cols]
lowlevel_df = no_list_data[lowlevel_cols]
rhythm_df = no_list_data[rhythm_cols]
tonal_df.to_sql("tonal_features", con=conn, if_exists="append")
lowlevel_df.to_sql("lowlevel_features", con=conn, if_exists="append")
rhythm_df.to_sql("rhythm_features", con=conn, if_exists="append")
for col in tqdm(list_cols):
    ser = list_data[col].apply(pd.Series)
    ser.columns = col + "_"+ ser.columns.astype(str)
    ser.to_sql(col+"_tbl", con = conn,if_exists="append")

path2id = {v:k for k, v in id_2_paths.items()}
act = Activator(input_length=2.05, 
                model_path="onnx_models/discogs-effnet-bsdynamic-1.onnx",
                   pathid_dict=path2id)

gcols = pd.read_sql_query("SELECT * FROM effnet_genres LIMIT 1 ", con = conn).columns[1:].tolist()

for song in act.batch_inference():
    with conn:
        sid, sf, output = song
        genre_acts = output["activations"]
        embeds = output["embeddings"]
        genre_acts = [np.expand_dims(genre_acts[:, i], 0) for i in range(400)]
        genre_acts = pd.DataFrame(index = [sid], data = [genre_acts], columns=gcols)
        genre_acts.index.rename("sid",inplace=True)
        cur.execute("INSERT INTO effnet_embeddings (sid, effnet_embedding) values (?,?)", 
                    (sid, np.expand_dims(embeds,0)))
        genre_acts.to_sql("effnet_genres", con=conn, if_exists="append")
    conn.commit()

model_paths = sorted(glob("onnx_models/*.onnx"))
model_infos = sorted(glob("onnx_models/json_info/*.json"))
effnet_models = [{"model": model_paths[i], 
                  "json":model_infos[i]} for i in range(len(model_paths)) if "effnet" in model_paths[i]]

effnet_models = effnet_models[:2] + effnet_models[4:]

new_ids = list(path2id.values())

for em in effnet_models:
    cls = Classifier(em, new_ids=new_ids)
    cls.batch_inference()
    cls.conn.commit()
    print("Completed => ", cls.table_name, "\n\n")