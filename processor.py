
import sqlite3
import pandas as pd
import swifter
import numpy as np
import matchering as mg
import json
from glob import glob
from tqdm import tqdm
import pathlib
import taglib
import shutil
import os
from essentia.standard import MusicExtractor, YamlOutput
import warnings
from zipfile import ZipFile
from project_tools.utils import json_opener, adapt_array, convert_array, tag_cleaner, fix_gcols, clear_directory, s3_uploader
from project_tools.llm import metadata_extract
#from project_tools.db_utils import create_record, Tag, TagsDB
from project_tools.models import Activator, Classifier

warnings.filterwarnings('ignore')
sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("array", convert_array)

conn = sqlite3.connect("jaage.db", detect_types= sqlite3.PARSE_DECLTYPES)
cur = conn.cursor()

load_path = "/Volumes/Storage/Loading Dock/"
dj_hub  = "/Volumes/Storage/DJ Hub/"
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


ref_file = '/Volumes/Storage/DJ Hub/Rayko - Magnetized (Rayko rework).wav'
#The directory where all my music is stored.
collection = "Collection"

new_file_paths = []
for f in tqdm(loading_files, desc = "Processing and Extracting Song Data"):
    out_stem = f.stem
    out_path = f.parent.parent/collection/out_stem
    out_path = out_path.as_posix() +".wav"
    
    mg.process(target= f.as_posix(),
              reference=ref_file, 
              results = [mg.pcm24(out_path)])
    
    load_tags = taglib.File(f.as_posix())
    mastered_tags = taglib.File(out_path)
    mastered_tags.tags = load_tags.tags
    extracted_metadata = metadata_extract(out_stem)
    for tag_name, tag_val in extracted_metadata.items():
      mastered_tags.tags[tag_name] = tag_val
    
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
        




tag_check = input("Dump new tracks in RekordBox/Serato and tag them, press enter when you're done")

music_ext = MusicExtractor(lowlevelStats=['mean', 'stdev'],
                                    rhythmStats=['mean', 'stdev', "max", "min", "median"],
                                    tonalStats=['mean', 'stdev'],
                           mfccStats = ["mean", "cov"],
                           gfccStats = ["mean", "cov"])


ids = pd.read_sql_query("SELECT sid FROM tags", con = conn).sid.tolist()

out_dir = 'temp_features/'
extracted_files = []
id_2_paths = {}

for fil in tqdm(new_file_paths, total = len(new_file_paths)):
    try:
        features, _ = music_ext(fil)
        idd = features['metadata.audio_properties.md5_encoded']
        if idd in ids:
          print(f"{fil} already in db")
          continue
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

meta_df = meta_df.swifter.applymap(tag_cleaner)
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
    ser = list_data[col].swifter.apply(pd.Series)
    ser.columns = col + "_"+ ser.columns.astype(str)
    ser.to_sql(col+"_tbl", con = conn,if_exists="append")
    
    
#cfg_avg = Config("/Users/georgemcintire/projects/djing/music-audio-representations/supporting_data/configs/mule_embedding_average.yml")
#cfg_avg["Analysis"]["feature_transforms"][1]['EmbeddingFeature']['model_location'] = "/Users/georgemcintire/projects/djing/music-audio-representations/supporting_data/model"
#cfg_avg["Analysis"]["source_feature"]["AudioWaveform"]["input_file"]["AudioFile"]['sample_rate'] = 16000
#
#analysis = Analysis(cfg_avg)  
#   
#   
#for sid, song_file in tqdm(id_2_paths.items(), desc = 'Mule Embeddings Extraction'):
#   data = analysis.analyze(song_file).data.T
##     data = file2embed(song_file).data.T
#   out = pd.DataFrame(index=[sid], data = data)
#   out.index.rename("sid",inplace=True)
#   out.to_sql("mule_embeddings", con = conn, if_exists="append")
  
  

path2id = {v:k for k, v in id_2_paths.items()}
act = Activator(input_length=2.05, 
                model_path="onnx_models/discogs-effnet-bsdynamic-1.onnx",
                   pathid_dict=path2id)



with open("onnx_models/json_info/discogs-effnet-bsdynamic-1.json") as f:
    gcols = json.load(f)["classes"]
    gcols = [fix_gcols(i) for i in gcols]
  
  
keep_gcols = np.load("keep_genre_cols.pkl", allow_pickle = True)

for song in act.batch_inference():
    with conn:
        sid, sf, output = song
        genre_acts = output["activations"]
        embeds = output["embeddings"]
        genre_acts = [np.expand_dims(genre_acts[:, i], 0) for i in range(400)]
        genre_acts = pd.DataFrame(index = [sid], data = [genre_acts], columns=gcols)
        genre_acts.index.rename("sid",inplace=True)
        genre_acts = genre_acts[keep_gcols]
        genre_acts_means = genre_acts.swifter.applymap(lambda x:x[0].mean())
        cur.execute("INSERT INTO effnet_embeddings (sid, effnet_embedding) values (?,?)", 
              (sid, np.expand_dims(embeds,0)))
        genre_acts.to_sql("effnet_genres", con=conn, if_exists="append")
        genre_acts_means.to_sql("effnet_genres_mean", con = conn, if_exists = "append")
    conn.commit()

model_paths = sorted(glob("onnx_models/*.onnx"))
model_infos = sorted(glob("onnx_models/json_info/*.json"))
effnet_models = [{"model": model_paths[i], 
                  "json":model_infos[i]} for i in range(len(model_paths)) if "effnet" in model_paths[i]]

effnet_models = effnet_models[:2] + effnet_models[4:]

new_ids = list(path2id.values())

for em in effnet_models:
    cls = Classifier(em, new_ids=new_ids)
    cls.batch_inference(tbl = "effnet_embeddings")
    cls.conn.commit()
    print("Completed => ", cls.table_name, "\n\n")
    
    
act = Activator(input_length=3, model_path="msd_models/msd-musicnn-1.onnx", pathid_dict=path2id)
ins_query = "INSERT INTO msd_musicnn_1_embeddings (sid, msd_embeddings) values (?,?)"
for song in act.batch_inference():
  with conn:
    sid, sf, output = song
    embeds = output["embeddings"]
    embeds = np.expand_dims(embeds,0)
    output = (sid, embeds)
    cur.execute(ins_query, output)
  conn.commit()
    

em = {"model":"msd_models/deam-msd-musicnn-2.onnx",
  'json':'msd_models/deam-msd-musicnn-2.json'}

tbl = "msd_musicnn_1_embeddings"
cls = Classifier(em, new_ids=new_ids)
ins_query = cls._create_ins_query()


with conn:
  results = cur.execute(f"SELECT * FROM {tbl}").fetchall()
  results = [i for i in results if i[0] in new_ids]
  for sid, embed in tqdm(results):
    embed = convert_array(embed)[0]
    preds = cls._inference(embed)
    output = [sid]
    for i in range(len(cls.classes)):
      array = preds[:, i]
      array = np.expand_dims(array,0)
      output.append(array)
    output = tuple(output)
    cur.execute(ins_query, output)
conn.commit()

print("Completed => ", cls.table_name, "\n\n")

s3_uploader(id_2_paths)

clear_directory(load_path)