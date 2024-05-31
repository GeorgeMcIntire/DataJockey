import streamlit as st
st.set_page_config(
	page_title="Jaage Data Jockey",
	page_icon="🎵",
	layout="centered",
)


import sqlite3
import pandas as pd
import swifter
import numpy as np
import json
from glob import glob
from tqdm import tqdm
from io import StringIO
import sys
import plotly_express as px
from datetime import datetime
from st_supabase_connection import SupabaseConnection, execute_query
import shutil
import os
import warnings
from project_tools.utils import json_opener, adapt_array, convert_array, tag_cleaner, digit2letters, fix_gcols, table_loader
from dash_utils import top_value_tables_dict, fetch_mood_scores, fetch_effnet_embeddings_dist_matrices
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity, euclidean_distances
sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("array", convert_array)


st.title("Jaage Data Jockey Dashboard")

#st.sidebar.success("Select a page above.")

conn = st.connection("supabase",type=SupabaseConnection)

st.image('../JAAGE Logo.png', width = 100)
print("=="*50)

if "playlist" not in st.session_state:
	st.session_state["playlist"] = set()
	
if "chosen_top_table" not in st.session_state:
	st.session_state["chosen_top_table"] = ''
	
if "chosen_top_col" not in st.session_state:
	st.session_state["chosen_top_col"] = ''
	


st.session_state["app_ready"] = True

st.subheader("Playlist")
col1, col2 = st.columns([1,1])



if len(st.session_state['playlist']) > 0:
	sids = list(st.session_state["playlist"])
#	output = conn.query("sid, title, artist, bpm, initialkey", table = "tags", ttl = 0).in_('sid', sids).execute().data
	output = execute_query(conn.table("tags").select("sid, title, artist, bpm, initialkey").in_('sid', sids), ttl=0).data
	output = pd.DataFrame(output).set_index('sid')
	st.dataframe(output, hide_index = True)
	
	selected_sid = st.selectbox(
		'Select Song to Play',
		sids, 
		format_func= lambda x: output.loc[x, 'title'] + " -- " + output.loc[x, 'artist'], index = None)
	
	
	if selected_sid:
		
#		song_file = conn.query('file_path', table = 'files', ttl = 0).eq('sid', selected_sid).execute().data
		song_file = execute_query(conn.table("files").select('file_path').eq('sid', selected_sid), ttl = 0).data
#			print(song_file[0]['file_path'])
		st.audio(song_file[0]['file_path'])
		
		
	if st.button("Clear Playlist"):
		st.session_state["playlist"] = set()
		
	
	
	
	