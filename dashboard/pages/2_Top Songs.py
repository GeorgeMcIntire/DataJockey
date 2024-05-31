import streamlit as st
from st_supabase_connection import SupabaseConnection, execute_query
import pandas as pd
import swifter
import numpy as np
from dash_utils import top_value_tables_dict, frame_checkbox, cols_dict
from project_tools.utils import convert_array


conn = st.connection("supabase",type=SupabaseConnection)
#tags = conn.query("*", table = "tags").execute()
#mood_scores = conn.query("*", table = "mood").execute()
#genre_scores = conn.query("*", table = "genres").execute()

st.title("Chosen Top N Songs from Table")
#print(dir(conn.))

if "app_ready" not in st.session_state:
	st.write("Database still loading...")

else:
	
	inner11, inner12 = st.columns([1,1])
	with inner11:
		table_option_name = st.selectbox("Select table", ["mood", "genres"], index = None)
	
	if table_option_name:
		column_names = cols_dict[table_option_name]
		column_names = [''] + column_names
		with inner12:
			column_option = st.selectbox("Select column", column_names, format_func=lambda x:x.title().replace("___", "_"))
		
		if column_option:			
			sort_option = inner11.radio("Sorting", options = [False, True], format_func=lambda x:"Ascending" if x else "Descending", horizontal=True)
			top_n = inner11.radio("N songs", options = [5, 10, 20], horizontal=True)
			top_col_songs = execute_query(conn.table(table_option_name).select(f"sid, {column_option}").order(column_option, desc = not sort_option).limit(top_n), ttl=0).data
		
			sids = [i['sid'] for i in top_col_songs]
			top_col_songs = {i['sid']:i[column_option] for i in top_col_songs}

			metadata = execute_query(conn.table("tags").select("sid, title, artist, bpm, initialkey").in_('sid', sids), ttl = 0 ).data

			top_songs = pd.json_normalize(metadata).rename(columns = lambda x:x.replace("tags.", "")).set_index('sid')
			top_songs[column_option] = top_songs.index.map(top_col_songs)

			frame_checkbox(top_songs, key = 'topsongs')
	
			