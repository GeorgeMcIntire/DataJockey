import streamlit as st
from st_supabase_connection import SupabaseConnection
import pandas as pd
import swifter
from collections import defaultdict
import numpy as np
from dash_utils import top_value_tables_dict, frame_checkbox, cols_dict, fetch_minmax, fetch_tags, fetch_mood, fetch_genres
from project_tools.utils import convert_array



conn = st.connection("supabase",type=SupabaseConnection)


tags = fetch_tags()
tbl_dict = {"mood":fetch_mood(), "genres" : fetch_genres()}


#mood_minmax = fetch_minmax("../mood_min_max.csv")
#genre_minmax = fetch_minmax("../genre_min_max.csv")
#
#min_max_dict = {"mood":mood_minmax, "genres":genre_minmax}

st.title("Song Filtering")

if "app_ready" not in st.session_state:
	st.write("Database still loading...")
	
else:
#	st.dataframe(tags)
	inner11, inner12 = st.columns([1,1])
	if 'filtered_songs' not in st.session_state:
		st.session_state["filtered_songs"] = {}
		
	if 'output_tbl' not in st.session_state:
		st.session_state["output_tbl"] = None
		
#	
	if st.session_state["app_ready"]:
		
		
		
		
		with inner11:
			low_bpm, high_bpm = st.slider(f'Select BPM range', min_value = 50, max_value = 180, value = (50, 180))
			mood_options = st.multiselect("Select Mood Columns for Querying",options = cols_dict["mood"], default = None)
		with inner12:
			key_options = st.multiselect("Select Keys Columns for Querying",options = sorted(tags.initialkey.unique()))
			st.markdown(" ")
			genre_options = st.multiselect("Select Genre Columns for Querying",options = cols_dict["genres"], default = None)
			
		
		with inner11:
			if mood_options:
				for mo in mood_options:
#					mimx_vals = min_max_dict["mood"]
#					min_val = mimx_vals.loc['min', mo]
#					max_val = mimx_vals.loc['max', mo]
					min_val = 0.
					max_val = 1.
					
					if mo in ["valence", "arousal"]:
						min_val = 4.
						max_val = 8.
					
					
					
					low_val, high_val = st.slider(f'Select {mo} range', min_value = min_val, max_value = max_val, value = (min_val, max_val))
#						st.session_state["value_intervals"]["mood"][mo] = {"lv":low_val, "hv":high_val}
					data = tbl_dict["mood"].copy()
					filtered_frame = data[data[mo].astype(float).between(low_val, high_val)][mo]
					st.session_state["filtered_songs"]["mood" + " | " + mo] =  filtered_frame
					

		with inner12:
			if genre_options:
				for go in genre_options:
#					mimx_vals = min_max_dict["genres"]
#					min_val = mimx_vals.loc['min', go]
#					max_val = mimx_vals.loc['max', go]
					min_val = 0.
					max_val = 1.
					low_val, high_val = st.slider(f'Select {go} range', min_value = min_val, max_value = max_val, value = (min_val, max_val))
#						st.session_state["value_intervals"]["genres"][go] = {"lv":low_val, "hv":high_val}
#						st.session_state["n_queries"] += 1
					
					data = tbl_dict["genres"].copy()
					filtered_frame = data[data[go].between(low_val, high_val)][go]
					st.session_state["filtered_songs"]["genres" + " | " + go] =  filtered_frame
			
		with st.form("song_filter_queries"):
			num_queries = len(st.session_state["filtered_songs"])
			
#			
			if num_queries > 0:
				delete_keys = []
				for key in st.session_state["filtered_songs"].keys():
					if genre_options:
						if key.startswith("genres"):
							if key.split(" | ")[1] not in genre_options:
								delete_keys.append(key)
					if mood_options:
						if key.startswith("mood"):
							if key.split(" | ")[1] not in mood_options:
								delete_keys.append(key)
				for dk in delete_keys:
					del st.session_state['filtered_songs'][dk]
					
				frames = pd.concat(st.session_state["filtered_songs"].values(), axis = 1, join = "inner")
				submit = st.form_submit_button("Submit")
				
				if submit:
					st.session_state["output_tbl"]  = pd.concat([tags, frames], axis = 1, join = "inner")
					st.session_state["output_tbl"] = st.session_state["output_tbl"][st.session_state["output_tbl"].bpm.between(low_bpm, high_bpm)]
					if key_options:
						st.session_state["output_tbl"] = st.session_state["output_tbl"][st.session_state["output_tbl"].initialkey.isin(key_options)]
					
					
					
#				if st.session_state["output_tbl"] is not None:
					rows = st.session_state["output_tbl"].shape[0]
					if rows > 0:
						st.write("Number of rows = {}".format(rows))
	#						st.dataframe(output, hide_index = True)
						frame_checkbox(st.session_state["output_tbl"], key = 'filter_songs')
						print("submit button", submit)
						st.session_state["filtered_songs"] = {}
						
					else:
						st.write("No song match criteria")
							
				
				