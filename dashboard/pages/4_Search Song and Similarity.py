import streamlit as st
from st_supabase_connection import SupabaseConnection, execute_query
import pandas as pd
import joblib
import vecs
from supabase import create_client, Client
import numpy as np
from dash_utils import top_value_tables_dict, frame_checkbox, cols_dict, fetch_tags
import plotly_express as px
from project_tools.utils import convert_array


conn = st.connection("supabase",type=SupabaseConnection, ttl = None)

DB_CONNECTION = st.secrets['vecs']['DB_CONNECTION']


if "vecs" not in st.session_state:
	st.session_state['vecs'] = vecs.create_client(DB_CONNECTION)
	
if 'vdb' not in st.session_state:
	st.session_state["vdb"] = {}
	
	with st.spinner(text = "Connecting to Vector Databases and Establishing Indices"):
		st.session_state["vdb"]['effnet'] = st.session_state['vecs'].get_or_create_collection(name="effnet", dimension=1280)
		st.session_state["vdb"]['msd'] = st.session_state['vecs'].get_or_create_collection(name="msd", dimension=200)
		st.session_state["vdb"]['mule'] = st.session_state['vecs'].get_or_create_collection(name="mule", dimension=1728)
		
		st.session_state["vdb"]['effnet'].create_index()
		st.session_state["vdb"]['msd'].create_index()
		st.session_state["vdb"]['mule'].create_index()
		

if "mood_scaler" not in st.session_state:
	st.session_state["mood_scaler"] = joblib.load("assets/mood_scaler.joblib")


tags = fetch_tags()

#genre_scores = conn.query("*", table = "genres").execute()

st.title("Song Searching and Similarity")


if "app_ready" not in st.session_state:
	st.write("Database still loading...")
	
else:
	
	
	
	song_names_dict = (tags["title"] + " -- " + tags["artist"].fillna('')).sort_values()
	song_names_dict = song_names_dict.to_dict()

	selected_sid = st.selectbox("Select song", song_names_dict.keys(), format_func = lambda x:song_names_dict[x], index = None, placeholder="Songs")
	inner11, inner12 = st.columns([1,1])
	
	if selected_sid:
		
		
		with inner11:
#			genre_scores = conn.query("*", table = "genres", ttl = 0).eq('sid', selected_sid).execute().data
			
			genre_scores = execute_query(conn.table("genres").select("*").eq('sid', selected_sid), ttl=0).data
			genre_scores = pd.json_normalize(genre_scores).T.iloc[1:, 0].sort_values(ascending = False).iloc[:5]
			
			
			genre_fig = px.bar(y=genre_scores.index[::-1].str.replace("___", " ").str.title().str.replace("_", " "), 
				x = genre_scores.values[::-1], 
				labels={'x': 'Genre', 'y': 'Score'}, orientation="h", 
			title='Top 5 Genres for {}'.format(song_names_dict[selected_sid]))
			
			st.plotly_chart(genre_fig, theme="streamlit", use_container_width=True)
			
		with inner12:
#			mood_scores = conn.query("*", table = "mood", ttl = 0).eq('sid', selected_sid).execute().data
			mood_scores = execute_query(conn.table("mood").select("*").eq('sid', selected_sid), ttl=0).data
#			mood_scores = pd.json_normalize(mood_scores).T.iloc[1:, 0]
			mood_scores = pd.DataFrame(mood_scores).set_index('sid')
			mood_labels  = mood_scores.columns.tolist()
			
			scaled_mood_scores = st.session_state["mood_scaler"].transform(mood_scores)[0]
			mood_scores = pd.DataFrame(mood_scores, columns = mood_labels)
			mood_scores = mood_scores.stack().reset_index().drop("sid", axis = 1).rename(columns = {"level_1":"Mood Category", 
											0:"Score"})
										
			mood_scores["Scaled Scores"] = scaled_mood_scores
										
			mood_fig = px.line_polar(mood_scores, r="Scaled Scores",theta = "Mood Category", line_close=True,
								hover_name = "Mood Category",
								hover_data = {"Mood Category":True, 
											"Score":True,
											"Scaled Scores":False}, 
								markers = True, width = 450, height = 450
								)
			mood_fig.update_traces(fill='toself', fillcolor='rgba(51, 102, 204, 0.3)')
			
			mood_fig.update_layout(
				polar=dict(
					radialaxis=dict(
						visible=False,
						showticklabels=True 
					)))
			
			# Show plot
			
			st.plotly_chart(mood_fig, 
#				use_container_width=True
			)
#			
			

	
	selected_embeddings = st.radio("Select Embeddings", options = ["effnet", "msd", "mule"], horizontal = True)
	embed_index = st.session_state['vdb'][selected_embeddings]
	top_n_sim_songs = st.radio("N Similar Songs", options = [6, 11, 21, 31], horizontal=True, format_func = lambda x:x-1)
	if selected_sid:
		out = embed_index.fetch(ids = [selected_sid])
		data = out[0][1]
		nearest = embed_index.query(data = data, include_value=True, include_metadata=True, limit = top_n_sim_songs, measure="cosine_distance")
		near_df = pd.DataFrame([(i[2]['title'], i[2]['artist'], i[2]['bpm'], i[2]['key'], i[1], i[0]) for i in nearest[1:]])
		near_df.columns = ["Title", "Artist", 'BPM', "Key", "Similarity Score", 'sid']
		near_df.set_index('sid', inplace=True)
		st.write("Nearest Neighbors to {} - {}".format(nearest[0][2]['title'], nearest[0][2]['artist']))
		frame_checkbox(near_df, key = 'search_songs')
		