import streamlit as st
from st_supabase_connection import SupabaseConnection, execute_query
import pandas as pd
import joblib
import vecs
from supabase import create_client, Client
import numpy as np
from dash_utils import top_value_tables_dict, frame_checkbox, cols_dict
import plotly_express as px
from project_tools.utils import convert_array
import os


conn = st.connection("supabase",type=SupabaseConnection)
#print(dir(conn))

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
		



@st.cache_resource
def load_umap_obj():
	umapper = joblib.load("/Users/georgemcintire/projects/djing/dashboard/assets/umap_obj.joblib")
	return umapper

umapper = load_umap_obj()

@st.cache_data
def fetch_mule_embeddings():
	sids = execute_query(conn.table("tags").select("sid"), ttl = 0).data
	sids = [i["sid"] for i in sids]
	mule_embeds = []
	
	for i in sids:
		row = {}
		data = st.session_state["vdb"]['mule'].fetch(ids = [i])[0]
		if data == []:
			continue
#		print(data[2])
		emb = data[1]
		umapped = umapper.transform(emb.reshape(1, -1))
		
		row["sid"] = i
		row["umap1"] = umapped[0][0]
		row["umap2"] = umapped[0][1]
		row["bpm"] = data[2]["bpm"]
		row["key"] = data[2]["key"]
		row["title"] = data[2]["title"]
		row["aritst"] = data[2]["artist"]
		
		mule_embeds.append(row)
	
	return pd.DataFrame(mule_embeds)
	

mule_embeds = fetch_mule_embeddings()

fig = px.scatter(mule_embeds, x = "umap1", y = "umap2", hover_name="title")
st.plotly_chart(fig)

#st.write("under construction...")