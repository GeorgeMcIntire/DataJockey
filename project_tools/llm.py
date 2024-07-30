from typing import Optional
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from tqdm import tqdm
import json
import openai
import sqlite3
from pathlib import Path
from openai import OpenAI
import numpy as np
import os
from inspect import getsourcefile

openai_key_filepath = "/Users/georgemcintire/projects/djing/openaikey.txt"

with open(openai_key_filepath) as f:
	api_key = f.read()
	
system_template = """You are a DJ digital assistant tasked with extracting song metadata from file paths. 
From the inputted text you will extract song information such as artist, title, album, and remixer. 
Remember that not every file path will have all four of those attributes. If the input does not have a certain
attribute, then return an empty string. The user will provide text delimited by three dashes.

Here are three examples

input:02 Kiki Gyan_ Sexy Dancer (Combo Edit)_PN
output: artist:Kiki Gyan, title:Sexy Dancer, remixer:Combo Edit, album:null

input: Parisian Soul - XPRESS Edits Vol.4 - 01 Love You Madly (Parisian Soul Re-Edit)_PN
output: artist:Parisian Soul, title:Love You Madly (Parisian Soul Re-Edit), remixer:Parisian Soul Re-Edit, album:XPRESS Edits Vol.4

input: 1-4_M_International_-_Space_Operator_(Donato_Dozzy_Cadillac_Rhythms_Reshape)_PN
output: artist:4 M International, title:Space Operator (Donato Dozzy Cadillac Rhythms Reshape), remixer:Donato Dozzy Cadillac Rhythms Reshape, album:Space Operator


---{text}---
"""

prompt_template = ChatPromptTemplate.from_messages([("system", system_template), ("user", "{text}")])
llm = ChatOpenAI(model_name = "gpt-4", api_key=api_key, temperature = 0.0)

class Song(BaseModel):
	"""Metadata for a song"""
	
	title: str = Field(default="", description="The name of the song")
	artist: Optional[str] = Field(default="", description="The name of the artist who made the song")
	remixer: Optional[str] = Field(default="", description="The name of the artist who made a remix or edit of the song")
	album: Optional[str] = Field(default = "", description= "The name of the album the song is on")

runnable = prompt_template | llm.with_structured_output(schema=Song)

def metadata_extract(song_path:str):
	result = runnable.invoke({"text":song_path})
	output = {"ARTIST":[result.artist], "TITLE": [result.title], "ALBUM": [result.album], "REMIXER":[result.remixer]}
	return output
	