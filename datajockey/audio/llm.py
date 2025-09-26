import os
from dotenv import load_dotenv
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import json

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
	
SYSTEM_TEMPLATE = """You are a DJ digital assistant tasked with extracting song metadata from file paths.
From the input text, extract: artist, title, album, remixer. 
If an attribute is missing, return an empty string.

Examples:

input: 02 Kiki Gyan_ Sexy Dancer (Combo Edit)_PN
output: artist:Kiki Gyan, title:Sexy Dancer, remixer:Combo Edit, album:

input: Parisian Soul - XPRESS Edits Vol.4 - 01 Love You Madly (Parisian Soul Re-Edit)_PN
output: artist:Parisian Soul, title:Love You Madly (Parisian Soul Re-Edit), remixer:Parisian Soul Re-Edit, album:XPRESS Edits Vol.4

input: 1-4_M_International_-_Space_Operator_(Donato_Dozzy_Cadillac_Rhythms_Reshape)_PN
output: artist:4 M International, title:Space Operator (Donato Dozzy Cadillac Rhythms Reshape), remixer:Donato Dozzy Cadillac Rhythms Reshape, album:Space Operator

---{text}---
"""

prompt_template = ChatPromptTemplate.from_messages([
	("system", SYSTEM_TEMPLATE),
	("user", "{text}")
])

llm = ChatOpenAI(
	model_name="gpt-4o-mini",
	api_key=api_key,
	temperature=0.0,
)

class Song(BaseModel):
	"""Metadata for a song"""
	artist: str = Field(default="", description="The song's primary artist")
	title: str = Field(default="", description="The name of the song")
	album: str = Field(default="", description="The album name")
	remixer: str = Field(default="", description="The remix/edit artist if applicable")
	
runnable = prompt_template | llm.with_structured_output(schema=Song)

def metadata_extract(song_path: str) -> dict[str, list[str]]:
	"""Extract structured song metadata from a file path using the LLM."""
	try:
		result: Song = runnable.invoke({"text": song_path})
		return {
			"ARTIST": [result.artist],
			"TITLE": [result.title],
			"ALBUM": [result.album],
			"REMIXER": [result.remixer],
		}
	except Exception as e:
		# Log error upstream; return safe empty defaults
		return {"ARTIST": [""], "TITLE": [""], "ALBUM": [""], "REMIXER": [""]}