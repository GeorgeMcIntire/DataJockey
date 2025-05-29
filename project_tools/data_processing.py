import swifter
from pathlib import Path
from typing import List, Tuple, Dict, Union, Any
import logging
import pandas as pd
from tqdm import tqdm
from essentia.standard import  MusicExtractor, YamlOutput
import taglib 
from project_tools.utils import (
	 convert_array, adapt_array, json_opener, tag_cleaner
    
)
import sqlite3
import numpy as np

sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("array", convert_array)

logger = logging.getLogger(__name__)

def insert_files_data(
	id_to_paths: Dict[Any, str], 
	conn: sqlite3.Connection,
	table_name: str = "files"
) -> int:
	"""Insert file mapping data into database."""
	logger.info(f"Processing {len(id_to_paths)} file mappings")
	
	try:
		files_df = pd.DataFrame(list(id_to_paths.items()), columns=["sid", "file_path"])
		files_df.to_sql(table_name, conn, if_exists="append", index=False, method='multi')
		conn.commit()
		
		logger.info(f"Inserted {len(files_df)} file records")
		return len(files_df)
	
	except sqlite3.Error as e:
		logger.error(f"Database error: {e}")
		conn.rollback()
		raise

def extract_single_feature(file_path: Path, music_ext: MusicExtractor, 
							existing_ids: set, out_dir: Path):
	"""
	Extract features for a single audio file and return a tuple of (features_dict, file_id, file_path).
	
	Parameters:
		file_path (Path): Path to the audio file.
		music_ext (MusicExtractor): An initialized Essentia MusicExtractor object.
		existing_ids (set): Set of existing MD5 IDs from the database to skip duplicates.
		out_dir (Path): Directory to store intermediate JSON feature files.
		json_opener (function): Function to open and parse JSON features from file.

	Returns:
		tuple or None: (features_dict, file_id, file_path) if successful and new, None otherwise.
	"""
	try:
		features, _ = music_ext(str(file_path))
		file_id = features['metadata.audio_properties.md5_encoded']
		
		if file_id in existing_ids:
			logger.info(f"{file_path.name} already in database, skipping.")
			return None
		
		out_path = out_dir / "features.json"
		YamlOutput(filename=str(out_path), format="json")(features)
		json_data = json_opener(str(out_path))
		
		logger.info(f"Extracted features for {file_path.name}")
		return json_data, file_id, file_path
	
	except Exception as e:
		logger.warning(f"Feature extraction failed for {file_path.name}: {e}")
		return None
	
def process_extracted_features(extracted_data):
	"""
	Normalize, clean, and split extracted feature data into metadata, list-based, and scalar-based DataFrames.
	
	Parameters:
		extracted_files (list): List of feature dictionaries.
		drop_cols_path (str): Path to .pkl file containing columns to drop.
	
	Returns:
		tuple: (no_list_df, list_df, meta_df)
	"""
	# Normalize and sanitize column names
	df = pd.json_normalize(extracted_data)
	df.columns = df.columns.str.replace(".", "_")
	df.rename(columns={"metadata_audio_properties_md5_encoded": "sid"}, inplace=True)
	
	# Drop unwanted columns
	drop_cols = np.load("drop_cols.pkl", allow_pickle=True).tolist()
	df.drop(columns=drop_cols, errors="ignore", inplace=True)
	df.set_index("sid", inplace=True)
	
	# Separate metadata columns
	meta_cols = [col for col in df.columns if col.startswith("meta")]
	meta_df = df[meta_cols].copy()
	df.drop(columns=meta_cols, inplace=True)
	
	# Split list-based vs scalar columns
	is_list_col = df.iloc[0].apply(lambda x: isinstance(x, list))
	list_cols = df.columns[is_list_col]
	no_list_cols = df.columns[~is_list_col]
	
	list_df = df[list_cols].copy()
	no_list_df = df[no_list_cols].copy()
	
	return no_list_df, list_df, meta_df


def process_metadata_for_database(
	meta_df: pd.DataFrame, 
	database_connection: sqlite3.Connection
) -> int:
	"""
	Process metadata DataFrame and insert relevant columns into SQLite database.
	
	Args:
		meta_df: DataFrame containing metadata with columns potentially prefixed
		database_connection: sqlite3.Connection object
		
	Returns:
		int: Number of rows processed and inserted into database
		
	Raises:
		ValueError: If no expected columns are found in the DataFrame
		sqlite3.Error: For database connection or insertion errors
		Exception: For other processing errors
	"""
	
	# Define expected database columns upfront
	EXPECTED_TAG_COLUMNS: List[str] = [
		'length', 'gain', 'codec', 'file_name', 'bpm', 'initialkey',
		'title', 'album', 'artist', 'date', 'genre', 'label'
	]
	
	logger.info(f"Starting metadata processing for DataFrame with shape {meta_df.shape}")
	logger.debug(f"Original columns: {list(meta_df.columns)}")
	
	try:
		# Create a copy to avoid modifying original DataFrame
		processed_df: pd.DataFrame = meta_df.copy()
		logger.debug("Created DataFrame copy for processing")
		
		# Clean the data using swifter for performance
		logger.info("Applying tag_cleaner function to all DataFrame cells")
		processed_df = processed_df.swifter.applymap(tag_cleaner)
		logger.debug("Tag cleaning completed")
		
		# Extract column suffixes (remove prefixes before last underscore)
		logger.info("Processing column names - extracting suffixes after last underscore")
		original_columns: List[str] = list(processed_df.columns)
		processed_df.columns = processed_df.columns.str.split("_").map(lambda x: x[-1])
		logger.debug(f"Column transformation: {dict(zip(original_columns, processed_df.columns))}")
		
		# Standardize column naming
		if "name" in processed_df.columns:
			logger.info("Renaming 'name' column to 'file_name'")
			processed_df.rename(columns={"name": "file_name"}, inplace=True)
			
		# Select only columns that exist in both DataFrame and expected schema
		available_columns: List[str] = [
			col for col in EXPECTED_TAG_COLUMNS if col in processed_df.columns
		]
		missing_columns: List[str] = [
			col for col in EXPECTED_TAG_COLUMNS if col not in processed_df.columns
		]
		
		logger.info(f"Available columns for database insert: {available_columns}")
		if missing_columns:
			logger.warning(f"Expected columns not found in DataFrame: {missing_columns}")
			
		if not available_columns:
			error_msg = (
				f"No expected columns found in processed DataFrame. "
				f"Available: {list(processed_df.columns)}, "
				f"Expected: {EXPECTED_TAG_COLUMNS}"
			)
			logger.error(error_msg)
			raise ValueError(error_msg)
			
		# Insert filtered data into SQLite database
		rows_to_insert: int = len(processed_df)
		logger.info(f"Inserting {rows_to_insert} rows into 'tags' table in SQLite database")
		
		# SQLite-specific insertion with proper error handling
		try:
			processed_df[available_columns].to_sql(
				name="tags", 
				con=database_connection, 
				if_exists="append",
				index=True,  # Explicitly exclude index
				method='multi'  # Use multi-row insert for better performance
			)
			
			# Commit the transaction
			database_connection.commit()
			logger.info(f"Successfully inserted and committed {rows_to_insert} rows to SQLite database")
			
		except sqlite3.Error as db_error:
			logger.error(f"SQLite database error during insertion: {str(db_error)}", exc_info=True)
			database_connection.rollback()
			logger.info("Transaction rolled back due to database error")
			raise
			
		return rows_to_insert
	
	except sqlite3.Error as db_error:
		logger.error(f"SQLite database error: {str(db_error)}", exc_info=True)
		raise
	except Exception as e:
		logger.error(f"Error during metadata processing: {str(e)}", exc_info=True)
		raise
def process_list_columns(
	list_data: pd.DataFrame, 
	conn: sqlite3.Connection
) -> Dict[str, int]:
	"""Process list-type columns into separate tables."""
	logger.info(f"Processing {len(list_data.columns)} list columns")
	results = {}
	
	try:
		for col in tqdm(list_data.columns, desc="Processing list columns"):
			# Expand list column to series
			expanded = list_data[col].swifter.apply(pd.Series)
			expanded.columns = f"{col}_" + expanded.columns.astype(str)
			
			# Insert to database
			table_name = f"{col}_tbl"
			expanded.to_sql(table_name, conn, if_exists="append", method='multi')
			results[col] = len(expanded)
			
			logger.debug(f"Processed {col} -> {table_name}")
			
		conn.commit()
		logger.info("All list columns processed successfully")
		return results
	
	except sqlite3.Error as e:
		logger.error(f"Database error: {e}")
		conn.rollback()
		raise
		
		
		
def get_categorized_columns(columns: pd.Index) -> Dict[str, Dict[str, Any]]:
	"""Categorize columns by feature type prefix."""
	categories = {
		'tonal': {'prefix': 'tonal', 'table': 'tonal_features'},
		'lowlevel': {'prefix': 'lowlevel', 'table': 'lowlevel_features'},
		'rhythm': {'prefix': 'rhyt', 'table': 'rhythm_features'}  # Note: 'rhyt' not 'rhythm'
	}
	
	result = {}
	for category, info in categories.items():
		cols = columns[columns.str.startswith(info['prefix'])].tolist()
		if cols:
			result[category] = {'columns': cols, 'table': info['table']}
			logger.debug(f"Found {len(cols)} {category} columns")
			
	return result

def insert_feature_data(feature_data: pd.DataFrame, conn: sqlite3.Connection) -> Dict[str, int]:
	"""Insert categorized feature data into respective tables."""
	logger.info(f"Processing feature data with shape {feature_data.shape}")
	
	categorized = get_categorized_columns(feature_data.columns)
	results = {}
	
	try:
		for category, info in categorized.items():
			columns = info['columns']
			table_name = info['table']
			
			logger.info(f"Inserting {len(columns)} {category} features")
			category_df = feature_data[columns]
			category_df.to_sql(table_name, conn, if_exists="append", method='multi')
			results[category] = len(category_df)
			
		conn.commit()
		logger.info("All feature categories inserted successfully")
		return results
	
	except sqlite3.Error as e:
		logger.error(f"Database error: {e}")
		conn.rollback()
		raise
		
def process_effnet_result(sid,file_path, output, conn, gcols, keep_gcols):
	"""
	Process a single inference result and store:
	- Genre activations
	- Mean activations
	- Embeddings
	"""
	try:
		genre_acts = output["activations"]
		embeds = output["embeddings"]
		
		# Expand activations into list of 1D arrays
		expanded = [np.expand_dims(genre_acts[:, i], 0) for i in range(genre_acts.shape[1])]
		genre_df = pd.DataFrame(index=[sid], data=[expanded], columns=gcols)
		genre_df.index.rename("sid", inplace=True)
		
		# Filter and compute mean
		genre_df = genre_df[keep_gcols]
		genre_df_mean = genre_df.applymap(lambda x: x[0].mean())
		
		with conn:
			conn.execute(
				"INSERT INTO effnet_embeddings (sid, effnet_embedding) VALUES (?, ?)",
				(sid, np.expand_dims(embeds, 0))
			)
			genre_df.to_sql("effnet_genres", con=conn, if_exists="append", index=True)
			genre_df_mean.to_sql("effnet_genres_mean", con=conn, if_exists="append", index=True)
			
		logger.info(f"Processed and saved effnet data for sid={sid} and file= {file_path}")
		
	except Exception as e:
		logger.warning(f"Failed to process result for sid={sid}: {e}")
		
def insert_msd_embeddings(activator, cur, conn, table_name="msd_musicnn_1_embeddings"):
	ins_query = f"INSERT INTO {table_name} (sid, msd_embeddings) VALUES (?, ?)"
	inserted_count = 0
	
	for sid, sf, output in activator.batch_inference():
		try:
			embeds = output["embeddings"]
			embeds = np.expand_dims(embeds, axis=0)  # Ensure 2D array for storage
			cur.execute(ins_query, (sid, embeds))
			inserted_count += 1
		except Exception as e:
			logger.exception(f"Failed to insert embedding for sid: {sid} — {e}")
			
	conn.commit()
	logger.info(f"Inserted {inserted_count} embeddings into {table_name}.")
	
def run_model_on_embeddings(cls, cur, conn, tbl, new_ids):
	ins_query = cls._create_ins_query()
	
	with conn:
		logger.info(f"Querying embeddings from {tbl} for {len(new_ids)} new IDs...")
		results = cur.execute(f"SELECT * FROM {tbl}").fetchall()
		results = [r for r in results if r[0] in new_ids]
		logger.info(f"Retrieved {len(results)} matching embeddings.")
		
		for sid, embed in tqdm(results, desc=f"Running inference for {cls.table_name}"):
			try:
				embed = convert_array(embed)[0]  # Assumes convert_array returns a batch
				preds = cls._inference(embed)
				
				output = [sid] + [np.expand_dims(preds[:, i], axis=0) for i in range(len(cls.classes))]
				cur.execute(ins_query, tuple(output))
				
			except Exception as e:
				logger.exception(f"Failed to process SID {sid}: {e}")
				
	conn.commit()
	logger.info(f"Inference and insertion completed for {cls.table_name}.")