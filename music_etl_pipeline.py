import sqlite3
import logging
import zipfile
from pathlib import Path
import shutil
from glob import glob
import json

import pandas as pd
from tqdm import tqdm
import numpy as np
import taglib
import matchering as mg
from essentia.standard import MusicExtractor, YamlOutput

from project_tools.utils import (
	get_audio_files,
	get_features,
	read_json,
	adapt_array,
	convert_array,
	fix_gcols,
	s3_uploader,
	clear_directory,
	load_genre_cols
)

from project_tools.llm import metadata_extract

from project_tools.data_processing import (
    process_extracted_features,
    extract_single_feature,
    insert_feature_data,
    process_list_columns,
    insert_files_data,
    process_effnet_result,
    insert_msd_embeddings,
    process_metadata_for_database,
)
from project_tools.models import create_activator_with_gpu_support, Classifier


logging.basicConfig(
    level=logging.INFO,
    filename="song_etl_log.log",
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("array", convert_array)
conn = sqlite3.connect(
    "/Volumes/Storage/jaage.db", detect_types=sqlite3.PARSE_DECLTYPES
)
cur = conn.cursor()


def unzip_files(zip_dir: Path, move_to_dir: Path):
    for zip_path in zip_dir.glob("*.zip"):
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(zip_dir)
        logger.info(f"Unzipped: {zip_path}")
        shutil.move(zip_path, move_to_dir)


def normalize_audio_extract_tags(audio_files, reference_path: Path):
    normalized_files = []
    for file in tqdm(audio_files, desc="Mastering Songs"):
        out_stem = file.stem
        out_path = file.parent.parent / "Collection" / out_stem
        out_path = out_path.as_posix() + ".wav"

        try:
            mg.process(
                target=str(file),
                reference=str(reference_path),
                results=[mg.pcm24(out_path)],
            )
            logger.info(f"Normalized: {file.name}")

            load_tags = taglib.File(file.as_posix())
            mastered_tags = taglib.File(out_path)
            mastered_tags.tags = load_tags.tags

            extracted_metadata = metadata_extract(out_stem)
            for tag_name, tag_val in extracted_metadata.items():
                mastered_tags.tags[tag_name] = tag_val

            mastered_tags.save()

            normalized_files.append(Path(out_path))

        except Exception as e:
            logger.warning(f"Error normalizing {file}: {e}")
    return normalized_files


def extract_features(file_path: Path, output_dir: Path):
    output_dir.mkdir(exist_ok=True, parents=True)
    json_path = output_dir / "features.json"
    try:
        get_features(file_path, json_path)
        features = read_json(json_path)
        return features
    except Exception as e:
        logger.warning(f"Failed feature extraction for {file_path.name}: {e}")
        return {}


def insert_data_to_sqlite(data_dict, db_path: Path):
    with sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
        for table, df in data_dict.items():
            if isinstance(df, pd.DataFrame):
                df.to_sql(table, conn, if_exists="append", index=False)
                logger.info(f"Inserted into {table}: {len(df)} records")


loadingdock_dir = "/Volumes/Storage/Loading Dock/"
djhub_dir = "/Volumes/Storage/DJ Hub/"
collection_dir = "/Volumes/Storage/Collection/"

loadingdock_path = Path(loadingdock_dir)
djhub_path = Path(djhub_dir)
collection_dir = Path(collection_dir)

ref_file = Path("/Volumes/Storage/DJ Hub/Rayko - Magnetized (Rayko rework).wav")
features_dir = Path("temp_features")
db_path = Path("/Volumes/Storage/jaage.db")

music_ext = MusicExtractor(
    lowlevelStats=["mean", "stdev"],
    rhythmStats=["mean", "stdev", "max", "min", "median"],
    tonalStats=["mean", "stdev"],
    mfccStats=["mean", "cov"],
    gfccStats=["mean", "cov"],
)


model_paths = sorted(glob("onnx_models/*.onnx"))
model_infos = sorted(glob("onnx_models/json_info/*.json"))
effnet_models = [
    {"model": model_paths[i], "json": model_infos[i]}
    for i in range(len(model_paths))
    if "effnet" in model_paths[i] and "bsdynamic-1" not in model_paths[i]
]

with open("onnx_models/json_info/discogs-effnet-bsdynamic-1.json") as f:
	gcols = json.load(f)["classes"]
	gcols = [fix_gcols(i) for i in gcols]
	
keep_gcols = load_genre_cols()

if __name__ == "__main__":
    # main()
    unzip_files(loadingdock_path, djhub_path)
    existing_ids = set(pd.read_sql("SELECT sid FROM tags", conn).sid)
    raw_files = get_audio_files(loadingdock_path)
    norm_files = normalize_audio_extract_tags(raw_files, ref_file)
    for i in raw_files:
        shutil.move(i, djhub_path)

    extracted = []
    fileid2paths = {}

    for file_path in tqdm(norm_files):
        result = extract_single_feature(
            file_path, music_ext, existing_ids, features_dir
        )
        if result:
            json_data, file_id, path = result
            fileid2paths[file_id] = str(path)
            extracted.append(json_data)
    paths2ids = {v: str(k) for k, v in fileid2paths.items()}
    new_ids = list(paths2ids.values())
    no_list_df, list_df, meta_df = process_extracted_features(extracted)
    rows_processed: int = process_metadata_for_database(meta_df, conn)
    insert_files_data(fileid2paths, conn)
    feature_counts = insert_feature_data(no_list_df, conn)
    list_counts = process_list_columns(list_df, conn)
    activator = create_activator_with_gpu_support(
        input_length=2.05,
        model_path="onnx_models/discogs-effnet-bsdynamic-1.onnx",
        pathid_dict=paths2ids,
    )
    for file_id, file_path, outputs in tqdm(activator.batch_inference()):
        process_effnet_result(file_id, file_path, outputs, conn, gcols, keep_gcols)

    placeholders = ",".join(["?"] * len(new_ids))  # generates ?,?,?... based on length
    query = f"SELECT * FROM effnet_embeddings WHERE sid IN ({placeholders})"
    new_id_effnet_embeddings = cur.execute(query, new_ids).fetchall()
    sid2embed = {
        sid: convert_array(embed)[0] for sid, embed in new_id_effnet_embeddings
    }
    for effnet_model in tqdm(effnet_models, desc="Classifying Moods"):
        cls = Classifier(model_info=effnet_model)
        logger.info(f"Starting inference for {cls.table_name}")
        for sid, embed in sid2embed.items():
            try:
                cls.insert_predictions(cur, sid, embed)
            except Exception as e:
                logger.exception(
                    f"Failed to insert prediction for {sid} in {cls.table_name}: {e}"
                )

        conn.commit()
        logger.info(f"✅ Completed => {cls.table_name}\n")

    msd_activator = create_activator_with_gpu_support(
        input_length=3,
        model_path="msd_models/msd-musicnn-1.onnx",
        pathid_dict=paths2ids,
    )
    insert_msd_embeddings(msd_activator, cur, conn)

    msd_deam_model_info = {
        "model": "msd_models/deam-msd-musicnn-2.onnx",
        "json": "msd_models/deam-msd-musicnn-2.json",
    }
    query = f"SELECT * FROM msd_musicnn_1_embeddings WHERE sid IN ({placeholders})"
    new_id_msd_embeddings = cur.execute(query, new_ids).fetchall()
    sid2embed = {sid: convert_array(embed)[0] for sid, embed in new_id_msd_embeddings}
    cls = Classifier(model_info=msd_deam_model_info)
    logger.info(f"Starting inference for {cls.table_name}")
    for sid, embed in sid2embed.items():
        try:
            cls.insert_predictions(cur, sid, embed)
        except Exception as e:
            logger.exception(
                f"Failed to insert prediction for {sid} in {cls.table_name}: {e}"
            )

    conn.commit()
    logger.info(f"✅ Completed => {cls.table_name}\n")

    s3_uploader(fileid2paths)
    clear_directory(loadingdock_dir)
	