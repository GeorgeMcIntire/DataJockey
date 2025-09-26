import zipfile
import shutil
from pathlib import Path
import logging
import itertools
from typing import List
import pandas as pd
import xmltodict
from urllib import parse
from tqdm import tqdm
import unicodedata
import os
import boto3
from botocore.client import Config as BotoConfig
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)


SUPPORTED_EXTS = [".wav", ".mp3", ".aiff", ".flac", ".aif"]



def xml_loader(xml_path):
    with open(xml_path) as xml_file:
        data_dict = xmltodict.parse(xml_file.read())
        collection = data_dict["DJ_PLAYLISTS"]["COLLECTION"]
        rbox = pd.json_normalize(collection["TRACK"])
        rbox.columns = rbox.columns.str.strip("@")
        rekord_cols = ['TrackID', 'Name', 'Artist', "Album", "Genre", 'DateAdded', "TotalTime", "AverageBpm",
                "Location", "TEMPO", "Tonality"]
        rbox = rbox[rekord_cols]
        rbox["Location_fixed"] = "/"+rbox.Location.str.lstrip("file://localhost").apply(parse.unquote)
        rbox["Location_unicode"] = rbox.Location_fixed.apply(lambda x: unicodedata.normalize("NFKC", x))
    return rbox


def get_audio_files(directory: Path, recursive: bool = False) -> List[Path]:
    """
    Return all audio files in a directory (optionally including subdirectories).

    Args:
        directory: Path to search.
        recursive: Whether to search subdirectories.

    Returns:
        A sorted list of Path objects pointing to audio files.
    """
    pattern = "**/*" if recursive else "*"
    files = [
        f for f in directory.glob(pattern)
        if f.suffix.lower() in SUPPORTED_EXTS and not f.name.startswith("._")
    ]
    return sorted(files)

def _is_real_zip(p: Path) -> bool:
    # Skip hidden & AppleDouble
    if p.name.startswith(".") or p.name.startswith("._"):
        return False
    # Sanity check: exists, file, and is a valid zip container
    return p.is_file() and zipfile.is_zipfile(p)

def unzip_and_stage(zip_dir: Path, move_to_dir: Path) -> None:

    for zip_path in sorted(zip_dir.glob("*.zip")):
        # Extra guard for weird files
        if not _is_real_zip(zip_path):
            logger.debug("Skipping non-zip or AppleDouble: %s", zip_path.name)
            continue

        try:
            logger.info("Extracting %s", zip_path.name)
            with zipfile.ZipFile(zip_path, "r") as zf:
                _safe_extractall(zf, zip_dir)

            # Move original archive after successful extract
            target = move_to_dir / zip_path.name
            if target.exists():
                logger.warning("Overwriting existing %s", target)
                target.unlink()
            shutil.move(str(zip_path), str(move_to_dir))

        except zipfile.BadZipFile:
            logger.error("Corrupt zip file: %s", zip_path)
        except FileNotFoundError:
            logger.warning("Zip vanished during processing: %s", zip_path)
        except Exception as e:
            logger.exception("Unexpected error on %s: %s", zip_path, e)

def _safe_extractall(zf: zipfile.ZipFile, dest: Path) -> None:
    # Avoid zip slip by validating members
    for member in zf.infolist():
        out_path = dest / member.filename
        if not str(out_path.resolve()).startswith(str(dest.resolve())):
            raise RuntimeError(f"Unsafe path in zip: {member.filename}")
        zf.extract(member, path=dest)

def wasabi_uploader(files_list):

    WASABI_ACCESS_KEY = os.getenv("WASABI_ACCESS_KEY", "")
    WASABI_SECRET_KEY = os.getenv("WASABI_SECRET_KEY", "")
    WASABI_REGION = os.getenv("WASABI_REGION")
    WASABI_ENDPOINT = os.getenv("WASABI_ENDPOINT")
    WASABI_BUCKET = os.getenv("WASABI_BUCKET") 

    boto_cfg = BotoConfig(s3={"addressing_style": "virtual"},
                          retries={"max_attempts": 10, "mode": "standard"},
                          request_checksum_calculation="when_required",
                          response_checksum_validation="when_required",
                          signature_version="s3v4",
                          region_name=WASABI_REGION)
    

    wasabi = boto3.client(
        "s3",
        aws_access_key_id=WASABI_ACCESS_KEY,
        aws_secret_access_key=WASABI_SECRET_KEY,
        endpoint_url=WASABI_ENDPOINT,
        config=boto_cfg)
    logger.info(f"Starting upload of {len(files_list)} files to S3 bucket '{WASABI_BUCKET}'.")
    for song_path in tqdm(files_list):
        key = "songs/" + Path(song_path).name
        try:
            wasabi.upload_file(song_path, WASABI_BUCKET, key)
        except Exception as e:
            print(e)
    
    logger.info(f"Completed upload of {len(files_list)} files to S3.")

def clear_directory(directory_path):
    try:
        # List all files in the directory
        files = os.listdir(directory_path)

        # Iterate over each file and delete it
        for file_name in files:
            file_path = os.path.join(directory_path, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
#                 print(f"Deleted: {file_path}")

        logger.info(f"All files in {directory_path} have been deleted.")

    except Exception as e:
        print(f"An error occurred: {e}")