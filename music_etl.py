from datetime import datetime
from glob import glob
from tqdm import tqdm
from pathlib import Path
import unicodedata
from essentia.standard import MusicExtractor, YamlOutput
import yaml
import shutil
import asyncio

from datajockey.io import clear_directory, unzip_and_stage, get_audio_files, wasabi_uploader, xml_loader
from datajockey.audio import normalize_audio_extract_tags
from datajockey.modeling import AudioProcessor, ONNXInferenceEngine, EffnetClassificationHeads
from datajockey.database import Databaser
from datajockey.database.vector_db import initialize_client
from datajockey.utils import init_logging
from datajockey.data_models import Metadata, Rhythm, Tonal, LowLevel, EffnetEmbeddings, EffnetGenres, EffnetMoods, MSDEmbeddings, MSDEmotions, MSDMoods, JamendoMoodTheme, jamendo_cols
from datajockey.modeling.utils import keep_gcol_dict, load_mono_16k_tensor
from datajockey.modeling.modal_models import modal_init, async_mule_gen, async_maest_gen

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)
MASTERING_PATH = Path(cfg["mastering"]["path"])
LOGGING_PATH = Path(cfg["logging"]["path"])

logger = init_logging(LOGGING_PATH)
logger.info(f"Logging initialized on {datetime.today()}.")

database = Databaser()
logger.info("DB Engine initialized.")

LOADING_DOCK_DIR = "/Volumes/Lacie/loading_dock/"
COLLECTION_DIR = "/Volumes/Lacie/collection/"
DJ_HUB_DIR = "/Volumes/Lacie/dj_hub/"

loadingdock_path = Path(LOADING_DOCK_DIR)
djhub_path = Path(DJ_HUB_DIR)
collection_path = Path(COLLECTION_DIR)

unzip_and_stage(loadingdock_path, djhub_path)
raw_files = get_audio_files(loadingdock_path, recursive=True)

logger.info(f"There are {len(raw_files)} tracks to process")

norm_files = normalize_audio_extract_tags(raw_files, reference_path=MASTERING_PATH, collection_path=collection_path)

for i in raw_files:
    try:
        shutil.move(i, djhub_path)
    except Exception as e:
        print(e)

pause = input("Dump new files into RekordBox")

music_ext = MusicExtractor(lowlevelStats=["mean"],rhythmStats=["mean"],tonalStats=["mean"],mfccStats=["mean"],gfccStats=["mean"])

xml_path = "rekordbox_data.xml"
rbox = xml_loader(xml_path)

sid2path = {}

for nf in tqdm(norm_files, desc = "Extracting Essentia Features and dumping to database"):
    path = nf.as_posix()
    features, _ = music_ext(path)
    sid = features['metadata.audio_properties.md5_encoded']
    logger.info(f"Extracted features for {path}.")
    path = collection_path.joinpath(features["metadata.tags.file_name"]).as_posix()
    rbox_data = rbox[rbox.Location_unicode ==  unicodedata.normalize("NFKC", path)].iloc[0].to_dict()
    metadata_dict = {'sid':sid, "path":path,
                     'rekordbox_id':rbox_data["TrackID"],
                    "title":rbox_data["Name"],
                    "artist":rbox_data["Artist"],
                    "genre":rbox_data["Genre"],
                    "dateadded":datetime.fromisoformat(rbox_data["DateAdded"]),
                    'length': rbox_data["TotalTime"],
                    'bpm':rbox_data["AverageBpm"],
                    'initialkey':rbox_data['Tonality']}
    metadata_dump = Metadata(**metadata_dict)
    
    lowlevel_dict = {"sid":sid}
    for ll_col in features.descriptorNames("lowlevel"):
        col_name = ll_col.replace("lowlevel.", "").replace(".", "_")
        lowlevel_dict[col_name] = features[ll_col]
    lowlevel_dump = LowLevel(**lowlevel_dict)

    rhythm_dict = {"sid":sid}
    for r_col in features.descriptorNames("rhythm"):
        col_name = r_col.replace("rhythm.", "").replace(".", "_")
        rhythm_dict[col_name] = features[r_col]
    rhythm_dump = Rhythm(**rhythm_dict)

    tonal_dict = {"sid":sid}
    for t_col in features.descriptorNames("tonal"):
        col_name = t_col.replace("tonal.", "").replace(".", "_")
        tonal_dict[col_name] = features[t_col]
    tonal_dump = Tonal(**tonal_dict)

    sid2path[sid] = path

    database.add_all([metadata_dump, lowlevel_dump,rhythm_dump, tonal_dump] )

effnet_model_path= "model_directory/onnx_models/discogs-effnet-bsdynamic-1.onnx"
effnet_spectrogram_maker = AudioProcessor(input_length=2.05)
effnet_embedder_genre_gen = ONNXInferenceEngine(model_path=effnet_model_path)

for sid, path in tqdm(sid2path.items(), desc= "Generating Effnet Embeddings and Genres and writing to DB"):
    batch = effnet_spectrogram_maker.load_and_process_audio(path)
    outputs = effnet_embedder_genre_gen.predict(batch)
    genres = outputs["activations"]
    embeddings = outputs['embeddings']
    eff_genre_data_dump = {"sid":sid}

    for i, col in keep_gcol_dict.items():
        genre_mean = genres[:, i].mean()
        eff_genre_data_dump[col] = genre_mean

    eff_genre_data_dump = EffnetGenres(**eff_genre_data_dump)

    eff_embed_data_dump = EffnetEmbeddings(sid = sid, embedding = embeddings)

    database.add_all([eff_genre_data_dump, eff_embed_data_dump])

classification_model_config_path = "datajockey/modeling/classification_head_cfg.yaml"

effnet_class_heads = EffnetClassificationHeads(config_path=classification_model_config_path)

for sid in tqdm(sid2path.keys()):
    obj = database.get(EffnetEmbeddings, sid)
    embedding = obj.embedding
    class_data_dump = effnet_class_heads.inference(sid, embedding)
    effnet_moods_dump = EffnetMoods(**class_data_dump)
    database.add(effnet_moods_dump)

msd_model_path = "model_directory/msd_models/msd-musicnn-1.onnx"
msd_spectrogram_maker = AudioProcessor(input_length=3)
msd_embedder_gen = ONNXInferenceEngine(model_path=msd_model_path, providers=["CPUExecutionProvider"])

for sid, path in tqdm(sid2path.items(), desc= "Generating MSD Embeddings and writing to DB"):
    batch = msd_spectrogram_maker.load_and_process_audio(path)
    outputs = msd_embedder_gen.predict(batch)
    embeddings = outputs['embeddings']
    msd_embeds_data_dump = MSDEmbeddings(sid = sid, embedding = embeddings)
    database.add(msd_embeds_data_dump)

deam_msd_model = "model_directory/msd_models/deam-msd-musicnn-2.onnx"
deam_msd_engine = ONNXInferenceEngine(model_path=deam_msd_model, providers=["CPUExecutionProvider"])
mirex_msd_model_path = "model_directory/msd_models/moods_mirex-msd-musicnn-1.onnx"
mirex_msd_engine = ONNXInferenceEngine(model_path=mirex_msd_model_path, providers=["CPUExecutionProvider"])

for sid in tqdm(sid2path.keys()):
    obj = database.get(MSDEmbeddings, sid)
    embedding = obj.embedding
    deam_output = deam_msd_engine.predict(embedding)['model/Identity:0']
    valence = deam_output[:, 0]
    arousal = deam_output[:, 1]
    msd_emotions_dump = MSDEmotions(sid = sid, valence = valence, arousal = arousal)
    mirex_output = mirex_msd_engine.predict(embedding)['model/Softmax']

    msd_moods_dump = MSDMoods(sid = sid, 
                              rousing = mirex_output[:, 0],
                             cheerful = mirex_output[:, 1],
                             wistful  = mirex_output[:, 2],
                             silly =  mirex_output[:, 3],
                             intense = mirex_output[:, 4])
    database.add_all([msd_emotions_dump, msd_moods_dump])

jamendo_path = "model_directory/onnx_models/mtg_jamendo_moodtheme-effnet-discogs-1.onnx"
jamendo_msd_engine = ONNXInferenceEngine(model_path=jamendo_path, providers=["CPUExecutionProvider"])

for sid in tqdm(sid2path.keys()):
    obj = database.get(EffnetEmbeddings, sid)
    embedding = obj.embedding
    outputs = jamendo_msd_engine.predict(embedding)["activations"]
    data_dump = {"sid":sid}
    for i in range(len(jamendo_cols)):
        col = jamendo_cols[i]
        data_dump[col] = outputs[:, i]
    data_dump = JamendoMoodTheme(**data_dump)
    database.add(data_dump)

chroma_client = initialize_client("chroma_data/")
effnet_vdb = chroma_client.get_collection("effnet")
msd_vdb = chroma_client.get_collection("msd")
mule_vdb = chroma_client.get_collection("mule")
maest_vdb = chroma_client.get_collection("maest")

sids = sid2path.keys()
for sid in tqdm(sids):
    meta_obj = database.get(Metadata, sid)
    ee_obj = database.get(EffnetEmbeddings, sid)
    embed_mean_pool = ee_obj.embedding.mean(axis = 0)
    effnet_vdb.upsert(ids=[sid],
                      embeddings=[embed_mean_pool.tolist()],
                      metadatas=[{"title":meta_obj.title, "artist":meta_obj.artist}])
    msd_obj = database.get(MSDEmbeddings, sid)
    msd_mean_pool = msd_obj.embedding.mean(axis = 0)
    msd_vdb.upsert(
        ids=[sid],
        embeddings=[msd_mean_pool.tolist()],
    metadatas=[{"title":meta_obj.title, "artist":meta_obj.artist}])

mule_embedder = modal_init("inspect-mule-volume", "MuleEmbedder")
maest_embedder = modal_init("inspect-maest-volume", "MaestEmbedder")
tensors4maest = [load_mono_16k_tensor(path) for path in sid2path.values()]

async def modal_async():
    mule_embeddings = await async_mule_gen(mule_embedder, sid2path)
    maest_embeds = await async_maest_gen(maest_embedder, sid2path.keys(), tensors4maest)
    return mule_embeddings, maest_embeds

mule_embeddings, maest_embeds = asyncio.run(modal_async())

for sid, embed in tqdm(mule_embeddings.items()):
    meta_obj = database.get(Metadata, sid)
    mule_vdb.upsert(
        ids=[sid],
        embeddings=[embed],
    metadatas=[{"title":meta_obj.title, "artist":meta_obj.artist}])

for sid, embed in tqdm(maest_embeds):
    meta_obj = database.get(Metadata, sid)
    maest_vdb.upsert(
        ids=[sid],
        embeddings=[embed],
        metadatas=[{"title":meta_obj.title, "artist":meta_obj.artist}])

wasabi_uploader(sid2path.values())

clear_directory(loadingdock_path)