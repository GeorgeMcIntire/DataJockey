import sqlite3
import numpy as np
from tqdm import tqdm
import onnx
import onnxruntime as ort
import librosa
import os
import pandas as pd
from .utils import json_opener, digit2letters, convert_array
import yaml
from inspect import getsourcefile


_this_module_file_path_ = os.path.abspath(getsourcefile(lambda: 0))
fname = os.path.join(os.path.dirname(_this_module_file_path_), "settings.yaml")
with open(fname) as f:
	spectrogram_cfg = yaml.full_load(f)["spectrogram"]
	
FFT_HOP = spectrogram_cfg["FFT_HOP"]
FFT_SIZE = spectrogram_cfg["FFT_SIZE"]
N_MELS = spectrogram_cfg["N_MELS"]


class Activator:   
	def __init__(self, input_length, model_path,pathid_dict, SR = 16000, FFT_HOP = FFT_HOP, FFT_SIZE= FFT_SIZE, N_MELS = N_MELS):
		self.FFT_HOP = FFT_HOP
		self.FFT_SIZE = FFT_SIZE
		self.N_MELS = N_MELS
		self.input_length = input_length
		self.SR = SR
		self.n_frames = librosa.time_to_frames(input_length, sr=SR, n_fft=FFT_SIZE, hop_length=FFT_HOP) + 1
		self.model = onnx.load(model_path)
		self.ort_session = ort.InferenceSession(model_path)
		self.input_name = self.model.graph.input[0].name
		self.song_files = pathid_dict.items()
		
	def spectro2batch(self, song_file:str) -> np.ndarray:
		signal,_ = librosa.load(song_file, sr = self.SR)
		audio_rep = librosa.feature.melspectrogram(y=signal,sr=self.SR,hop_length=self.FFT_HOP,n_fft=self.FFT_SIZE, n_mels=self.N_MELS).T
		audio_rep = audio_rep.astype(np.float16)
		audio_rep = np.log10(10000 * audio_rep + 1)
		last_frame = audio_rep.shape[0] - self.n_frames + 1
		first = True
		for time_stamp in range(0, last_frame, self.n_frames):
			patch = np.expand_dims(audio_rep[time_stamp : time_stamp + self.n_frames, : ], axis=0)
			if first:
				batch = patch
				first = False
			else:
				batch = np.concatenate((batch, patch), axis=0)
		return batch
	
	def _inference(self, batch:np.ndarray) -> dict:
		outputs = self.ort_session.run(None,{self.input_name: batch},)  
		o_names = [i.name for i in self.model.graph.output]
		outputs = dict(zip(o_names, outputs))
		return outputs
	
	def batch_inference(self):
		for sf, sid in tqdm(self.song_files, desc = "Effnet Genre and Embeddings Activations"):
			batch = self.spectro2batch(sf)
			outputs = self._inference(batch)
			yield (sid, sf,  outputs)



class Classifier:
	def __init__(self,model_info,new_ids,  db="/Users/georgemcintire/projects/djing/jaage.db"):
		self.model = onnx.load(model_info["model"])
		self.model_path = model_info["model"]
		self.classes = json_opener(model_info["json"])["classes"]
		self.classes = [digit2letters(i.replace(" ", "_")) for i in self.classes]
		self.table_name = self.model_path.split("/")[-1].rstrip(".onnx").replace("-", "_") + "_activations"
		self.ort_session = ort.InferenceSession(self.model_path)
		self.input_name = self.model.graph.input[0].name
		self.db = db
		self.new_ids = new_ids
		
	def _create_table(self):
		tbl_query =  "CREATE TABLE IF NOT EXISTS {} (sid VARCHAR, {})"
		cols = [i + " ARRAY" for i in self.classes]
		cols = ", ".join(cols)
		tbl_query = tbl_query.format(self.table_name, cols)
		with self.conn:
			self.cur.execute(tbl_query)
			
	def _inference(self, batch:np.ndarray) -> np.ndarray:
		outputs = self.ort_session.run(None,{self.input_name: batch})
		return outputs[0]
	
	def _create_ins_query(self) -> str:
		col_names = ", ".join(self.classes)
		q = ["?"]*len(self.classes)
		q = ",".join(q)
		ins_query = "INSERT INTO {} (sid, {}) values (?,{})".format(self.table_name, col_names, q)
		return ins_query
	
	
	def batch_inference(self, tbl):
		ins_query = self._create_ins_query()
		self.conn = sqlite3.connect(self.db,detect_types= sqlite3.PARSE_DECLTYPES)
		self.cur = self.conn.cursor()
		self._create_table()
		
		with self.conn:
			results = self.cur.execute(f"SELECT * FROM {tbl}").fetchall()
			results = [i for i in results if i[0] in self.new_ids]
			for sid, embed in tqdm(results):
				embed = convert_array(embed)
				embed = embed[0]
				preds = self._inference(embed)
				output = [sid]
				for i in range(len(self.classes)):
					array = preds[:, i]
					array = np.expand_dims(array,0)
					output.append(array)
				output = tuple(output)
				self.cur.execute(ins_query, output)
		self.conn.commit()
        
# model = keras.models.load_model("/Users/georgemcintire/projects/djing/music-audio-representations/supporting_data/model", compile=False)
# sr = 16000


# cfg_avg = Config("/Users/georgemcintire/projects/djing/music-audio-representations/supporting_data/configs/mule_embedding_average.yml")
# cfg_avg["Analysis"]["feature_transforms"][1]['EmbeddingFeature']['model_location'] = "/Users/georgemcintire/projects/djing/music-audio-representations/supporting_data/model"
# cfg_avg["Analysis"]["source_feature"]["AudioWaveform"]["input_file"]["AudioFile"]['sample_rate'] = sr

# analysis = Analysis(cfg_avg)     
# def file2embed(file_path):
#     analysis._source_feature.from_file(file_path)
#     input_feature = analysis._source_feature
#     for feature in analysis._feature_transforms:
#         feature.from_feature(input_feature)
#         input_feature = feature
#     return input_feature