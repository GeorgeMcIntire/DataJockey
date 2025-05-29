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
import logging
from pathlib import Path
from typing import Dict, Tuple, Iterator, Optional, Any
from dataclasses import dataclass
import yaml

logger = logging.getLogger(__name__)

_this_module_file_path_ = os.path.abspath(getsourcefile(lambda: 0))
fname = os.path.join(os.path.dirname(_this_module_file_path_), "settings.yaml")
with open(fname) as f:
	spectrogram_cfg = yaml.full_load(f)["spectrogram"]
	
FFT_HOP = spectrogram_cfg["FFT_HOP"]
FFT_SIZE = spectrogram_cfg["FFT_SIZE"]
N_MELS = spectrogram_cfg["N_MELS"]


@dataclass
class SpectrogramConfig:
	"""Configuration for spectrogram generation."""
	fft_hop: int
	fft_size: int
	n_mels: int
	
	@classmethod
	def from_yaml(cls, config_path: str = "project_tools/settings.yaml") -> 'SpectrogramConfig':
		"""Load configuration from YAML file."""
		
		try:
			with open(config_path) as f:
				config = yaml.safe_load(f)["spectrogram"]
				
			return cls(
				fft_hop=config["FFT_HOP"],
				fft_size=config["FFT_SIZE"],
				n_mels=config["N_MELS"]
			)
		except (FileNotFoundError, KeyError) as e:
			logger.error(f"Failed to load config from {config_path}: {e}")
			# Return sensible defaults
			return cls(fft_hop=512, fft_size=1024, n_mels=128)
class AudioProcessor:
	"""Handles audio loading and spectrogram generation."""
	
	def __init__(self, config: SpectrogramConfig, sample_rate: int = 16000):
		self.config = config
		self.sample_rate = sample_rate
		logger.info(f"AudioProcessor initialized: SR={sample_rate}, config={config}")
		
	def load_and_process_audio(self, audio_path: str, input_length: float) -> np.ndarray:
		"""
		Load audio file and convert to mel-spectrogram patches.
		
		Args:
			audio_path: Path to audio file
			input_length: Length of each input segment in seconds
			
		Returns:
			Batch of spectrogram patches as numpy array
		"""
		try:
			# Load audio
			signal, _ = librosa.load(audio_path, sr=self.sample_rate)
			logger.debug(f"Loaded audio: {audio_path}, duration: {len(signal)/self.sample_rate:.2f}s")
			
			# Generate mel-spectrogram
			mel_spec = librosa.feature.melspectrogram(
				y=signal,
				sr=self.sample_rate,
				hop_length=self.config.fft_hop,
				n_fft=self.config.fft_size,
				n_mels=self.config.n_mels
			).T
			
			# Convert to log scale and optimize data type
			mel_spec = mel_spec.astype(np.float32)  # float16 can cause precision issues
			mel_spec = np.log10(10000 * mel_spec + 1)
			
			# Create patches
			return self._create_patches(mel_spec, input_length)
		
		except Exception as e:
			logger.error(f"Error processing audio {audio_path}: {e}")
			raise
	def _create_patches(self, mel_spec: np.ndarray, input_length: float) -> np.ndarray:
		"""Create overlapping patches from mel-spectrogram."""
		n_frames = librosa.time_to_frames(
			input_length, 
			sr=self.sample_rate, 
			n_fft=self.config.fft_size, 
			hop_length=self.config.fft_hop
		) + 1
		
		if mel_spec.shape[0] < n_frames:
			logger.warning(f"Audio too short: {mel_spec.shape[0]} < {n_frames} frames")
			# Pad with zeros if audio is too short
			padding = n_frames - mel_spec.shape[0]
			mel_spec = np.pad(mel_spec, ((0, padding), (0, 0)), mode='constant')
			
		last_frame = mel_spec.shape[0] - n_frames + 1
		
		# Pre-allocate batch array for efficiency
		n_patches = (last_frame + n_frames - 1) // n_frames  # Ceiling division
		batch = np.empty((n_patches, n_frames, mel_spec.shape[1]), dtype=mel_spec.dtype)
		
		patch_idx = 0
		for time_stamp in range(0, last_frame, n_frames):
			batch[patch_idx] = mel_spec[time_stamp:time_stamp + n_frames]
			patch_idx += 1
			
		return batch[:patch_idx]

class ONNXInferenceEngine:
	"""Handles ONNX model loading and inference."""
	
	def __init__(self, model_path: str, providers: Optional[list] = None):
		"""
		Initialize ONNX inference engine.
		
		Args:
			model_path: Path to ONNX model file
			providers: List of execution providers (e.g., ['CUDAExecutionProvider', 'CPUExecutionProvider'])
		"""
		self.model_path = Path(model_path)
		
		if not self.model_path.exists():
			raise FileNotFoundError(f"Model file not found: {model_path}")
			
		try:
			# Load ONNX model for metadata
			self.onnx_model = onnx.load(str(self.model_path))
			
			# Create inference session with optimizations
			sess_options = ort.SessionOptions()
			sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
			
			if providers is None:
				providers = ['CPUExecutionProvider']  # Default to CPU
				
			self.session = ort.InferenceSession(
				str(self.model_path),
				sess_options,
				providers=providers
			)
			
			# Cache input/output names
			self.input_name = self.onnx_model.graph.input[0].name
			self.output_names = [output.name for output in self.onnx_model.graph.output]
			
			logger.info(f"ONNX model loaded: {self.model_path}")
			logger.info(f"Available providers: {self.session.get_providers()}")
			
		except Exception as e:
			logger.error(f"Failed to load ONNX model: {e}")
			raise
			
	def predict(self, batch: np.ndarray) -> Dict[str, np.ndarray]:
		"""
		Run inference on a batch of data.
		
		Args:
			batch: Input batch for inference
			
		Returns:
			Dictionary mapping output names to results
		"""
		try:
			outputs = self.session.run(None, {self.input_name: batch})
			return dict(zip(self.output_names, outputs))
		except Exception as e:
			logger.error(f"Inference failed: {e}")
			raise

class AudioActivator:
	"""Main class for audio processing and model inference."""
	
	def __init__(
		self, 
		input_length: float,
		model_path: str,
		pathid_dict: Dict[str, str],
		sample_rate: int = 16000,
		onnx_providers: Optional[list] = None
	):
		"""
		Initialize AudioActivator.
		
		Args:
			input_length: Length of audio segments in seconds
			model_path: Path to ONNX model
			pathid_dict: Dictionary mapping file paths to IDs
			sample_rate: Audio sample rate
			config_path: Path to configuration YAML file
			onnx_providers: ONNX execution providers
		"""
		self.input_length = input_length
		self.pathid_dict = pathid_dict
		
		# Load configuration
		self.config = SpectrogramConfig.from_yaml()
		
		# Initialize components
		self.audio_processor = AudioProcessor(self.config, sample_rate)
		self.inference_engine = ONNXInferenceEngine(model_path, onnx_providers)
		
		logger.info(f"AudioActivator initialized with {len(pathid_dict)} files")
		
	def process_single_file(self, file_path: str) -> Tuple[str, Dict[str, np.ndarray]]:
		"""
		Process a single audio file through the complete pipeline.
		
		Args:
			file_path: Path to audio file
			
		Returns:
			Tuple of (file_id, inference_results)
		"""
		file_id = self.pathid_dict[str(file_path)]
		
		try:
			# Process audio to spectrogram patches
			batch = self.audio_processor.load_and_process_audio(file_path, self.input_length)
			
			# Run inference
			outputs = self.inference_engine.predict(batch)
			
			logger.debug(f"Processed {file_path}: {batch.shape[0]} patches")
			return file_id, outputs
		
		except Exception as e:
			logger.error(f"Failed to process {file_path}: {e}")
			raise
			
	def batch_inference(self) -> Iterator[Tuple[str, str, Dict[str, np.ndarray]]]:
		"""
		Process all files and yield results.
		
		Yields:
			Tuple of (file_id, file_path, inference_results)
		"""
		total_files = len(self.pathid_dict)
		logger.info(f"Starting batch inference on {total_files} files")
		
		with tqdm(self.pathid_dict.items(), desc="Processing audio files") as pbar:
			for  file_path, file_id in pbar:
				try:
					pbar.set_description(f"Processing {Path(file_path).name}")
					
					file_id_result, outputs = self.process_single_file(file_path)
					yield file_id_result, file_path, outputs
					
				except Exception as e:
					logger.error(f"Skipping {file_path} due to error: {e}")
					print(e)
					continue  # Continue with next file instead of failing completely

def create_activator_with_gpu_support(
	input_length: float,
	model_path: str, 
	pathid_dict: Dict[str, str]
) -> AudioActivator:
	"""Create AudioActivator with GPU support if available."""
	
	# Try GPU first, fallback to CPU
	providers = []
	if ort.get_device() == 'GPU':
		providers.extend(['CUDAExecutionProvider', 'ROCMExecutionProvider'])
	providers.append('CPUExecutionProvider')
	
	return AudioActivator(
		input_length=input_length,
		model_path=model_path,
		pathid_dict=pathid_dict,
		onnx_providers=providers
	)



class Classifier:
	def __init__(self, model_info):
		self.model_path = model_info["model"]
		self.classes = json_opener(model_info["json"])["classes"]
		self.classes = [digit2letters(cls.replace(" ", "_")) for cls in self.classes]
		self.table_name = (
			self.model_path.split("/")[-1].rstrip(".onnx").replace("-", "_") + "_activations"
		)
		self.model = onnx.load(self.model_path)
		self.ort_session = ort.InferenceSession(self.model_path)
		self.input_name = self.model.graph.input[0].name
		
	def _inference(self, batch: np.ndarray) -> np.ndarray:
		outputs = self.ort_session.run(None, {self.input_name: batch})
		return outputs[0]
	
	def _create_ins_query(self) -> str:
		col_names = ", ".join(self.classes)
		q_marks = ",".join(["?"] * len(self.classes))
		return f"INSERT INTO {self.table_name} (sid, {col_names}) VALUES (?, {q_marks})"
	
	def insert_predictions(self, cursor, sid: str, embedding: np.ndarray):
		preds = self._inference(embedding)
		values = [sid]
		for i in range(len(self.classes)):
			array = preds[:, i]
			values.append(np.expand_dims(array, 0))
		cursor.execute(self._create_ins_query(), tuple(values))
		logger.debug(f"Inserted activations for sid: {sid}")