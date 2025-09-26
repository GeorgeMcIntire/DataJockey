import numpy as np
from tqdm import tqdm
import onnx
import onnxruntime as ort
import librosa
import os
import pandas as pd
import yaml
import logging
from pathlib import Path
from typing import Dict, Tuple, Iterator, Optional, Any
from dataclasses import dataclass, field
import yaml


logger = logging.getLogger(__name__)

class AudioProcessor:
    """Handles audio loading and spectrogram generation."""
    def __init__(self, input_length):
        self.fft_hop: int = 256
        self.fft_size: int = 512
        self.n_mels: int = 96
        self.sample_rate: int = 16000
        self.input_length = input_length
        logger.info(f"AudioProcessor initialized")
		
    def load_and_process_audio(self, audio_path: str) -> np.ndarray:
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
                hop_length=self.fft_hop,
                n_fft=self.fft_size,
                n_mels=self.n_mels
            ).T

            # Convert to log scale and optimize data type
            mel_spec = mel_spec.astype(np.float32)  # float16 can cause precision issues
            mel_spec = np.log10(10000 * mel_spec + 1)

            # Create patches
            return self._create_patches(mel_spec, self.input_length)
		
        except Exception as e:
            logger.error(f"Error processing audio {audio_path}: {e}")
            raise
    def _create_patches(self, mel_spec: np.ndarray, input_length: float) -> np.ndarray:
        """Create overlapping patches from mel-spectrogram."""
        n_frames = librosa.time_to_frames(
            input_length, 
            sr=self.sample_rate, 
            n_fft=self.fft_size, 
            hop_length=self.fft_hop
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
    """Handles ONNX model loading and inference (prefers CoreML on Apple Silicon)."""

    def __init__(self, model_path: str, providers: Optional[list] = None):

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.enable_mem_pattern = False
        # Verbose logs help when CoreML fails

        available = ort.get_available_providers()

        default_providers = ["CPUExecutionProvider"]
        if providers is None and "CoreMLExecutionProvider" in available:
            # Try CoreML first, then CPU
            default_providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        elif providers is not None:
            default_providers = providers

        # Try to create a CoreML session, fall back to CPU on failure
        try:
            self.session = ort.InferenceSession(
                str(self.model_path), sess_options=so, providers=default_providers
            )
        except Exception as e:
            logger.warning(f"CoreML session failed ({e}); falling back to CPU.")
            self.session = ort.InferenceSession(
                str(self.model_path), sess_options=so, providers=["CPUExecutionProvider"]
            )

        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]


    def predict(self, batch: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Run inference on a batch of data.

        Ensures float32 & contiguous input (CoreML/ORT can be picky about dtypes).
        """
        try:
            if not isinstance(batch, np.ndarray):
                batch = np.asarray(batch)
            if batch.dtype != np.float32:
                batch = batch.astype(np.float32, copy=False)
            if not batch.flags.c_contiguous:
                batch = np.ascontiguousarray(batch, dtype=np.float32)
            
            outputs = self.session.run(None, {self.input_name: batch})
            output_dict = dict(zip(self.output_names, outputs))
            return output_dict
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise