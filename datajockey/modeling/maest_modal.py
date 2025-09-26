import os
import tempfile
import modal
from typing import List
from time import time
import numpy as np
import io
import tempfile
import soundfile as sf
import onnxruntime as ort
from typing import Literal
import torch

Pooling = Literal["cls", "mean", "cls_dist_mean", "block7_2304"]


app = modal.App("inspect-maest-volume")
volume = modal.Volume.from_name("maest_models", create_if_missing=True)

image = modal.Image.from_dockerfile("Dockerfile.modal")


@app.cls(
	image=image,
	gpu="T4",
	timeout=1200,
	cpu=4,                # Allocate 4 CPU cores
	memory=32288, 
	volumes={"/model": volume},
	scaledown_window = 300,
	max_containers=6
)
class MaestEmbedder:
	
	sr: int = modal.parameter(default=16000)
	target_seconds: int = modal.parameter(default=30)
	samples:int = modal.parameter(default=16000 * 30)

	
	@modal.enter()
	def enter(self):
		self.samples = self.sr * self.target_seconds
		self.backend = "onnx"   # or "torch", depending on model
		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		
#		import onnxruntime as ort
		MODEL_PATH = "/model/discogs-maest-30s-pw-129e-swa.ckpt"
#		self.backend = "onnx"
#		providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
#		sess = ort.InferenceSession(MODEL_PATH, providers=providers)
#		self.model = sess
#		self.in_name = sess.get_inputs()[0].name
#		self.out_name = sess.get_outputs()[0].name
		from maest import get_maest
		self.model = get_maest(arch="discogs-maest-30s-pw-129e")
		
		
		
	def _prep(self, file_bytes: bytes) -> np.ndarray:
		"""Decode → mono → float32@16k → pad/trim to 30s."""
		audio, sr = sf.read(io.BytesIO(file_bytes), always_2d=False)
		if audio.ndim == 2:
			audio = audio.mean(axis=1)
		x = audio.astype(np.float32, copy=False)
		if sr != self.sr:
			import librosa
			x = librosa.resample(x, orig_sr=sr, target_sr=self.sr).astype(np.float32, copy=False)
		if len(x) < self.samples:
			out = np.zeros(self.samples, dtype=np.float32)
			out[: len(x)] = x
			x = out
		else:
			x = x[: self.samples]
		return x
	

		
	@modal.method()
	def inference(self, sid, data) -> list[float]:
		outputs = []
		for row in data:
			logits, embeddings = self.model(row, transformer_block=6)
			outputs.append(embeddings)
		return (sid, torch.cat(outputs, 0).mean(axis = 0).tolist())
		
			