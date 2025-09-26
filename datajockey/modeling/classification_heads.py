import numpy as np

import onnx
import onnxruntime as ort
import logging
from pathlib import Path
from typing import Dict, Tuple, Iterator, Optional, Any
from dataclasses import dataclass
import yaml

class EffnetClassificationHeads:

    def __init__(self, config_path):
        with open(config_path) as f:
            self.classification_cfg = yaml.safe_load(f)['effnet_moods_model_cfg']
        self.sessions = []
        providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        for m in self.classification_cfg:
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            so.enable_mem_pattern = False
            sess = ort.InferenceSession(m["path"], sess_options=so, providers=providers)  # choose providers if you want
            input_name = sess.get_inputs()[0].name
            self.sessions.append({
                "name":  m["col_name"],
                "sess":  sess,
                "input": input_name,
                "index": m["index"],
            })

    def inference(self, sid, embeddings):
        data_dump = {"sid":sid}
        for mi in self.sessions:
            outputs = mi["sess"].run(None, {mi["input"]: embeddings})[0]
            mood_vec = outputs[:, mi["index"]]
            data_dump[mi["name"]] = mood_vec

        return data_dump

    

    