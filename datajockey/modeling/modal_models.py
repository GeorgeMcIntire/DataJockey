import modal
import numpy as np
import torch
import torchaudio



def audio_bytes_gen(paths):
    for path in paths:
        with open(path, "rb") as f:
            yield f.read()

async def async_mule_gen(modal_obj, id_to_path):
    ids = list(id_to_path.keys())
    paths = list(id_to_path.values())
    
    i = 0
    
    embeddings = [emb async for emb in modal_obj.inference.map.aio(audio_bytes_gen(paths))]
    return dict(zip(ids,embeddings))


def modal_init(app_name, class_name):
    modal_obj = modal.Cls.from_name(app_name, class_name)()
    return modal_obj

async def async_maest_gen(modal_obj, sid, data):
    output = [emb async for emb in modal_obj.inference.map.aio(sid, data)]
    return output