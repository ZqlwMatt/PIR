import numpy as np
import torch
import gc

def cleanup():
    gc.collect()
    torch.cuda.empty_cache()

def to8b(x):
    return np.clip(x * 255.0, 0.0, 255.0).astype(np.uint8)
