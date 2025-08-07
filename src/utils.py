import os, random, numpy as np, torch

def set_seeds(seed: int = 42, num_threads: int = 4):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(num_threads)

def device():
    return "cpu"   # keep CPU-only per your environment
