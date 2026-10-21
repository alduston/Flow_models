import random
import numpy as np
import torch

def capture_rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])

def restore_rng_state(s):
    random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['torch'].cpu())
    if torch.cuda.is_available():torch.cuda.set_rng_state_all([t.cpu() for t in s['cuda']])
