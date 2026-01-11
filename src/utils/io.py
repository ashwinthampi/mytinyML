#input/output utilities for model serialization
#save and load model parameters to/from numpy .npz files

import numpy as np
#save the model to a file
def save_model(path: str, params: dict[str, np.ndarray]) -> None:
    np.savez(path, **params)

#load the model from a file
def load_model(path: str) -> dict[str, np.ndarray]:
    return dict(np.load(path))