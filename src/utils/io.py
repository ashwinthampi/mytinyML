import numpy as np
#save the model to a file
def save_model(path: str, W: np.ndarray, b: np.ndarray) -> None:
    np.savez(path, W=W, b=b)

#load the model from a file
def load_model(path: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["W"], data["b"]