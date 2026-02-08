#input/output utilities for model serialization
#save and load model parameters and optional metadata to/from numpy .npz files
#metadata is stored as a json string in the "__metadata__" key for backward compatibility

import json
import numpy as np

#save the model parameters and optional metadata to a file
def save_model(path: str, params: dict[str, np.ndarray], metadata: dict | None = None) -> None:
    save_dict = dict(params)
    if metadata is not None:
        save_dict["__metadata__"] = np.array(json.dumps(metadata))
    np.savez(path, **save_dict)

#load the model parameters from a file (strips metadata for backward compatibility)
def load_model(path: str) -> dict[str, np.ndarray]:
    data = dict(np.load(path, allow_pickle=False))
    data.pop("__metadata__", None)
    return data

#load the model parameters and metadata from a file
def load_model_with_metadata(path: str) -> tuple[dict[str, np.ndarray], dict | None]:
    data = dict(np.load(path, allow_pickle=False))
    metadata = None
    if "__metadata__" in data:
        metadata = json.loads(str(data.pop("__metadata__")))
    return data, metadata
