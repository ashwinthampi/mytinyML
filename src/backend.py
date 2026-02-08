#backend module for NumPy/CuPy switching
#provides a unified interface (xp) that is either numpy or cupy
#set once at startup via use_gpu()/use_cpu(), then all layers use backend.xp

import os
import sys
import numpy
import cupy

try:
    #fix CUDA paths for pip-installed nvidia packages on Windows
    #nvidia packages install DLLs/headers into site-packages/nvidia/*/bin/ and include/
    import site
    for sp in site.getsitepackages():
        nvidia_path = os.path.join(sp, "nvidia")
        if os.path.isdir(nvidia_path):
            for subdir in os.listdir(nvidia_path):
                bin_path = os.path.join(nvidia_path, subdir, "bin")
                if os.path.isdir(bin_path):
                    #add to Python DLL search (for ctypes loading)
                    os.add_dll_directory(bin_path)
                    #add to PATH so NVRTC can find its builtins DLL at runtime
                    if bin_path not in os.environ.get("PATH", ""):
                        os.environ["PATH"] = bin_path + os.pathsep + os.environ.get("PATH", "")

            #set CUDA_PATH if not already set (CuPy needs this for JIT compilation)
            nvcc_path = os.path.join(nvidia_path, "cuda_nvcc")
            if os.path.isdir(nvcc_path) and "CUDA_PATH" not in os.environ:
                os.environ["CUDA_PATH"] = nvcc_path

    _gpu_available = True
except ImportError:
    _gpu_available = False

#active array module (default: numpy)
xp = numpy

def use_gpu():
    """Switch backend to CuPy (GPU). Call before creating model."""
    global xp
    if not _gpu_available:
        raise RuntimeError("CuPy is not installed. Install with: pip install cupy-cuda12x")
    xp = cupy

def use_cpu():
    """Switch backend to NumPy (CPU)."""
    global xp
    xp = numpy

def is_gpu():
    """Check if GPU backend is active."""
    return xp is not numpy

def to_device(arr):
    """Move a numpy array to the active device (GPU or CPU)."""
    if xp is numpy:
        return arr
    return cupy.asarray(arr)

def to_numpy(arr):
    """Move an array back to numpy (CPU). Safe to call on numpy arrays."""
    if isinstance(arr, numpy.ndarray):
        return arr
    return arr.get()
