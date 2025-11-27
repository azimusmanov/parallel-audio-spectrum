"""GPU-based spectrogram computation using CuPy."""

import time
import numpy as np
import cupy as cp


def compute_spectrogram_gpu(frames: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Compute spectrogram using GPU (CuPy FFT).
    
    Args:
        frames: 2D NumPy array of shape (num_frames, frame_size)
        
    Returns:
        Tuple of (spectrogram magnitude array as NumPy, elapsed time in seconds)
    """
    start_time = time.perf_counter()
    
    # Transfer data to GPU
    frames_gpu = cp.asarray(frames)
    
    # Compute FFT along the frame dimension (axis=1)
    fft_result = cp.fft.rfft(frames_gpu, axis=1)
    
    # Get magnitude
    spectrogram_gpu = cp.abs(fft_result)
    
    # Synchronize to ensure computation is complete
    cp.cuda.Stream.null.synchronize()
    
    # Transfer result back to CPU
    spectrogram = cp.asnumpy(spectrogram_gpu)
    
    elapsed = time.perf_counter() - start_time
    
    return spectrogram, elapsed
