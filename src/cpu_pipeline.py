"""CPU-based spectrogram computation using NumPy."""

import time
import numpy as np


def compute_spectrogram_cpu(frames: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Compute spectrogram using CPU (NumPy FFT).
    
    Args:
        frames: 2D array of shape (num_frames, frame_size)
        
    Returns:
        Tuple of (spectrogram magnitude array, elapsed time in seconds)
    """
    start_time = time.perf_counter()
    
    # Compute FFT along the frame dimension (axis=1)
    fft_result = np.fft.rfft(frames, axis=1)
    
    # Get magnitude
    spectrogram = np.abs(fft_result)
    
    elapsed = time.perf_counter() - start_time
    
    return spectrogram, elapsed
