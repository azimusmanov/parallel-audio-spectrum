"""CPU-based spectrogram computation using NumPy."""

import time
import numpy as np


def compute_spectrogram_cpu(
    frames: np.ndarray,
    *,
    window: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """
    Compute spectrogram using CPU (batched NumPy RFFT) fairly vs GPU.

    Args:
        frames: 2D array of shape (num_frames, frame_size)
        window: Optional 1D NumPy window of length frame_size to apply

    Returns:
        Tuple of (spectrogram magnitude array, elapsed time in seconds)
    """
    # Ensure float32 for fairness with GPU and reduced memory bandwidth
    if frames.dtype != np.float32:
        frames = frames.astype(np.float32, copy=False)

    # Optional windowing on CPU
    if window is not None:
        if window.dtype != np.float32:
            window = window.astype(np.float32, copy=False)
        frames = frames * window  # broadcast multiply

    start_time = time.perf_counter()

    # Batched RFFT along frame dimension (axis=1)
    fft_result = np.fft.rfft(frames, axis=1)

    # Magnitude
    spectrogram = np.abs(fft_result)

    elapsed = time.perf_counter() - start_time

    return spectrogram, elapsed
