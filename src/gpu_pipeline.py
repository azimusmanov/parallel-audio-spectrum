"""GPU-based spectrogram computation using CuPy."""

import time
import numpy as np
import cupy as cp


def compute_spectrogram_gpu(
        frames: np.ndarray,
        *,
        window: np.ndarray | None = None,
        return_numpy: bool = True,
    ) -> tuple[np.ndarray, float]:
    """
    Compute spectrogram using GPU (batched CuPy FFT) with minimal transfers.

    Args:
        frames: 2D NumPy array of shape (num_frames, frame_size)
        window: Optional 1D NumPy window of length frame_size to apply
        return_numpy: If True, copy back to CPU; otherwise keep on GPU

    Returns:
        Tuple of (spectrogram magnitude, elapsed seconds)
        - If return_numpy=True: spectrogram is a NumPy array
        - If return_numpy=False: spectrogram is a CuPy array
    """
    # Ensure float32 to reduce transfer size and speed up FFT
    if frames.dtype != np.float32:
        frames = frames.astype(np.float32, copy=False)

    # Transfer batch once
    frames_gpu = cp.asarray(frames)

    # Optional windowing on-GPU (avoids CPU-side multiply cost)
    if window is not None:
        if window.dtype != np.float32:
            window = window.astype(np.float32, copy=False)
        window_gpu = cp.asarray(window)
        # Broadcast multiply: (num_frames, frame_size) * (frame_size,)
        frames_gpu *= window_gpu

    # Warm-up to build FFT plan and JIT caches (not timed)
    _ = cp.fft.rfft(frames_gpu[: min(8, frames_gpu.shape[0])], axis=1)
    cp.cuda.Stream.null.synchronize()

    # Timed compute: batched RFFT then magnitude
    start_time = time.perf_counter()
    fft_result = cp.fft.rfft(frames_gpu, axis=1)
    spectrogram_gpu = cp.abs(fft_result)
    cp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - start_time

    if return_numpy:
        return cp.asnumpy(spectrogram_gpu), elapsed
    else:
        return spectrogram_gpu, elapsed
