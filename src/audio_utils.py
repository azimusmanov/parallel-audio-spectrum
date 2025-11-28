"""Audio utilities for loading and framing audio data."""

import numpy as np
import librosa


def load_audio(path: str) -> tuple[np.ndarray, int]:
    """
    Load a mono WAV file.
    
    Args:
        path: Path to the audio file
        
    Returns:
        Tuple of (audio samples, sample rate)
    """
    # Load with original sample rate (sr=None) to avoid resampling issues
    y, sr = librosa.load(path, sr=None, mono=True)
    return y, sr


def frame_audio(y: np.ndarray, frame_size: int = 1024, hop_size: int = 512) -> np.ndarray:
    """
    Split audio into overlapping frames.
    
    Args:
        y: Audio samples as 1D array
        frame_size: Number of samples per frame
        hop_size: Number of samples to advance between frames
        
    Returns:
        2D array of shape (num_frames, frame_size)
    """
    num_frames = 1 + (len(y) - frame_size) // hop_size
    frames = np.zeros((num_frames, frame_size), dtype=y.dtype)
    
    for i in range(num_frames):
        start = i * hop_size
        frames[i] = y[start:start + frame_size]
    
    return frames
