"""Visualization utilities for spectrograms."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def animate_spectrum(spectrogram: np.ndarray, interval_ms: int = 30) -> None:
    """
    Animate the spectrum over time using matplotlib.
    
    Args:
        spectrogram: 2D array of shape (num_frames, num_freq_bins)
        interval_ms: Interval between frames in milliseconds
    """
    num_frames, num_bins = spectrogram.shape
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Initialize bar plot with first frame
    x = np.arange(num_bins)
    bars = ax.bar(x, spectrogram[0], width=1.0)
    
    # Set labels and limits
    ax.set_xlabel('Frequency Bin')
    ax.set_ylabel('Magnitude')
    ax.set_title('Audio Spectrum Animation')
    ax.set_ylim(0, np.max(spectrogram) * 1.1)
    
    # Animation update function
    def update(frame_idx):
        """Update bar heights for the current frame."""
        for bar, height in zip(bars, spectrogram[frame_idx]):
            bar.set_height(height)
        ax.set_title(f'Audio Spectrum Animation (Frame {frame_idx + 1}/{num_frames})')
        return bars
    
    # Create animation
    anim = FuncAnimation(
        fig, 
        update, 
        frames=num_frames,
        interval=interval_ms,
        blit=False,
        repeat=True
    )
    
    plt.tight_layout()
    plt.show()
