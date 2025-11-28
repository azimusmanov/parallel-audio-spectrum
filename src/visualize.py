"""Visualization utilities for spectrograms."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Try to import sounddevice for audio playback
try:
    import sounddevice as sd
    AUDIO_PLAYBACK_AVAILABLE = True
except ImportError:
    AUDIO_PLAYBACK_AVAILABLE = False


def animate_spectrum(spectrogram: np.ndarray, interval_ms: int = 30, 
                     audio: np.ndarray = None, sample_rate: int = None,
                     hop_size: int = 512) -> None:
    """
    Animate the spectrum over time using matplotlib.
    
    Args:
        spectrogram: 2D array of shape (num_frames, num_freq_bins)
        interval_ms: Interval between frames in milliseconds (ignored if audio provided)
        audio: Optional audio samples to play synchronized with animation
        sample_rate: Sample rate of audio (required if audio is provided)
        hop_size: Hop size used for framing (required for sync with audio)
    """
    num_frames, num_bins = spectrogram.shape
    
    # Calculate proper interval to sync with audio
    if audio is not None and sample_rate is not None:
        # Time between frames = hop_size / sample_rate
        frame_duration_ms = (hop_size / sample_rate) * 1000
        interval_ms = int(frame_duration_ms)
        print(f"  Synchronized interval: {interval_ms}ms per frame (hop={hop_size}, sr={sample_rate}Hz)")
    
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
    
    # Start audio playback if available
    audio_started = False
    if audio is not None and sample_rate is not None and AUDIO_PLAYBACK_AVAILABLE:
        try:
            # Convert to the right format for sounddevice
            audio_play = np.ascontiguousarray(audio, dtype=np.float32)
            
            # Normalize if needed
            max_val = np.max(np.abs(audio_play))
            if max_val > 1.0:
                audio_play = audio_play / max_val * 0.9
            
            # Play with explicit parameters - DON'T use blocking=False, keep stream open
            sd.play(audio_play, samplerate=int(sample_rate))
            audio_started = True
            print("  Playing audio synchronized with animation...")
        except Exception as e:
            print(f"  Warning: Could not play audio: {e}")
            audio_started = False
    elif audio is not None and not AUDIO_PLAYBACK_AVAILABLE:
        print("  Note: Install sounddevice for synchronized audio playback")
    
    # Animation update function
    def update(frame_idx):
        """Update bar heights for the current frame."""
        for bar, height in zip(bars, spectrogram[frame_idx]):
            bar.set_height(height)
        ax.set_title(f'Audio Spectrum Animation (Frame {frame_idx + 1}/{num_frames})')
        return bars
    
    # Create animation - play only once (repeat=False)
    anim = FuncAnimation(
        fig, 
        update, 
        frames=num_frames,
        interval=interval_ms,
        blit=False,
        repeat=False  # Play only once
    )
    
    plt.tight_layout()
    plt.show()
    
    # Stop audio when window closes
    if audio_started:
        sd.stop()
