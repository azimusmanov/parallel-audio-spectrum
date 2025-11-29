"""Visualization utilities for spectrograms."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

try:
    import sounddevice as sd
    AUDIO_PLAYBACK_AVAILABLE = True
except ImportError:
    AUDIO_PLAYBACK_AVAILABLE = False


def animate_spectrum(spectrogram: np.ndarray, interval_ms: int = 30, 
                     audio: np.ndarray = None, sample_rate: int = None,
                     hop_size: int = 512) -> None:
    """Animate the spectrum over time using matplotlib."""
    num_frames, num_bins = spectrogram.shape
    
    # Calculate correct frame timing
    if audio is not None and sample_rate is not None:
        frame_duration_ms = (hop_size / sample_rate) * 1000.0
        # Use a faster refresh rate for smoother playback (half the frame duration)
        interval_ms = max(10, int(frame_duration_ms / 2))  
        print(f"  Frame duration: {frame_duration_ms:.2f}ms, Animation refresh: {interval_ms}ms")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(num_bins)
    bars = ax.bar(x, spectrogram[0], width=1.0)
    
    ax.set_xlabel('Frequency Bin')
    ax.set_ylabel('Magnitude')
    ax.set_title('Audio Spectrum Animation')
    ax.set_ylim(0, np.max(spectrogram) * 1.1)
    
    # Store animation start time for synchronization
    animation_start_time = None
    audio_started = False
    
    def update(frame_idx):
        nonlocal animation_start_time, audio_started
        
        # Start audio playback on first frame
        if not audio_started and audio is not None and sample_rate is not None and AUDIO_PLAYBACK_AVAILABLE:
            try:
                audio_play = np.ascontiguousarray(audio, dtype=np.float32)
                sd.play(audio_play, samplerate=int(sample_rate), blocking=False)
                animation_start_time = time.time()
                audio_started = True
                print("  Playing audio...")
            except Exception as e:
                print(f"  Warning: Could not play audio: {e}")
        
        # Calculate which frame we should be showing based on elapsed time
        if animation_start_time is not None and audio is not None and sample_rate is not None:
            elapsed_ms = (time.time() - animation_start_time) * 1000
            target_frame = int(elapsed_ms / frame_duration_ms)
            # Clamp to valid frame range
            actual_frame = min(max(target_frame, 0), num_frames - 1)
        else:
            actual_frame = frame_idx
        
        # Update visualization with time-synchronized frame
        for bar, height in zip(bars, spectrogram[actual_frame]):
            bar.set_height(height)
        
        # Show timing info for debugging
        if animation_start_time is not None:
            elapsed_ms = (time.time() - animation_start_time) * 1000
            expected_frame_ms = actual_frame * frame_duration_ms
            ax.set_title(f'Audio Spectrum (Frame {actual_frame + 1}/{num_frames}) - '
                        f'Elapsed: {elapsed_ms:.0f}ms, Frame time: {expected_frame_ms:.0f}ms')
        else:
            ax.set_title(f'Audio Spectrum Animation (Frame {frame_idx + 1}/{num_frames})')
        
        return bars
    
    anim = FuncAnimation(fig, update, frames=num_frames, interval=interval_ms, blit=False, repeat=False)
    
    plt.tight_layout()
    plt.show()
    
    if AUDIO_PLAYBACK_AVAILABLE:
        sd.stop()
