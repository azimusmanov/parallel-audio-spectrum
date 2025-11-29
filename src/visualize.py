"""Visualization utilities for spectrograms."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
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
    
    # Apply logarithmic scaling to balance magnitudes
    # Handle zero/near-zero values more intelligently
    # Set a threshold - anything below this is considered "silence"
    threshold = np.max(spectrogram) * 1e-6  # Dynamic threshold based on max signal
    
    # Create a masked version where very small values are set to a reasonable floor
    spectrogram_masked = np.where(spectrogram < threshold, threshold, spectrogram)
    log_spectrogram = 20 * np.log10(spectrogram_masked)  # Use 20*log10 for dB scale
    
    # Normalize to make it more intuitive: silence ≈ 0, loud signals = positive
    # Find the noise floor and peak
    noise_floor = 20 * np.log10(threshold)  # dB value of our threshold
    
    # Shift so that noise floor becomes ~0
    log_spectrogram = log_spectrogram - noise_floor
    
    # Clamp negative values (below noise floor) to 0
    log_spectrogram = np.maximum(log_spectrogram, 0)
    
    # Create a colorful gradient colormap
    colors = ['#000080', '#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FF8000', '#FF0000', '#FF00FF']
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('spectrum', colors, N=n_bins)
    
    # Set up the plot with dark background for better contrast
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(num_bins)
    
    # Create mirrored bars for professional visualizer look
    # Top bars (positive direction from center)
    bars_top = ax.bar(x, log_spectrogram[0], width=1.0, color='cyan', alpha=0.8, bottom=0)
    # Bottom bars (negative direction from center, but showing same positive magnitude values)
    bars_bottom = ax.bar(x, log_spectrogram[0], width=1.0, color='cyan', alpha=0.8, bottom=-log_spectrogram[0])
    
    # Calculate frequency labels (assuming we have sample rate info)
    if sample_rate is not None:
        # For rfft, frequencies go from 0 to sample_rate/2
        freqs = np.linspace(0, sample_rate/2, num_bins)
        # Create frequency tick labels every ~2kHz
        freq_ticks = np.arange(0, num_bins, max(1, num_bins // 10))
        freq_labels = [f'{freqs[i]/1000:.1f}k' if freqs[i] >= 1000 else f'{freqs[i]:.0f}' for i in freq_ticks]
        ax.set_xticks(freq_ticks)
        ax.set_xticklabels(freq_labels)
        ax.set_xlabel('Frequency (Hz)', fontsize=12, color='white')
    else:
        ax.set_xlabel('Frequency Bin', fontsize=12, color='white')
    
    ax.set_ylabel('Magnitude (dB above noise floor)', fontsize=12, color='white') 
    ax.set_title('Audio Spectrum Animation', fontsize=14, color='white')
    
    # Set y-limits based on log-scaled data - make it symmetric around 0
    data_max = np.max(log_spectrogram) + 0.5
    ax.set_ylim(-data_max, data_max)  # Symmetric: from -max to +max
    y_min = -data_max
    y_max = data_max
    
    # Custom Y-axis labels for mirrored visualizer
    # Both sides should show positive magnitude values
    tick_positions = np.linspace(-data_max, data_max, 9)  # 9 tick marks
    tick_labels = [f'{abs(pos):.0f}' for pos in tick_positions]  # Show absolute values
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, color='gray')
    ax.set_facecolor('#0a0a0a')
    
    # Add a center line for professional look
    ax.axhline(y=0, color='white', linewidth=1, alpha=0.7)
    
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
        
        # Update visualization with time-synchronized frame using log-scaled data
        frame_data = log_spectrogram[actual_frame]
        
        # Normalize the frame data for color mapping
        norm_data = (frame_data - y_min) / (y_max - y_min)
        colors_for_frame = cmap(norm_data)
        
        # Update bar heights and colors for both top and bottom bars
        for bar_top, bar_bottom, height, color in zip(bars_top, bars_bottom, frame_data, colors_for_frame):
            # Top bars go upward from 0 (positive direction)
            bar_top.set_height(height)
            bar_top.set_color(color)
            
            # Bottom bars: mirror the visual but keep positive magnitude values
            # Set bottom position to -height so bars extend from -height to 0
            bar_bottom.set_height(height)  # Height is positive (same magnitude)
            bar_bottom.xy = (bar_bottom.xy[0], -height)  # Move bottom of bar to -height
            bar_bottom.set_color(color)
        
        # Show timing info for debugging
        if animation_start_time is not None:
            elapsed_ms = (time.time() - animation_start_time) * 1000
            expected_frame_ms = actual_frame * frame_duration_ms
            ax.set_title(f'Audio Spectrum (Frame {actual_frame + 1}/{num_frames}) - '
                        f'Elapsed: {elapsed_ms:.0f}ms, Frame time: {expected_frame_ms:.0f}ms', 
                        fontsize=14, color='white')
        else:
            ax.set_title(f'Audio Spectrum Animation (Frame {frame_idx + 1}/{num_frames})', 
                        fontsize=14, color='white')
        
        return bars_top + bars_bottom  # Return both sets of bars
    
    anim = FuncAnimation(fig, update, frames=num_frames, interval=interval_ms, blit=False, repeat=False)
    
    # Add a colorbar to show the magnitude scale
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=y_min, vmax=y_max))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8, aspect=20)
    cbar.set_label('Magnitude (dB)', rotation=270, labelpad=20, fontsize=12, color='white')
    cbar.ax.yaxis.label.set_color('white')
    cbar.ax.tick_params(colors='white')
    
    plt.tight_layout()
    plt.show()
    
    if AUDIO_PLAYBACK_AVAILABLE:
        sd.stop()
    
    # Reset matplotlib style
    plt.style.use('default')
