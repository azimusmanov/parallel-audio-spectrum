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
    """Animate the spectrum over time using matplotlib with logarithmic frequency scale."""
    num_frames, num_bins = spectrogram.shape
    
    # Calculate correct frame timing
    if audio is not None and sample_rate is not None:
        frame_duration_ms = (hop_size / sample_rate) * 1000.0
        interval_ms = max(10, int(frame_duration_ms / 2))  
        print(f"  Frame duration: {frame_duration_ms:.2f}ms, Animation refresh: {interval_ms}ms")
    
    # Apply logarithmic scaling to balance magnitudes
    threshold = np.max(spectrogram) * 1e-6
    spectrogram_masked = np.where(spectrogram < threshold, threshold, spectrogram)
    log_spectrogram = 20 * np.log10(spectrogram_masked)
    noise_floor = 20 * np.log10(threshold)
    log_spectrogram = log_spectrogram - noise_floor
    log_spectrogram = np.maximum(log_spectrogram, 0)
    
    # Create logarithmic x-axis positions
    # Start from 1 to avoid log(0), map bins to log scale
    x_linear = np.arange(1, num_bins + 1)
    x_log = np.log10(x_linear)
    
    # Create a colorful gradient colormap
    colors = ['#000080', '#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FF8000', '#FF0000', '#FF00FF']
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('spectrum', colors, N=n_bins)
    
    # Set up the plot with dark background
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Calculate bar widths for log scale - each bar spans to the next position
    bar_widths = np.diff(x_log, append=x_log[-1] + (x_log[-1] - x_log[-2]))
    
    # Create mirrored bars using log-scale x positions
    bars_top = ax.bar(x_log, log_spectrogram[0], width=bar_widths, color='cyan', alpha=0.8, bottom=0, align='edge')
    bars_bottom = ax.bar(x_log, log_spectrogram[0], width=bar_widths, color='cyan', alpha=0.8, bottom=-log_spectrogram[0], align='edge')
    
    # Calculate frequency labels
    if sample_rate is not None:
        freqs = np.linspace(0, sample_rate/2, num_bins)
        # Create log-spaced frequency ticks for better readability
        freq_values = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
        freq_ticks = []
        freq_labels = []
        
        for freq in freq_values:
            if freq <= sample_rate/2:
                # Find closest bin to this frequency
                bin_idx = int(freq / (sample_rate/2) * num_bins)
                if bin_idx < num_bins:
                    freq_ticks.append(x_log[bin_idx])
                    if freq >= 1000:
                        freq_labels.append(f'{freq/1000:.0f}k')
                    else:
                        freq_labels.append(f'{freq:.0f}')
        
        ax.set_xticks(freq_ticks)
        ax.set_xticklabels(freq_labels)
        ax.set_xlabel('Frequency (Hz) - Log Scale', fontsize=12, color='white')
    else:
        ax.set_xlabel('Frequency Bin (Log Scale)', fontsize=12, color='white')
    
    ax.set_ylabel('Magnitude (dB above noise floor)', fontsize=12, color='white') 
    ax.set_title('Audio Spectrum Animation', fontsize=14, color='white')
    
    # Set x-limits to the log scale range
    ax.set_xlim(x_log[0], x_log[-1])
    
    # Set y-limits based on log-scaled data - symmetric around 0
    data_max = np.max(log_spectrogram) + 0.5
    ax.set_ylim(-data_max, data_max)
    y_min = -data_max
    y_max = data_max
    
    # Custom Y-axis labels
    tick_positions = np.linspace(-data_max, data_max, 9)
    tick_labels = [f'{abs(pos):.0f}' for pos in tick_positions]
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)
    
    # Add grid and styling
    ax.grid(True, alpha=0.3, color='gray')
    ax.set_facecolor('#0a0a0a')
    ax.axhline(y=0, color='white', linewidth=1, alpha=0.7)
    
    # Store animation timing
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
        
        # Calculate time-synchronized frame
        if animation_start_time is not None and audio is not None and sample_rate is not None:
            elapsed_ms = (time.time() - animation_start_time) * 1000
            target_frame = int(elapsed_ms / frame_duration_ms)
            actual_frame = min(max(target_frame, 0), num_frames - 1)
        else:
            actual_frame = frame_idx
        
        # Update visualization with time-synchronized frame
        frame_data = log_spectrogram[actual_frame]
        
        # Normalize for color mapping
        norm_data = (frame_data - y_min) / (y_max - y_min)
        colors_for_frame = cmap(norm_data)
        
        # Update bar heights and colors
        for bar_top, bar_bottom, height, color in zip(bars_top, bars_bottom, frame_data, colors_for_frame):
            bar_top.set_height(height)
            bar_top.set_color(color)
            
            bar_bottom.set_height(height)
            bar_bottom.xy = (bar_bottom.xy[0], -height)
            bar_bottom.set_color(color)
        
        # Update title with timing info
        if animation_start_time is not None:
            elapsed_ms = (time.time() - animation_start_time) * 1000
            expected_frame_ms = actual_frame * frame_duration_ms
            ax.set_title(f'Audio Spectrum (Frame {actual_frame + 1}/{num_frames}) - '
                        f'Elapsed: {elapsed_ms:.0f}ms, Frame time: {expected_frame_ms:.0f}ms', 
                        fontsize=14, color='white')
        else:
            ax.set_title(f'Audio Spectrum Animation (Frame {frame_idx + 1}/{num_frames})', 
                        fontsize=14, color='white')
        
        return bars_top + bars_bottom
    
    anim = FuncAnimation(fig, update, frames=num_frames, interval=interval_ms, blit=False, repeat=False)
    
    # Add colorbar
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
    
    plt.style.use('default')