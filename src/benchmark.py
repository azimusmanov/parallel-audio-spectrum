"""Benchmark CPU vs GPU spectrogram computation."""

import os
from audio_utils import load_audio, frame_audio
from cpu_pipeline import compute_spectrogram_cpu
from visualize import animate_spectrum

# Try to import GPU pipeline, but make it optional
try:
    from gpu_pipeline import compute_spectrogram_gpu
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("Note: CuPy not installed - GPU pipeline unavailable")


def main(audio_path):
    """Main benchmark entry point."""
    # Path to audio file (relative to src/)
    # audio_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'example.wav')
    
    # Check if file exists
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at {audio_path}")
        print("Please add an example.wav file to the data/ directory.")
        return
    
    print("=" * 60)
    print("Audio Spectrum Benchmark: CPU vs GPU")
    print("=" * 60)
    
    # Load audio
    print(f"\nLoading audio from: {audio_path}")
    y, sr = load_audio(audio_path)
    print(f"  Sample rate: {sr} Hz")
    print(f"  Duration: {len(y) / sr:.2f} seconds")
    print(f"  Samples: {len(y)}")
    
    # Frame audio
    frame_size = 1024
    hop_size = 512
    print(f"\nFraming audio...")
    print(f"  Frame size: {frame_size}")
    print(f"  Hop size: {hop_size}")
    frames = frame_audio(y, frame_size=frame_size, hop_size=hop_size)
    print(f"  Number of frames: {frames.shape[0]}")
    
    # CPU computation
    print(f"\nRunning CPU spectrogram computation...")
    spec_cpu, time_cpu = compute_spectrogram_cpu(frames)
    print(f"  CPU time: {time_cpu:.6f} seconds")
    print(f"  Spectrogram shape: {spec_cpu.shape}")
    
    # GPU computation
    print(f"\nRunning GPU spectrogram computation...")
    if not GPU_AVAILABLE:
        print(f"  GPU pipeline not available (CuPy not installed)")
        print(f"  Skipping GPU benchmark - run on Linux/Windows with NVIDIA GPU")
        print(f"\n  Animating CPU result...")
        animate_spectrum(spec_cpu, interval_ms=30)
        return
    
    try:
        spec_gpu, time_gpu = compute_spectrogram_gpu(frames)
        print(f"  GPU time: {time_gpu:.6f} seconds")
        print(f"  Spectrogram shape: {spec_gpu.shape}")
        
        # Compute speedup
        speedup = time_cpu / time_gpu
        print(f"\n{'=' * 60}")
        print(f"RESULTS:")
        print(f"  CPU Time:  {time_cpu:.6f} seconds")
        print(f"  GPU Time:  {time_gpu:.6f} seconds")
        print(f"  Speedup:   {speedup:.2f}x")
        print(f"{'=' * 60}")
        
        # Animate GPU result
        print(f"\nLaunching spectrum animation...")
        animate_spectrum(spec_gpu, interval_ms=30)
        
    except Exception as e:
        print(f"  GPU computation failed: {e}")
        print(f"  This may be due to missing CUDA or CuPy installation.")
        print(f"\n  Animating CPU result instead...")
        animate_spectrum(spec_cpu, interval_ms=30)


if __name__ == '__main__':
    import sys
    
    # Default to riser.mp3 in data folder
    if len(sys.argv) > 1:
        # Use custom path from command line argument
        audio_path = sys.argv[1]
    else:
        # Use default path relative to this script
        audio_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'riser.mp3')
    
    main(audio_path)
