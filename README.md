# parallel-audio-spectrum

Final project for CS-358: Parallel Computing

## Overview

This project compares CPU vs GPU performance for audio spectrum computation. It processes audio files by computing spectrograms using both NumPy (CPU) and CuPy (GPU), measures the execution time for each approach, and visualizes the results with an animated spectrum display.

## Features

- **CPU Pipeline**: NumPy FFT-based spectrogram computation
- **GPU Pipeline**: CuPy CUDA-accelerated FFT computation
- **Performance Benchmarking**: Direct CPU vs GPU runtime comparison with speedup metrics
- **Visualization**: Real-time animated spectrum display using matplotlib

## Project Structure

```
parallel-audio-spectrum/
├── src/
│   ├── audio_utils.py      # Audio loading and framing utilities
│   ├── cpu_pipeline.py     # CPU-based spectrogram computation
│   ├── gpu_pipeline.py     # GPU-based spectrogram computation
│   ├── visualize.py        # Spectrum animation
│   └── benchmark.py        # Main benchmark entry point
├── data/
│   └── example.wav         # Sample audio file (user-provided)
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.8-3.12 (Python 3.14 not yet supported by CuPy)
- **For GPU acceleration**: NVIDIA CUDA-capable GPU + CUDA Toolkit
- **macOS users**: Can test CPU pipeline only (no CUDA support on Apple Silicon/Intel Macs)

## Installation

### macOS (CPU-only development)

```bash
# Clone the repository
git clone https://github.com/azimusmanov/parallel-audio-spectrum.git
cd parallel-audio-spectrum

# Create virtual environment with Python 3.10-3.12
python3.10 -m venv venv
source venv/bin/activate

# Install CPU-only dependencies
pip install -r requirements.txt
```

The code will gracefully handle missing GPU and run CPU-only with visualization.

### Linux/Windows with NVIDIA GPU

```bash
# Clone the repository
git clone https://github.com/azimusmanov/parallel-audio-spectrum.git
cd parallel-audio-spectrum

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Check your CUDA version
nvidia-smi  # Look for "CUDA Version: X.X"

# Install dependencies with GPU support
pip install -r requirements-gpu.txt
```

**Note**: Adjust the CuPy package in `requirements-gpu.txt` based on your CUDA version:
- CUDA 11.x: `cupy-cuda11x`
- CUDA 12.x: `cupy-cuda12x`

## Usage

1. Place a mono WAV audio file in the `data/` directory and name it `example.wav`

2. Run the benchmark from the `src/` directory:
```bash
cd src
python benchmark.py
```

3. The program will:
   - Load and frame the audio file
   - Compute spectrogram on CPU using NumPy
   - Compute spectrogram on GPU using CuPy
   - Display timing results and speedup factor
   - Show an animated visualization of the spectrum

## Example Output

```
============================================================
Audio Spectrum Benchmark: CPU vs GPU
============================================================

Loading audio from: ../data/example.wav
  Sample rate: 22050 Hz
  Duration: 10.00 seconds
  Samples: 220500

Framing audio...
  Frame size: 1024
  Hop size: 512
  Number of frames: 429

Running CPU spectrogram computation...
  CPU time: 0.012345 seconds
  Spectrogram shape: (429, 513)

Running GPU spectrogram computation...
  GPU time: 0.002345 seconds
  Spectrogram shape: (429, 513)

============================================================
RESULTS:
  CPU Time:  XXX seconds
  GPU Time:  XXX seconds
  Speedup:   XXXx
============================================================
```

## Implementation Details

### Audio Processing
- Audio is loaded as mono using librosa
- Frames are created with 50% overlap (frame_size=1024, hop_size=512)
- Each frame undergoes FFT to extract frequency components

### CPU Pipeline
- Uses `numpy.fft.rfft()` for real FFT computation
- Computes magnitude spectrum with `numpy.abs()`

### GPU Pipeline
- Transfers data to GPU using `cupy.asarray()`
- Uses `cupy.fft.rfft()` for GPU-accelerated FFT
- Synchronizes GPU computation before timing
- Transfers results back to CPU for visualization

### Visualization
- Matplotlib FuncAnimation creates real-time bar chart
- Each frame shows frequency bin magnitudes
- Animation loops through all frames

## Performance Considerations

- **Data Transfer Overhead**: GPU timing includes CPU↔GPU transfer (realistic for end-to-end performance)
- **Batch Size**: Larger audio files with more frames benefit more from GPU parallelization
- **Frame Size**: Larger FFT sizes increase computational complexity, potentially increasing speedup

## Troubleshooting

**CuPy Import Error**: Ensure CUDA is installed and the correct `cupy-cudaXXx` package matches your CUDA version.

**No GPU Speedup**: For small audio files, CPU overhead and data transfer may outweigh GPU benefits. Try longer audio files.

**Audio File Not Found**: Ensure `example.wav` exists in the `data/` directory.

## License

MIT

## Author

Azim Usmanov - CS-358 Final Project
