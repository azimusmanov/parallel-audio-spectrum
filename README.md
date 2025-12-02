# parallel-audio-spectrum

Final project for CS-358: Parallel Computing

## Overview

This project compares CPU vs GPU performance for audio spectrum computation. It processes audio files by computing spectrograms using both NumPy (CPU) and CuPy (GPU), measures the execution time for each approach, and visualizes the results with an animated spectrum display. The pipelines now perform fair, batched FFTs with float32 precision and optional windowing, and the GPU path includes a warm‑up to avoid first‑call overhead.

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
│   ├── gpu_pipeline.py     # GPU-based spectrogram computation (batched, warm-up)
│   ├── visualize.py        # Spectrum animation
│   └── benchmark.py        # Main benchmark entry point
├── data/
│   └── riser.mp3           # Example audio file (user-provided)
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.8–3.12 (CuPy does not yet support 3.14 at time of writing)
- For GPU acceleration: NVIDIA CUDA‑capable GPU + matching CUDA runtime
- Windows/Linux supported for GPU; macOS runs CPU only

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

### Windows/Linux with NVIDIA GPU

```bash
# Clone the repository
git clone https://github.com/azimusmanov/parallel-audio-spectrum.git
cd parallel-audio-spectrum

# Create virtual environment (PowerShell shown); use your Python
python -m venv .venv
. .\.venv\Scripts\Activate.ps1

# Check CUDA version (Driver reports CUDA runtime)
nvidia-smi    # Look for: CUDA Version: 12.x or 11.x

# Install dependencies (choose the matching CuPy build)
# Edit requirements-gpu.txt to select the right cupy package OR run one of:
pip install -r requirements-gpu.txt
# If mismatch errors occur, explicitly install the correct wheel, e.g.:
pip install "cupy-cuda12x"  # for CUDA 12.x
# pip install "cupy-cuda11x"  # for CUDA 11.x
```

Note: The prefilled `requirements-gpu.txt` may show `cupy-cuda13x`; on most systems today you likely need `cupy-cuda12x` (CUDA 12) or `cupy-cuda11x` (CUDA 11).

Quick CUDA/CuPy sanity check:
```powershell
cd src
python .\cuda_test.py   # should print a CuPy array and your GPU name
```

## Usage

1. Place an audio file in the `data/` directory (e.g., `riser.mp3`).

2. Run the benchmark from the `src/` directory (PowerShell shown):
```powershell
cd src
python .\benchmark.py ..\data\riser.mp3
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
- Float32 data; optional Hann window (broadcast)
- Batched `numpy.fft.rfft(frames, axis=1)` and `numpy.abs` in a single timed block

### GPU Pipeline
- Float32 data; optional Hann window applied on device
- Single batch transfer to GPU; batched `cupy.fft.rfft(frames_gpu, axis=1)`
- Warm‑up call before timing to build plans/JIT; explicit synchronize for accurate timing
- Copies results back to CPU only if needed for visualization

### Visualization
- Matplotlib FuncAnimation creates real-time bar chart
- Each frame shows frequency bin magnitudes
- Animation loops through all frames

## Performance & Fairness

- Float32 everywhere: matches precision between CPU and GPU and improves throughput.
- Batched FFTs: both CPU and GPU compute across all frames in one call.
- GPU warm‑up: prevents first‑call costs from inflating measured GPU time.
- Minimize transfers: move frames to GPU once; keep windowing and magnitude on device.
- Larger workloads yield better GPU speedups. Try:
  - `frame_size = 4096`, `hop_size = 2048`
  - Full‑length audio (minutes), not tiny clips
  - Avoid per‑frame loops; use `axis=1` batched calls

## Troubleshooting

- Missing `nvrtc64_120_0.dll` (or similar): Install the CuPy wheel that matches your CUDA runtime.
  - CUDA 12.x → `pip install cupy-cuda12x`
  - CUDA 11.x → `pip install cupy-cuda11x`
  - Then re-run: `python .\src\cuda_test.py`

- No GPU speedup: Ensure batching is active and workload is large enough. Increase `frame_size` and use full tracks. Integrated/low-end GPUs may not beat a fast CPU for tiny FFTs.

- Audio file not found: Provide a valid path, e.g. `python .\benchmark.py ..\data\riser.mp3`.

### Optional: Apply a Hann window in both pipelines
Both CPU and GPU functions support an optional `window` argument for fairness. Example snippet if you want to enable it inside `benchmark.py`:
```python
import numpy as np
window = np.hanning(frame_size).astype(np.float32)
spec_cpu, t_cpu = compute_spectrogram_cpu(frames, window=window)
spec_gpu, t_gpu = compute_spectrogram_gpu(frames, window=window)
```

## License

MIT

## Author

Azim Usmanov - CS-358 Final Project
