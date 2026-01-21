# FEM Data Processing - FFT and PSD Analysis

A Python-based tool for processing Finite Element Method (FEM) data using Fast Fourier Transform (FFT) and Power Spectral Density (PSD) analysis.

## Features

- **FFT Analysis**: Compute Fast Fourier Transform to analyze frequency components in time-domain signals
- **PSD Computation**: Calculate Power Spectral Density using Welch's method or periodogram
- **Visualization**: Generate plots for time-domain signals, FFT results, and PSD
- **Dominant Frequency Detection**: Automatically identify the most significant frequency components
- **Flexible Data Loading**: Support for various data formats and sampling rates

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Oilsmell/Data-processing-FEM-1.git
cd Data-processing-FEM-1
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Example

```python
from fem_data_processing import FEMDataProcessor, generate_test_signal

# Generate a test signal
time, signal = generate_test_signal(
    duration=2.0,
    sampling_rate=1000.0,
    frequencies=[10, 50, 120],
    amplitudes=[1.0, 0.5, 0.3]
)

# Initialize the processor
processor = FEMDataProcessor(sampling_rate=1000.0)

# Load data
processor.load_data(signal, time=time)

# Compute FFT
frequencies, magnitudes = processor.compute_fft()

# Compute PSD
frequencies_psd, psd = processor.compute_psd(method='welch')

# Get dominant frequencies
dominant_freqs, dominant_mags = processor.get_dominant_frequencies(n=5)

# Plot results
processor.plot_time_domain()
processor.plot_fft()
processor.plot_psd()
```

### Running the Example Script

To see a complete demonstration of the FEM data processing capabilities:

```bash
python example_usage.py
```

This will:
1. Generate a test signal with multiple frequency components
2. Perform FFT analysis
3. Compute PSD using Welch's method
4. Identify dominant frequencies
5. Generate and save visualizations

## API Reference

### FEMDataProcessor

Main class for processing FEM data.

#### Methods

- `__init__(sampling_rate=1000.0)`: Initialize the processor with a sampling rate
- `load_data(data, time=None)`: Load time-domain data for processing
- `compute_fft()`: Compute the Fast Fourier Transform
- `compute_psd(method='welch', nperseg=None)`: Compute Power Spectral Density
- `plot_time_domain(title, save_path)`: Plot time-domain signal
- `plot_fft(title, save_path)`: Plot FFT results
- `plot_psd(title, save_path, scale)`: Plot PSD results
- `get_dominant_frequencies(n=5)`: Get the n most dominant frequencies

### Utility Functions

- `generate_test_signal(duration, sampling_rate, frequencies, amplitudes)`: Generate a multi-frequency test signal

## Dependencies

- NumPy >= 1.21.0
- SciPy >= 1.7.0
- Matplotlib >= 3.4.0

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.