"""
Example usage of the FEM Data Processing module.
Demonstrates FFT and PSD analysis on test signals.
"""

import numpy as np
import matplotlib.pyplot as plt
from fem_data_processing import FEMDataProcessor, generate_test_signal


def main():
    """
    Main function demonstrating FEM data processing with FFT and PSD.
    """
    print("=" * 60)
    print("FEM Data Processing - FFT and PSD Analysis")
    print("=" * 60)
    print()
    
    # Generate a test signal with multiple frequency components
    print("1. Generating test signal...")
    duration = 2.0  # seconds
    sampling_rate = 1000.0  # Hz
    frequencies = [10, 50, 120]  # Hz
    amplitudes = [1.0, 0.5, 0.3]
    
    time, signal = generate_test_signal(
        duration=duration,
        sampling_rate=sampling_rate,
        frequencies=frequencies,
        amplitudes=amplitudes
    )
    
    print(f"   Signal duration: {duration} seconds")
    print(f"   Sampling rate: {sampling_rate} Hz")
    print(f"   Frequency components: {frequencies} Hz")
    print(f"   Number of samples: {len(signal)}")
    print()
    
    # Initialize the FEM Data Processor
    print("2. Initializing FEM Data Processor...")
    processor = FEMDataProcessor(sampling_rate=sampling_rate)
    
    # Load the data
    print("3. Loading data...")
    processor.load_data(signal, time=time)
    print()
    
    # Compute FFT
    print("4. Computing FFT (Fast Fourier Transform)...")
    frequencies_fft, magnitudes_fft = processor.compute_fft()
    print(f"   FFT computed with {len(frequencies_fft)} frequency bins")
    print()
    
    # Get dominant frequencies from FFT
    print("5. Identifying dominant frequencies from FFT...")
    dominant_freqs, dominant_mags = processor.get_dominant_frequencies(n=5)
    print("   Top 5 dominant frequencies:")
    for i, (freq, mag) in enumerate(zip(dominant_freqs, dominant_mags)):
        print(f"      {i+1}. {freq:.2f} Hz (magnitude: {mag:.2f})")
    print()
    
    # Compute PSD using Welch's method
    print("6. Computing PSD (Power Spectral Density) using Welch's method...")
    frequencies_psd, psd = processor.compute_psd(method='welch', nperseg=256)
    print(f"   PSD computed with {len(frequencies_psd)} frequency bins")
    print()
    
    # Get dominant frequencies from PSD
    print("7. Identifying dominant frequencies from PSD...")
    dominant_freqs_psd, dominant_psd = processor.get_dominant_frequencies(n=5, use_psd=True)
    print("   Top 5 dominant frequencies:")
    for i, (freq, psd_val) in enumerate(zip(dominant_freqs_psd, dominant_psd)):
        print(f"      {i+1}. {freq:.2f} Hz (PSD: {psd_val:.2e})")
    print()
    
    # Create visualization
    print("8. Creating visualizations...")
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot time domain
    axes[0].plot(time, signal)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Time Domain Signal')
    axes[0].grid(True)
    
    # Plot FFT
    axes[1].plot(frequencies_fft, magnitudes_fft)
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Magnitude')
    axes[1].set_title('FFT - Frequency Domain')
    axes[1].grid(True)
    axes[1].set_xlim([0, 200])  # Focus on the frequency range of interest
    
    # Plot PSD
    axes[2].semilogy(frequencies_psd, psd)
    axes[2].set_xlabel('Frequency (Hz)')
    axes[2].set_ylabel('PSD')
    axes[2].set_title('Power Spectral Density (Welch Method)')
    axes[2].grid(True)
    axes[2].set_xlim([0, 200])  # Focus on the frequency range of interest
    
    plt.tight_layout()
    plt.savefig('fem_analysis_results.png', dpi=150)
    print("   Visualization saved as 'fem_analysis_results.png'")
    print()
    
    print("=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
