"""
Tests for FEM Data Processing module.
Validates FFT and PSD functionality.
"""

import numpy as np
from fem_data_processing import FEMDataProcessor, generate_test_signal


def test_generate_signal():
    """Test signal generation."""
    print("Testing signal generation...")
    time, signal = generate_test_signal(duration=1.0, sampling_rate=100.0)
    assert len(time) == 100, f"Expected 100 samples, got {len(time)}"
    assert len(signal) == 100, f"Expected 100 samples, got {len(signal)}"
    print("✓ Signal generation test passed")


def test_data_loading():
    """Test data loading."""
    print("Testing data loading...")
    processor = FEMDataProcessor(sampling_rate=100.0)
    test_data = np.array([1, 2, 3, 4, 5])
    processor.load_data(test_data)
    assert processor.time_data is not None, "Data should be loaded"
    assert len(processor.time_data) == 5, "Data length should be 5"
    print("✓ Data loading test passed")


def test_fft_computation():
    """Test FFT computation."""
    print("Testing FFT computation...")
    time, signal = generate_test_signal(
        duration=1.0, 
        sampling_rate=1000.0,
        frequencies=[10, 50],
        amplitudes=[1.0, 0.5]
    )
    
    processor = FEMDataProcessor(sampling_rate=1000.0)
    processor.load_data(signal, time=time)
    freqs, mags = processor.compute_fft()
    
    assert freqs is not None, "Frequencies should be computed"
    assert mags is not None, "Magnitudes should be computed"
    assert len(freqs) == len(mags), "Frequencies and magnitudes should have same length"
    assert all(freqs >= 0), "All frequencies should be positive"
    print(f"✓ FFT computation test passed ({len(freqs)} frequency bins)")


def test_psd_computation():
    """Test PSD computation."""
    print("Testing PSD computation (Welch method)...")
    time, signal = generate_test_signal(duration=2.0, sampling_rate=1000.0)
    
    processor = FEMDataProcessor(sampling_rate=1000.0)
    processor.load_data(signal, time=time)
    freqs, psd = processor.compute_psd(method='welch')
    
    assert freqs is not None, "Frequencies should be computed"
    assert psd is not None, "PSD should be computed"
    assert len(freqs) == len(psd), "Frequencies and PSD should have same length"
    assert all(psd >= 0), "All PSD values should be non-negative"
    print(f"✓ PSD computation test passed ({len(freqs)} frequency bins)")


def test_psd_periodogram():
    """Test PSD computation with periodogram method."""
    print("Testing PSD computation (Periodogram method)...")
    time, signal = generate_test_signal(duration=1.0, sampling_rate=500.0)
    
    processor = FEMDataProcessor(sampling_rate=500.0)
    processor.load_data(signal, time=time)
    freqs, psd = processor.compute_psd(method='periodogram')
    
    assert freqs is not None, "Frequencies should be computed"
    assert psd is not None, "PSD should be computed"
    print(f"✓ Periodogram computation test passed ({len(freqs)} frequency bins)")


def test_dominant_frequencies():
    """Test dominant frequency detection."""
    print("Testing dominant frequency detection...")
    time, signal = generate_test_signal(
        duration=2.0,
        sampling_rate=1000.0,
        frequencies=[10, 50, 120],
        amplitudes=[1.0, 0.5, 0.3]
    )
    
    processor = FEMDataProcessor(sampling_rate=1000.0)
    processor.load_data(signal, time=time)
    processor.compute_fft()
    
    dom_freqs, dom_mags = processor.get_dominant_frequencies(n=3)
    
    assert len(dom_freqs) == 3, "Should return 3 dominant frequencies"
    assert len(dom_mags) == 3, "Should return 3 magnitudes"
    
    # The dominant frequencies should be close to 10, 50, and 120 Hz
    # (within 1 Hz due to frequency resolution)
    expected = [10, 50, 120]
    for i, exp_freq in enumerate(expected):
        found = False
        for dom_freq in dom_freqs:
            if abs(dom_freq - exp_freq) < 1.0:
                found = True
                break
        assert found, f"Expected frequency {exp_freq} Hz not found in dominant frequencies"
    
    print(f"✓ Dominant frequency detection test passed")
    print(f"  Detected: {dom_freqs}")


def test_psd_dominant_frequencies():
    """Test dominant frequency detection from PSD."""
    print("Testing dominant frequency detection from PSD...")
    time, signal = generate_test_signal(
        duration=2.0,
        sampling_rate=1000.0,
        frequencies=[15, 60],
        amplitudes=[1.0, 0.8]
    )
    
    processor = FEMDataProcessor(sampling_rate=1000.0)
    processor.load_data(signal, time=time)
    processor.compute_psd(method='welch')
    
    dom_freqs, dom_psd = processor.get_dominant_frequencies(n=2, use_psd=True)
    
    assert len(dom_freqs) == 2, "Should return 2 dominant frequencies"
    assert len(dom_psd) == 2, "Should return 2 PSD values"
    
    print(f"✓ PSD dominant frequency detection test passed")
    print(f"  Detected: {dom_freqs}")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running FEM Data Processing Tests")
    print("=" * 60)
    print()
    
    tests = [
        test_generate_signal,
        test_data_loading,
        test_fft_computation,
        test_psd_computation,
        test_psd_periodogram,
        test_dominant_frequencies,
        test_psd_dominant_frequencies,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed += 1
        print()
    
    print("=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
