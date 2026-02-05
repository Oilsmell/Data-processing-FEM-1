"""
FEM Data Processing Module
Provides functions for processing Finite Element Method data using FFT and PSD analysis.
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt


class FEMDataProcessor:
    """
    A class for processing FEM (Finite Element Method) data using FFT and PSD analysis.
    """
    
    def __init__(self, sampling_rate=1000.0):
        """
        Initialize the FEM Data Processor.
        
        Parameters:
        -----------
        sampling_rate : float
            Sampling rate of the data in Hz (default: 1000.0)
        """
        self.sampling_rate = sampling_rate
        self.time_data = None
        self.fft_frequencies = None
        self.fft_result = None
        self.psd_frequencies = None
        self.psd_result = None
        
    def load_data(self, data, time=None):
        """
        Load time-domain data for processing.
        
        Parameters:
        -----------
        data : array-like
            Time-domain signal data
        time : array-like, optional
            Time values corresponding to the data points
        """
        self.time_data = np.array(data)
        
        if time is not None:
            self.time = np.array(time)
        else:
            # Generate time array based on sampling rate
            self.time = np.arange(len(self.time_data)) / self.sampling_rate
            
    def compute_fft(self):
        """
        Compute the Fast Fourier Transform (FFT) of the loaded data.
        
        Returns:
        --------
        frequencies : ndarray
            Frequency values
        magnitudes : ndarray
            Magnitude of the FFT
        """
        if self.time_data is None:
            raise ValueError("No data loaded. Use load_data() first.")
            
        n = len(self.time_data)
        
        # Compute FFT
        fft_values = fft(self.time_data)
        frequencies = fftfreq(n, 1/self.sampling_rate)
        
        # Only take the positive frequencies
        positive_freq_idx = frequencies >= 0
        frequencies = frequencies[positive_freq_idx]
        fft_values = fft_values[positive_freq_idx]
        
        # Compute magnitude
        magnitudes = np.abs(fft_values)
        
        # Store results
        self.fft_frequencies = frequencies
        self.fft_result = magnitudes
        
        return frequencies, magnitudes
    
    def compute_psd(self, method='welch', nperseg=None):
        """
        Compute the Power Spectral Density (PSD) of the loaded data.
        
        Parameters:
        -----------
        method : str
            Method to use for PSD computation ('welch' or 'periodogram')
        nperseg : int, optional
            Length of each segment for Welch's method
            
        Returns:
        --------
        frequencies : ndarray
            Frequency values
        psd : ndarray
            Power spectral density values
        """
        if self.time_data is None:
            raise ValueError("No data loaded. Use load_data() first.")
        
        if method == 'welch':
            if nperseg is None:
                nperseg = min(256, len(self.time_data))
            frequencies, psd = signal.welch(self.time_data, 
                                           fs=self.sampling_rate, 
                                           nperseg=nperseg)
        elif method == 'periodogram':
            frequencies, psd = signal.periodogram(self.time_data, 
                                                 fs=self.sampling_rate)
        else:
            raise ValueError("Method must be 'welch' or 'periodogram'")
        
        # Store results
        self.psd_frequencies = frequencies
        self.psd_result = psd
        
        return frequencies, psd
    
    def plot_time_domain(self, title='Time Domain Signal', save_path=None):
        """
        Plot the time-domain signal.
        
        Parameters:
        -----------
        title : str
            Title for the plot
        save_path : str, optional
            Path to save the figure
        """
        if self.time_data is None:
            raise ValueError("No data loaded. Use load_data() first.")
        
        plt.figure(figsize=(10, 4))
        plt.plot(self.time, self.time_data)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title(title)
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
    def plot_fft(self, title='FFT - Frequency Domain', save_path=None):
        """
        Plot the FFT results.
        
        Parameters:
        -----------
        title : str
            Title for the plot
        save_path : str, optional
            Path to save the figure
        """
        if self.fft_result is None:
            raise ValueError("FFT not computed. Use compute_fft() first.")
        
        plt.figure(figsize=(10, 4))
        plt.plot(self.fft_frequencies, self.fft_result)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.title(title)
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
    def plot_psd(self, title='Power Spectral Density', save_path=None, scale='log'):
        """
        Plot the PSD results.
        
        Parameters:
        -----------
        title : str
            Title for the plot
        save_path : str, optional
            Path to save the figure
        scale : str
            Scale for y-axis ('log' or 'linear')
        """
        if self.psd_result is None:
            raise ValueError("PSD not computed. Use compute_psd() first.")
        
        plt.figure(figsize=(10, 4))
        if scale == 'log':
            plt.semilogy(self.psd_frequencies, self.psd_result)
        else:
            plt.plot(self.psd_frequencies, self.psd_result)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('PSD')
        plt.title(title)
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
    def get_dominant_frequencies(self, n=5, use_psd=False):
        """
        Get the dominant frequencies from the FFT or PSD results.
        
        Parameters:
        -----------
        n : int
            Number of dominant frequencies to return
        use_psd : bool
            If True, use PSD results. If False, use FFT results.
            
        Returns:
        --------
        dominant_freqs : ndarray
            Array of dominant frequency values
        magnitudes : ndarray
            Array of corresponding magnitudes
        """
        if use_psd and self.psd_result is not None:
            # Use stored PSD data
            data = self.psd_result
            freqs = self.psd_frequencies
        elif not use_psd and self.fft_result is not None:
            # Use stored FFT data
            data = self.fft_result
            freqs = self.fft_frequencies
        elif self.psd_result is not None:
            # Fallback to PSD if available
            data = self.psd_result
            freqs = self.psd_frequencies
        elif self.fft_result is not None:
            # Fallback to FFT if available
            data = self.fft_result
            freqs = self.fft_frequencies
        else:
            raise ValueError("No frequency analysis results available.")
        
        # Find the indices of the n largest peaks
        peak_indices = np.argsort(data)[-n:][::-1]
        
        dominant_freqs = freqs[peak_indices]
        magnitudes = data[peak_indices]
        
        return dominant_freqs, magnitudes


def generate_test_signal(duration=1.0, sampling_rate=1000.0, 
                         frequencies=[10, 50, 120], amplitudes=[1.0, 0.5, 0.3]):
    """
    Generate a test signal with multiple frequency components.
    
    Parameters:
    -----------
    duration : float
        Duration of the signal in seconds
    sampling_rate : float
        Sampling rate in Hz
    frequencies : list
        List of frequency components to include
    amplitudes : list
        List of amplitudes for each frequency component
        
    Returns:
    --------
    time : ndarray
        Time array
    signal : ndarray
        Generated signal
    """
    n_samples = int(duration * sampling_rate)
    time = np.linspace(0, duration, n_samples)
    
    signal = np.zeros(n_samples)
    for freq, amp in zip(frequencies, amplitudes):
        signal += amp * np.sin(2 * np.pi * freq * time)
    
    # Add some noise
    signal += 0.1 * np.random.randn(n_samples)
    
    return time, signal
