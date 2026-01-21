import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch, find_peaks, detrend
from scipy.fft import rfft, rfftfreq
import os
from typing import Optional, Tuple, List

# ==========================================
# 1. CONFIGURATION - 설정
# ==========================================
class Config:
    """Centralized configuration for signal analysis."""
    
    # ★★★ 데이터 경로 설정 ★★★
    DATA_PATH = r"E:\Benchmark Code\benchmarktu1402-master\f_accerlerations"
    
    DT = 0.001              # Time increment (seconds)
    FS = 1000.0             # Sampling frequency (1000 Hz)
    
    # Target sensor nodes (1-indexed -> 0-indexed)
    TARGET_NODES = [3, 21, 39, 57, 63, 81, 99, 117]
    SENSOR_INDICES = [x - 1 for x in TARGET_NODES]
    
    # Expected mode frequencies (Hz)
    TARGET_MODES = [20, 22, 55, 60]
    
    # Analysis parameters
    NPERSEG = 16384
    SEARCH_RANGE_HZ = 3.0
    PEAK_PROMINENCE = 0.5
    FREQ_DISPLAY_MAX = 100


# ==========================================
# 2. Data Loading - 당신의 방식 참조
# ==========================================
def load_acceleration_data(filename: str, config: Config = None) -> Optional[np.ndarray]:
    """
    Load acceleration data using np.loadtxt (your method).
    
    Returns:
        np.ndarray: Shape (n_nodes, n_samples) or None if failed
    """
    if config is None:
        config = Config()
    
    filepath = os.path.join(config.DATA_PATH, filename)
    
    print(f"[Info] Loading:  {filepath}")
    
    if not os.path.exists(filepath):
        print(f"[Error] File not found: {filepath}")
        return None
    
    try:
        # ★ 당신의 방식:  np.loadtxt 사용 ★
        full_data = np.loadtxt(filepath)
        
        print(f"[Info] Raw data shape: {full_data. shape}")
        
        # Shape 확인:  (samples, nodes) -> (nodes, samples)로 변환
        if full_data.shape[0] < full_data. shape[1]:
            # 이미 (nodes, samples) 형태
            pass
        else:
            # (samples, nodes) -> transpose to (nodes, samples)
            full_data = full_data. T
        
        print(f"[Success] Final data shape: {full_data.shape}")
        return full_data
        
    except Exception as e: 
        print(f"[Error] Failed to load {filename}: {e}")
        return None


def extract_sensor_data(data: np.ndarray, config: Config = None) -> np.ndarray:
    """
    Extract data from selected sensor nodes.
    
    Returns:
        np.ndarray: Shape (n_selected_sensors, n_samples)
    """
    if config is None:
        config = Config()
    
    # 유효한 인덱스만 선택
    max_idx = data.shape[0] - 1
    valid_indices = [i for i in config.SENSOR_INDICES if i <= max_idx]
    
    return data[valid_indices, :]


# ==========================================
# 3. Signal Analysis (FFT & PSD)
# ==========================================
def analyze_signal(
    data_matrix: np.ndarray,
    method: str = 'PSD',
    config: Config = None,
    apply_detrend: bool = True
) -> Tuple[np.ndarray, np.ndarray]: 
    """
    Analyze signal using FFT or PSD method.
    """
    if config is None: 
        config = Config()
    
    # Extract sensor data
    signals = extract_sensor_data(data_matrix, config)
    
    # Remove DC offset and linear trend
    if apply_detrend:
        signals = detrend(signals, axis=1, type='linear')
    
    if method. upper() == 'FFT':
        return _compute_fft(signals, config)
    elif method.upper() == 'PSD':
        return _compute_psd(signals, config)
    else:
        raise ValueError(f"Unknown method:  {method}")


def _compute_fft(signals: np.ndarray, config: Config) -> Tuple[np.ndarray, np.ndarray]:
    """Compute FFT magnitude spectrum."""
    n_samples = signals.shape[1]
    
    # Apply Hanning window to reduce spectral leakage
    window = np.hanning(n_samples)
    windowed_signals = signals * window
    
    # Compute FFT
    freqs = rfftfreq(n_samples, d=config.DT)
    fft_vals = rfft(windowed_signals, axis=1)
    
    # Normalize by window sum
    window_correction = 2.0 / window.sum()
    magnitude = np.abs(fft_vals) * window_correction
    
    # Average across sensors
    avg_magnitude = np.mean(magnitude, axis=0)
    
    # Convert to dB
    magnitude_db = 20 * np.log10(np.maximum(avg_magnitude, 1e-20))
    
    return freqs, magnitude_db


def _compute_psd(signals: np.ndarray, config: Config) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Power Spectral Density using Welch method."""
    freqs, pxx = welch(
        signals,
        fs=config.FS,
        window='hann',
        nperseg=config.NPERSEG,
        noverlap=config.NPERSEG // 2,
        axis=1,
        detrend='linear'
    )
    
    # Average across sensors
    avg_pxx = np.mean(pxx, axis=0)
    
    # Convert to dB
    psd_db = 10 * np.log10(np.maximum(avg_pxx, 1e-20))
    
    return freqs, psd_db


# ==========================================
# 4. Peak Detection
# ==========================================
def find_mode_peaks(
    freqs: np.ndarray,
    mag_db: np.ndarray,
    target_modes: List[float],
    config: Config = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect peaks near target mode frequencies."""
    if config is None:
        config = Config()
    
    peaks, _ = find_peaks(mag_db, prominence=config.PEAK_PROMINENCE, distance=10)
    
    peak_freqs = freqs[peaks]
    peak_mags = mag_db[peaks]
    
    detected_freqs = []
    detected_mags = []
    
    for target in target_modes:
        mask = (
            (peak_freqs >= target - config.SEARCH_RANGE_HZ) &
            (peak_freqs <= target + config.SEARCH_RANGE_HZ)
        )
        
        candidates_f = peak_freqs[mask]
        candidates_m = peak_mags[mask]
        
        if len(candidates_f) > 0:
            best_idx = np.argmax(candidates_m)
            detected_freqs. append(candidates_f[best_idx])
            detected_mags.append(candidates_m[best_idx])
        else:
            # Fallback:  find max in raw spectrum within range
            raw_mask = (
                (freqs >= target - config.SEARCH_RANGE_HZ) &
                (freqs <= target + config.SEARCH_RANGE_HZ)
            )
            if raw_mask.any():
                best_idx = np.argmax(mag_db[raw_mask])
                detected_freqs.append(freqs[raw_mask][best_idx])
                detected_mags.append(mag_db[raw_mask][best_idx])
            else:
                detected_freqs.append(np. nan)
                detected_mags.append(np.nan)
    
    return np.array(detected_freqs), np.array(detected_mags)


# ==========================================
# 5. Main Execution
# ==========================================
def main():
    config = Config()
    
    print("=" * 60)
    print("Signal Analysis - FFT & PSD")
    print(f"Data path: {config. DATA_PATH}")
    print("=" * 60)
    
    # Load healthy reference data
    healthy_data = load_acceleration_data("fh_accelerations.dat", config)
    
    if healthy_data is None:
        print("[Error] Cannot proceed without healthy reference data")
        return
    
    # Analyze healthy data
    print("\n[Analysis] Processing healthy data...")
    f_fft_h, db_fft_h = analyze_signal(healthy_data, 'FFT', config)
    f_psd_h, db_psd_h = analyze_signal(healthy_data, 'PSD', config)
    
    modes_fft_h, _ = find_mode_peaks(f_fft_h, db_fft_h, config.TARGET_MODES, config)
    modes_psd_h, _ = find_mode_peaks(f_psd_h, db_psd_h, config.TARGET_MODES, config)
    
    print(f"[Healthy] FFT Modes: {np.round(modes_fft_h, 2)} Hz")
    print(f"[Healthy] PSD Modes: {np.round(modes_psd_h, 2)} Hz")
    
    # Process damaged data
    print("\n" + "=" * 60)
    print("Processing damaged data...")
    
    fft_ratios = []
    psd_ratios = []
    damages = []
    f_fft_d, db_fft_d, f_psd_d, db_psd_d = None, None, None, None
    
    for i in range(1, 11):
        fname = f"f{i}_accelerations.dat"
        damage_percent = i * 10
        
        d_data = load_acceleration_data(fname, config)
        
        if d_data is not None: 
            damages.append(damage_percent)
            
            f_fft_d, db_fft_d = analyze_signal(d_data, 'FFT', config)
            modes_fft, _ = find_mode_peaks(f_fft_d, db_fft_d, config. TARGET_MODES, config)
            
            f_psd_d, db_psd_d = analyze_signal(d_data, 'PSD', config)
            modes_psd, _ = find_mode_peaks(f_psd_d, db_psd_d, config.TARGET_MODES, config)
            
            # Calculate frequency change ratio
            with np.errstate(divide='ignore', invalid='ignore'):
                r_fft = np.where(modes_fft_h != 0,
                                (modes_fft_h - modes_fft) / modes_fft_h, np.nan)
                r_psd = np.where(modes_psd_h != 0,
                                (modes_psd_h - modes_psd) / modes_psd_h, np.nan)
            
            fft_ratios. append(r_fft)
            psd_ratios.append(r_psd)
            print(f"  ✓ {fname} (Damage {damage_percent}%)")
    
    fft_ratios = np.array(fft_ratios)
    psd_ratios = np.array(psd_ratios)
    
    # Visualization
    if len(damages) > 0 and f_fft_d is not None: 
        create_visualizations(
            f_fft_h, db_fft_h, f_psd_h, db_psd_h,
            f_fft_d, db_fft_d, f_psd_d, db_psd_d,
            damages, fft_ratios, psd_ratios, config
        )
    
    # Print summary
    print_summary(damages, fft_ratios, psd_ratios, config. TARGET_MODES)


# ==========================================
# 6. Visualization
# ==========================================
def create_visualizations(
    f_fft_h, db_fft_h, f_psd_h, db_psd_h,
    f_fft_d, db_fft_d, f_psd_d, db_psd_d,
    damages, fft_ratios, psd_ratios, config
):
    """Create analysis visualization plots."""
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']
    
    # Plot 1: FFT Spectrum
    axs[0, 0].plot(f_fft_h, db_fft_h, 'k-', lw=1.5, label='Healthy')
    axs[0, 0].plot(f_fft_d, db_fft_d, 'r--', lw=1, alpha=0.7,
                   label=f'Damage {damages[-1]}%')
    axs[0, 0].set_title("FFT Magnitude Spectrum", fontsize=12, fontweight='bold')
    axs[0, 0].set_xlabel("Frequency (Hz)")
    axs[0, 0].set_ylabel("Magnitude (dB)")
    axs[0, 0].set_xlim(0, config. FREQ_DISPLAY_MAX)
    axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: PSD Spectrum
    axs[0, 1].plot(f_psd_h, db_psd_h, 'k-', lw=1.5, label='Healthy')
    axs[0, 1].plot(f_psd_d, db_psd_d, 'r--', lw=1, alpha=0.7,
                   label=f'Damage {damages[-1]}%')
    axs[0, 1].set_title("Power Spectral Density (Welch)", fontsize=12, fontweight='bold')
    axs[0, 1].set_xlabel("Frequency (Hz)")
    axs[0, 1].set_ylabel("PSD (dB/Hz)")
    axs[0, 1].set_xlim(0, config.FREQ_DISPLAY_MAX)
    axs[0, 1].legend()
    axs[0, 1]. grid(True, alpha=0.3)
    
    # Plot 3: FFT Change Ratio
    for m_idx, mode_freq in enumerate(config.TARGET_MODES):
        axs[1, 0].plot(damages, fft_ratios[:, m_idx], 'o-',
                       color=colors[m_idx], lw=2, markersize=6,
                       label=f'Mode {m_idx+1} (~{mode_freq}Hz)')
    axs[1, 0].set_title("Frequency Change Ratio (FFT)", fontsize=12, fontweight='bold')
    axs[1, 0].set_xlabel("Damage Level (%)")
    axs[1, 0].set_ylabel("Change Ratio (fH-fD)/fH")
    axs[1, 0].legend(loc='upper left')
    axs[1, 0].grid(True, alpha=0.3)
    axs[1, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Plot 4: PSD Change Ratio
    for m_idx, mode_freq in enumerate(config.TARGET_MODES):
        axs[1, 1]. plot(damages, psd_ratios[:, m_idx], 's--',
                       color=colors[m_idx], lw=2, markersize=6,
                       label=f'Mode {m_idx+1} (~{mode_freq}Hz)')
    axs[1, 1]. set_title("Frequency Change Ratio (PSD)", fontsize=12, fontweight='bold')
    axs[1, 1].set_xlabel("Damage Level (%)")
    axs[1, 1]. set_ylabel("Change Ratio (fH-fD)/fH")
    axs[1, 1].legend(loc='upper left')
    axs[1, 1].grid(True, alpha=0.3)
    axs[1, 1]. axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(config.DATA_PATH, 'signal_analysis_results.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n[Saved] Results saved to: {save_path}")
    plt.show()


def print_summary(damages, fft_ratios, psd_ratios, target_modes):
    """Print analysis summary statistics."""
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)
    
    for m_idx, mode_freq in enumerate(target_modes):
        print(f"\nMode {m_idx+1} (~{mode_freq} Hz):")
        
        fft_vals = fft_ratios[:, m_idx]
        psd_vals = psd_ratios[:, m_idx]
        
        fft_valid = fft_vals[~np.isnan(fft_vals)]
        psd_valid = psd_vals[~np.isnan(psd_vals)]
        
        if len(fft_valid) > 0:
            print(f"  FFT:  Max ratio = {np.max(fft_valid)*100:.2f}% at {damages[np.nanargmax(fft_vals)]}% damage")
        if len(psd_valid) > 0:
            print(f"  PSD: Max ratio = {np.max(psd_valid)*100:.2f}% at {damages[np.nanargmax(psd_vals)]}% damage")


if __name__ == "__main__":
    main()
