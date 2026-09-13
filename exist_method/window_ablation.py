"""Controlled temporal-window override for the EAV dataset."""
import numpy as np
from scipy import signal
from scipy.integrate import simpson


def configure_eav_window(dataset_class, window_seconds):
    samples = int(round(100 * window_seconds))
    if samples <= 0 or 500 % samples:
        raise ValueError("window must divide the five-second (500 sample) trial")

    def segment(self, eeg_data, fs=100):
        del fs
        n, nodes, length = eeg_data.shape
        count = length // samples
        data = eeg_data[:, :, :count * samples]
        segmented = data.reshape(n, nodes, count, samples).transpose(0, 2, 1, 3)
        print(f"[window override] {window_seconds}s -> {segmented.shape} (T={count})")
        return segmented

    dataset_class.segment_into_1s_windows = segment

    def compute_de(self, eeg_data, fs=100):
        bands = ((1, 4), (4, 8), (8, 14), (14, 30), (30, 45))
        n, count, nodes, width = eeg_data.shape
        flat = eeg_data.reshape(n * count, nodes, width)
        nperseg = min(fs, width)
        frequencies, psd = signal.welch(
            flat, fs=fs, nperseg=nperseg, noverlap=nperseg // 2, axis=-1)
        features = np.zeros((n * count, nodes, len(bands)), dtype=np.float32)
        for index, (low, high) in enumerate(bands):
            mask = (frequencies >= low) & (frequencies <= high)
            power = (simpson(psd[..., mask], x=frequencies[mask], axis=-1)
                     if mask.sum() > 1 else psd[..., mask].sum(axis=-1))
            features[..., index] = .5 * np.log(2 * np.pi * np.e * (power + 1e-8))
        result = features.reshape(n, count, nodes, len(bands))
        print(f"[window override DE] -> {result.shape}")
        return result

    dataset_class.compute_de_features = compute_de
    return dataset_class
