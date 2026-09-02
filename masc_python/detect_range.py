import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import sys
from scipy.signal import find_peaks

np.set_printoptions(threshold=sys.maxsize)


def detect_range(img, center_crop, draw, plot_graph, save_path):
    sum_range = np.sum(center_crop, axis=0)
    h_ori, w_ori, c = img.shape
    data = sum_range.astype(np.float32)

    window_size = max(1, round(len(data) / 100))
    smoothed_data = np.convolve(data, np.ones(window_size) / window_size, mode='same')

    positive_signal = smoothed_data[smoothed_data > 0]
    typical_signal = np.percentile(positive_signal, 75)

    troughs, _ = find_peaks(-smoothed_data, distance=max(1, window_size * 2))

    search_radius = max(1, window_size * 4)
    gaps = []
    gap_ratios = []

    for trough in troughs:
        left_start = max(0, trough - search_radius)
        right_end = min(len(smoothed_data), trough + search_radius)

        left_signal = smoothed_data[left_start:trough]
        right_signal = smoothed_data[trough + 1:right_end]

        if len(left_signal) == 0 or len(right_signal) == 0:
            continue

        left_level = np.percentile(left_signal, 75)
        right_level = np.percentile(right_signal, 75)
        flank_level = min(left_level, right_level)

        if flank_level <= 0:
            continue

        valley_level = smoothed_data[trough]
        valley_ratio = valley_level / flank_level

        strong_left = left_level >= 0.5 * typical_signal
        strong_right = right_level >= 0.5 * typical_signal

        print(
            "Trough:", trough,
            "| valley:", round(float(valley_level), 4),
            "| left:", round(float(left_level), 4),
            "| right:", round(float(right_level), 4),
            "| ratio:", round(float(valley_ratio), 4),
            "| strong:", strong_left, strong_right
        )

        if valley_ratio <= 0.20 and strong_left and strong_right:
            gaps.append(trough)
            gap_ratios.append(valley_ratio)

    gaps = np.array(gaps, dtype=int)

    print("Window size:", window_size)
    print("Typical signal:", typical_signal)
    print("Candidate troughs:", troughs)
    print("Detected gaps:", gaps)
    print("Detected gap ratios:", gap_ratios)

    if plot_graph:
        plt.figure(figsize=(10, 6))
        plt.plot(data, label='Original Data')
        plt.plot(smoothed_data, label='Smoothed Data', linewidth=2)
        plt.plot(troughs, smoothed_data[troughs], "kx", label='Candidate gaps')

        if len(gaps) > 0:
            plt.plot(gaps, smoothed_data[gaps], "r*", markersize=12, label='Selected gaps')

        plt.legend()
        plt.title('Detected Range Gaps')
        plt.savefig(os.path.join(save_path, 'range_plot_original_smooth.png'))
        plt.close()

    if draw:
        for lm in gaps:
            cv2.line(img, (lm, 0), (lm, h_ori), (16, 255, 239), 5)

    return gaps, img
