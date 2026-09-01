import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from scipy.signal import argrelextrema
import os
import sys
from scipy.signal import find_peaks
np.set_printoptions(threshold=sys.maxsize)

def detect_range(img, center_crop, num_range, draw, plot_graph, save_path):
    sum_range = np.sum(center_crop, axis=0)
    h_ori, w_ori, c = img.shape

    sum_range_ori = sum_range
    
    data = sum_range_ori
    
    window_size = round(len(sum_range) / 100)  
    smoothed_data = np.convolve(data, np.ones(window_size)/window_size, mode='same')

    peaks, _ = find_peaks(smoothed_data, distance = round(len(sum_range) / num_range))  
     
    troughs, _ = find_peaks(-smoothed_data, distance=round(len(sum_range) / (num_range*2)))  

    gaps = [trough for trough in troughs if any(peaks[i] < trough < peaks[i+1] for i in range(len(peaks)-1))]

    if plot_graph:
       plt.figure(figsize=(10, 6))
       plt.plot(data, label='Original Data')
       plt.plot(smoothed_data, label='Smoothed Data', linewidth=2)
       plt.plot(peaks, smoothed_data[peaks], "x", label='Peaks')
       plt.plot(gaps, smoothed_data[gaps], "r*", label='gaps')
       plt.legend()
       plt.title('Data with Identified Peaks')
       plt.savefig(os.path.join(save_path, 'range_plot_original_smooth.png'))

    groups = []
    start = 0
    for end in gaps:
       groups.append(data[start:end])
       start = end

    groups.append(data[start:])

    if draw:
        for lm in gaps:
            #cv2.line(img, (0, lm), (w_ori, lm), (16, 255, 239), 5)
            cv2.line(img, (lm, 0), (lm, h_ori), (16, 255, 239), 5)
    return gaps, img   
