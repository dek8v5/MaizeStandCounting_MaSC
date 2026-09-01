import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from scipy.signal import argrelmin, argrelextrema
import os
from scipy.signal import find_peaks

def smoothdata(data, window_len):
    return uniform_filter1d(data, size=window_len)

def rescale(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def detect_row(img, center_crop, range_separator, each_range, draw, plot_graph, save_path):
    lm_row_len = 10000
    range_separator_array = np.asarray(range_separator)
    gaps = []
    gaps_all = []
    h_ori, w_ori, c = img.shape

    #idx = np.where(range_separator == 1)[0]
    range_separator = np.concatenate(([0], range_separator_array, [w_ori])).astype(int)
    print('range separator shape', range_separator.shape)
    print('separator', range_separator)

    if each_range:
        for i in range(len(range_separator)-1):
            print('i in row detection: ', i)
            print('range 1 in row detection: ', range_separator[i])
            print(' range 2 in row detection: ', range_separator[i+1])
            sum_col = np.sum(center_crop[:, range_separator[i]:range_separator[i+1]], axis=1)
            smoothed_data, peaks, gaps = find_row_gaps(sum_col)
            '''
            while lm_row_len > num_row - 1:
                sum_col = smoothdata(sum_col, window_len=round(len(sum_col) / 100))
                lm_col = argrelmin(sum_col, axis=0)# np.less)[0]
                lm_row_len = len(lm_col)
            '''
            if draw:
                for lm in gaps:
                    cv2.line(img, (range_separator[i], lm), (range_separator[i+1], lm), (16, 255, 239), 5)
										
            if plot_graph:
               plt.figure(figsize=(10, 6))
               plt.plot(sum_col, label='Original Data')
               plt.plot(smoothed_data, label='Smoothed Data', linewidth=2)
               plt.plot(peaks, smoothed_data[peaks], "x", label='Peaks')
               plt.plot(gaps, smoothed_data[gaps], "r*", label='gaps')
               plt.legend()
               plt.title('data with gaps')
               plt.savefig(f'{save_path}/row_gaps_on_range{i}.png')

										
            print('lm_col: ', gaps)
            gaps_all.append(gaps)


    else:
        sum_col = np.sum(center_crop, axis=1)

        smoothed_data, peaks, gaps = find_row_gaps(sum_col)
        #print(gaps)
        if plot_graph:
           plt.figure(figsize=(10, 6))
           plt.plot(sum_col, label='Original Data')
           plt.plot(smoothed_data, label='Smoothed Data', linewidth=2)
           plt.plot(peaks, smoothed_data[peaks], "x", label='Peaks')
           plt.plot(gaps, smoothed_data[gaps], "r*", label='gaps')
           plt.legend()
           plt.title('row separation plot')
           plt.savefig(os.path.join(save_path, 'row_plot_original_smooth.png'))
					 
        if draw:
            for lm in gaps:
                cv2.line(img, (0, lm), (w_ori, lm), (16, 255, 239), 5)
        gaps_all.append(gaps)
        '''
        while lm_row_len > num_row - 1:
            sum_col = sum_col #smoothdata(sum_col, window_len=round(len(sum_col) / 200))
            #lm_col = argrelextrema(sum_col, np.less)[0]
            lm_col = argrelmin(sum_col, axis=0)# np.less)[0]
            lm_row_len = len(lm_col)
        
        sum_col = rescale(sum_col)

        separator = np.tile(lm_col, (len(range_separator) - 1, 1))

        if draw:
            for lm in lm_col:
                cv2.line(img, (0, lm), (w_ori, lm), (16, 255, 239), 5)

        if plot_graph:
            plt.figure()
            plt.plot(sum_col)
            #plt.plot(sum_col[lm_col], lm_col, 'r*')
            plt.title('Plot sum green on width')
            plt.savefig(f'{save_path}/row_plot_single_range.png')
            plt.close()
        '''
    return gaps_all, img


def find_row_gaps(sum_rows):
   #window_size = round(len(sum_rows)/num_rows)
   smoothed_data = smoothdata(sum_rows, window_len=round(len(sum_rows) / 100))
   #smoothed_data = np.convolve(sum_rows, np.ones(window_size)/window_size, mode = 'same')

   peaks, _ = find_peaks(smoothed_data, distance = round(len(sum_rows)/100))
   troughs, _ = find_peaks(-smoothed_data, distance = round(len(sum_rows)/(100)))
   #print(peaks)
   gaps = [trough for trough in troughs if any(peaks[i] < trough < peaks[i+1] for i in range(len(peaks)-1))]
   f_gaps = filter_gaps(gaps, peaks)
   													
   return smoothed_data, peaks, f_gaps

def filter_gaps(gaps, peaks):

   gap_distances = np.diff(gaps)
   median_gap_distance = np.median(gap_distances)
   threshold_percentage = 0.2
   dynamic_threshold = median_gap_distance * threshold_percentage

   filtered_gaps = [gap for gap in gaps if all(abs(gap - peak) >= dynamic_threshold for peak in peaks)]

   return filtered_gaps
