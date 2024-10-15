#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
project: DeepMaizeCounter

author: Dewi Kharismawati

about this script:
    - this is dmc_main
    - this expects mosaic of seedling field
    - result will be png file of stand count contains:
          + seedling is detected
          + range is detected
          + row is detected
          + row number is defined
          + stand count on each row is printed

to call:
    python mini_mosaic_360.py -image_path /path/to/mini/mosaic -save_path /path/to/save

'''



import cv2
import numpy as np
import os
import time
from matplotlib import pyplot as plt
from tif2png import *


def dmc(directory, result_dir):
    start_time = time.time()

    frag = 1280
    num_rows = 4
    num_range = 1
    num_class = 3
    custom_color = True
    empty_row = []
    file_format = 'png'
    each_range = False
    to_right = True
    num_border = 0		

    raw_files = [f for f in os.listdir(directory) if f.endswith(file_format)]
    
    if file_format == 'png':
        img = cv2.imread(os.path.join(directory, raw_files[0]))
        img = rot_radon(img)
    else:
        img = tif2png(os.path.join(directory, raw_files[0]))
        cv2.imwrite(os.path.join(directory, 'preprocessed_mosaic.png'), img)

    h_ori, w_ori, c = img.shape

    att = fragment_mosaic(img, frag, directory)

		#after fragmented, feed the fragments to seedling detector
		#-----------------------------------------------------
		'''
    label_all = mosaicback(directory, att, frag)

    #filtered out the label (remove duplication from overlap area)
    label_all_filtered = label_all[(label_all[:, 1:5] > 0) & (label_all[:, 1:5] < 1)]

    selectedBbox, selectedScore, index = selectStrongestBbox(label_all[:, 1:5], label_all[:, 5], 'Min')
    filtered_label = label_all[index, :]

    img_with_bbx = draw_bbox(img, filtered_label, num_class, custom_color, filtered_label)

    center_crop = np.zeros((h_ori, w_ori))
    lind = np.ravel_multi_index((filtered_label[:, 2] + np.round(filtered_label[:, 4] / 2).astype(int), filtered_label[:, 1] + np.round(filtered_label[:, 3] / 2).astype(int)), center_crop.shape)
    center_crop.flat[lind] = 1

    range_separator, img_w_range = detect_range(img_with_bbx, center_crop, num_range, True, True, directory)
    row_separator, img_w_range_row = detect_row(img_w_range, center_crop, num_rows, range_separator, each_range, True, True, directory)

    #plot both ranges and row
    plt.figure()
    plt.imshow(img_w_range_row.astype(np.uint8))
    plt.show()

    img, count_per_row, final_row = count_object(img_w_range_row, range_separator, row_separator, filtered_label, to_right, num_border, directory)

    plt.figure()
    plt.imshow(img)
    plt.show()

    cv2.imwrite(os.path.join(directory, 'final_stand_count_per_row.png'), img)

    print("Elapsed time: ", time.time() - start_time)
    '''

		
if __name__ == "__main__":
	
  #argument parsers
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('-image_path', type=str, nargs='+', help="paths to one or more images or image directories")
  parser.add_argument("-save_path", dest='save_path', default="RESULTS/stitched_"+datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), type=str, help="path to save result")
  args = parser.parse_args()

	
	directory = args.image_path
  result_dir = args.save_path

	dmc(directory, result_directory)
