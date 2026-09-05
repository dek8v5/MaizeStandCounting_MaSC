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
     mosaic mode:
          python masc_main.py -image_path data/mosaic/global_mosaic_7.png -save_path results/mosaic-mode-publish -mode mosaic -num_classes 3
     raw mode:
         python masc_main.py -image_path data/group_002/ -save_path results/raw_group2 -mode raw -num_classes 3 -hm data/group_002/H_asift_group_002.csv
'''



import cv2
import numpy as np
import os
import time
from matplotlib import pyplot as plt
from yolov9.detect import *
from masc_python.tif2png import tif2png, rot_radon, label_rotation
from masc_python.fragment_mosaic import fragment_mosaic
from masc_python.mosaicback import mosaicback
from masc_python.draw_bbox import draw_bbox
from masc_python.detect_range import detect_range
from masc_python.detect_row import detect_row
from masc_python.count_object import count_object
from masc_python.mosaic_label import mosaic_label
import subprocess
from datetime import datetime
import csv


def dmc_from_mosaic(image_path, save_path, num_class, mode):

    
    frag = [1280]
    #num_rows = 28
    #num_range = 4
    #num_class = 3
    empty_row = []
    file_format = 'png'
    
    	
    
    #raw_files = image_path
    
    print('after raw_files')
    print(image_path)
    #img = cv2.imread(image_path)
    
    if file_format == 'png':
        img = cv2.imread(image_path)
        if img is None:
           raise FileNotFoundError(f"Could not read image: {image_path}")


    else:
        img = tif2png(image_path)
        cv2.imwrite(os.path.join(save_path, 'preprocessed_mosaic.png'),img)
    #h_ori, w_ori, c = img.shape
    #print(h_ori, w_ori)
 
    #print("Before Radon:", img.shape)

    radon_start_time = time.time()
    img, degree = rot_radon(img)
    radon_elapsed_time = time.time() - radon_start_time
    print('==========================================================')
    print('RADON processing time: ', radon_elapsed_time)

    #print("After Radon:", img.shape)

    #cv2.imwrite(os.path.join(save_path, "rotated_mosaic_after_radon.png"),img)

    frag_start_time = time.time()
    att = fragment_mosaic(img, frag, save_path)
    frag_elapsed_time = time.time() - frag_start_time
    print('==========================================================')
    print('PATCHIFY processing time: ', frag_elapsed_time)


    frag_dir = os.path.join(save_path, f'fragment_{frag[0]}')
   
    yolo_path = os.path.join(frag_dir, 'yolo_result')

    print(yolo_path)	
   	# after fragmented, feed the fragments to seedling detector
   	# ----------------------------------------------------- 
   	#
    yolo_start_time = time.time()
    run_yolo_detection(frag_dir, yolo_path)
    yolo_elapsed_time = time.time() - yolo_start_time
    print('==========================================================')
    print('YOLO processing time: ', yolo_elapsed_time)


    mozback_start_time = time.time()
    label_all = mosaicback(yolo_path, att, frag)    #filtered out the label (remove duplication from overlap area)
    #print(label_all)
    mozback_elapsed_time = time.time() - mozback_start_time
    print('==========================================================')
    print('MOSAIC BACK & BBOX Aggregation processing time: ', mozback_elapsed_time)


    counting(img, save_path, label_all, num_class, mode, frag[0])
		
		
def dmc_from_raw(image_path, save_path, homography, num_class, mode):
    #if not os.path.exists(save_path):
    #   os.mkdir(save_path)
			 
    print(save_path)
    rawyolo_start_time = time.time()
    yolo_path = os.path.join(save_path, 'yolo_result')

    #star of the show
    run_yolo_detection(image_path, yolo_path)
    

    label_all = mosaic_label(yolo_path, homography, os.path.join(yolo_path, 'labels'))
    #print(len(label_all))
    
    #just for test
    #label_file = os.path.join(save_path, "2024-10-04_23-00-16_full.txt")
    #np.savetxt(label_file, label_all, fmt=["%d", "%.6f", "%.6f", "%.6f", "%.6f", "%.6f"])
    

    rawyolo_elapsed_time = time.time() - rawyolo_start_time
    print('==========================================================')
    print('RAW yolo pred Aggregation processing time: ', rawyolo_elapsed_time)
    
    mosaic_path = os.path.join(save_path, 'mosaic')
    
    all_file_inside_mosaic_path = sorted([f for f in os.listdir(mosaic_path) if f.endswith('.png')])
    img = cv2.imread(os.path.join(mosaic_path, all_file_inside_mosaic_path[0]))
    h,w,c = img.shape
    #normalized the label first before sending it to counting
	
    #justforpaper
    #label_file = os.path.join(save_path, "2024-10-04_23-00-16_full.txt")
    #label_all = np.loadtxt(label_file, ndmin=2)

    counting(img, save_path, label_all, num_class, mode)

def run_yolo_detection(image_path, save_path):
    #result_dir
    print('inside run yolo')
    print(image_path)
    print(save_path)
    
    save_path = os.path.abspath(save_path)
    project_dir = os.path.dirname(save_path)
    run_name = os.path.basename(save_path)
    print('project_dir: ', project_dir)
    print('run_name: ', run_name)
    command = [
        'python', 'yolov9/detect.py',
        '--source', image_path,
        '--img', '640',
        '--device', '0,1',
        '--weights', 'yolov9_model/best.pt',
        '--project', project_dir,
        '--name', run_name,
        '--save-txt',
        #'--save-crop',
        '--save-conf',
        '--conf-thres', '0.3'
    ]

    result = subprocess.run(command, capture_output=True, text=True)


    if result.returncode == 0:
        print("detection is done. wohoooooo")
    else:
        print("ummm, something is wrong in running yolo")

		
def counting(img, save_path, label_all, num_class, mode, frag='raw'):
    custom_color = True
    h_ori, w_ori, c = img.shape
    each_range = True
    to_right = True
    num_border = 0
    
    #num_range = 7
    #conditions = (label_all_raw[:, 1] > 0) & (label_all_raw[:, 1] < w_ori) & \
    #         (label_all_raw[:, 2] > 0) & (label_all_raw[:, 2] < h_ori) & \
    #         (label_all_raw[:, 3] > 0) & (label_all_raw[:, 3] < w_ori) & \
    #         (label_all_raw[:, 4] > 0) & (label_all_raw[:, 4] < h_ori)
    #label
    #label_all_filtered = label_all[(label_all[:, 1:5] > 0) & (label_all[:, 1:5] < 1)]
    #label_all = label_all_raw[conditions]
    #comment only for test
    #filtered_label = label_all
    
    nms_start_time = time.time() 
    
    index = cv2.dnn.NMSBoxes(label_all[:, 1:5], label_all[:, 5], score_threshold = 0.25, nms_threshold=0.25)
    filtered_label = label_all[index, :]
    
    NMS_elapsed_time = time.time() - nms_start_time
    print('==========================================================')
    print('NMS processing time: ', NMS_elapsed_time)


    #print(' bbox filtered: ', filtered_label)
    if mode == 'raw': 
       original_img = img.copy()
       radon_start_time = time.time()
       img, rot_angle = rot_radon(img)
       filtered_label = label_rotation(original_img, filtered_label, rot_angle)
       radon_elapsed_time = time.time() - radon_start_time
       print('==========================================================')
       print('RADON processing time: ', radon_elapsed_time)
    
    img_with_bbx = draw_bbox(img, filtered_label, num_class, custom_color, False)
    cv2.imwrite(os.path.join(save_path, ('mosaic_with_bbox_'+f'fragment_{frag}'+'.png')), np.uint8(img_with_bbx))

    print('image is saved at: ', os.path.join(save_path, ('mosaic_with_bbox_'+f'fragment_{frag}'+'.png')))

    ####### just for posters ##########
    #save label
    #filtered_label[:, 2] = ((filtered_label[:, 2] + (filtered_label[:, 4] / 2)) / h_ori)
    #filtered_label[:, 1] = ((filtered_label[:, 1] + (filtered_label[:, 3] / 2)) / w_ori)
    #filtered_label[:, 4] = filtered_label[:, 4] / h_ori
    #filtered_label[:, 3] = filtered_label[:, 3] / w_ori
    #print(filtered_label)
    #with open(f'{save_path}/bounding_box_mosaic.csv', mode='w', newline='') as file:
    #    writer = csv.writer(file)
    #    for row in filtered_label:
    #        writer.writerow(row)

		
    
    
    center_crop = np.zeros((h_ori, w_ori))
    y_coords = (filtered_label[:, 2] + (filtered_label[:, 4] / 2)).astype(int)
    x_coords = (filtered_label[:, 1] + (filtered_label[:, 3] / 2)).astype(int)

    y_coords = np.clip(y_coords, 0, center_crop.shape[0] - 1)
    x_coords = np.clip(x_coords, 0, center_crop.shape[1] - 1)
    
    lind = np.ravel_multi_index((y_coords, x_coords), center_crop.shape)
    #lind = np.ravel_multi_index(((filtered_label[:, 2] + (filtered_label[:, 4] / 2)).astype(int), (filtered_label[:, 1] + (filtered_label[:, 3] / 2)).astype(int)), center_crop.shape)
    center_crop.flat[lind] = 1

    print('before row separator')
    range_start_time = time.time()
    range_separator, img_w_range = detect_range(img_with_bbx, center_crop, True, True, save_path)
    
    #range detection
    print('range separator: ', range_separator)
    cv2.imwrite(os.path.join(save_path, ('mosaic_with_range_'+f'fragment_{frag}'+'.png')), np.uint8(img_w_range))
    range_elapsed_time = time.time() - range_start_time
    print('==========================================================')
    print('RANGE processing time: ', range_elapsed_time)

    #row detection
    row_start_time = time.time()
    row_separator, img_w_range_row = detect_row(img_w_range, center_crop, range_separator, each_range, True, True, save_path)
    row_elapsed_time = time.time() - row_start_time
    print('==========================================================')
    print('ROW processing time: ', row_elapsed_time)
    print('row_separator: ', row_separator)

    cv2.imwrite(os.path.join(save_path, ('mosaic_with_row_'+f'fragment_{frag}'+'.png')), np.uint8(img_w_range_row))
    
    
    '''
    saving instead of showing
    July.9.2024
    '''
    #plot both ranges and row
    #plt.figure()
    #plt.imshow(img_w_range_row.astype(np.uint8))
    #plt.show()
   
    count_start_time = time.time()
    img, count_per_row, final_row = count_object(img_w_range_row, range_separator, row_separator, filtered_label, to_right, num_border, save_path)
    count_elapsed_time = time.time() - count_start_time
    print('==========================================================')
    print('COUNT processing time: ', count_elapsed_time)
    #print('row_separator: ', _separator)

    #plt.figure()
    #plt.imshow(img)
    #plt.show()
    
    cv2.imwrite(os.path.join(save_path, 'final_stand_count_per_row.png'), img)
  


if __name__ == "__main__":	

  start_time = time.time()
  #argument parsers
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('-image_path', type=str, help="paths to one or more images or image directories")
  parser.add_argument('-mode', type=str, default='mosaic',  help="stand count mode, either mosaic or raw")
  parser.add_argument('-save_path', dest='save_path', default="results/mosaic_"+datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), type=str, help="path to save result")
  parser.add_argument('-hm', type=str, default = "homography_matrices/H_surf.csv", help='txt file that stores homography matrices')
  parser.add_argument('-num_classes', type=int, default=3, help="number of classes to detect on yolo")
 
  args = parser.parse_args()
  #for mosaic mode
  if args.mode == 'mosaic':
    mosaic = True
  #for raw mode
  else:
    mosaic = False 
  	 
  image_path = args.image_path
  save_path = args.save_path
  
  #homography = args.hm
  num_classes = args.num_classes

  if not os.path.exists(save_path):
    os.makedirs(save_path)
  
  print(args.mode)
  if args.mode=="mosaic":
    dmc_from_mosaic(image_path, save_path, num_classes, args.mode)
  else:
    dmc_from_raw(image_path, save_path, args.hm, num_classes, args.mode)

  print("Elapsed time: ", time.time() - start_time)
		
