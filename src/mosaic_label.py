import os
import numpy as np
import csv
from numpy.linalg import inv
import cv2


def homography_manager(homography):
    H = []
    H_cum = []
    counter = 0
    #read homography file and get offset
    with open(homography, 'r') as csvFile:
        reader = csv.reader(csvFile, delimiter = ",")

        for row in reader:
            H_each = np.asarray(row, dtype=np.float).reshape(3,3)

            H.append(H_each)

            #print(H_each)
 

            if counter == 0:
               H_temp = inv(H[counter])
               #H_temp = H_temp/H_temp[2,2]
               H_cum.append(( H_temp))
               
            elif counter > 0 :
               
               H_temp = np.dot((H_cum[counter-1]), inv(H[counter]))
               #H_temp = H_temp/H_temp[2,2]
               H_cum.append(H_temp)
            counter += 1
    #print(H_cum)
    return H_cum


def calculate_offset(image_path, H_cum):
    all_img = sorted([f for f in os.listdir(image_path) if f.endswith('.png')])
    img = cv2.imread(os.path.join(image_path, all_img[0]))
    h, w, _ = img.shape
    print('Image dimensions (h, w):', h, w)
    
    #w = np.round(3816 / 3)
    #h = np.round(2138 / 3)
    
    corners_h = []
    corners_4 = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    
    corners_h.append(corners_4)
    
    for i in range(len(H_cum)):
        transformed_corners = cv2.perspectiveTransform(corners_4.reshape((-1, 1, 2)), H_cum[i])
        corners_h.append(transformed_corners.reshape(-1, 2))
    
    corners_h_arr = np.array(corners_h)
    
    max_x = np.max(corners_h_arr[..., 0].flatten())
    min_x = np.min(corners_h_arr[..., 0].flatten())
    max_y = np.max(corners_h_arr[..., 1].flatten())
    min_y = np.min(corners_h_arr[..., 1].flatten())
    
    print("Max x:", max_x, "Min x:", min_x, "Max y:", max_y, "Min y:", min_y)
    
    offset_x = np.ceil(-min_x) if min_x <= 0 else 0
    offset_y = np.ceil(-min_y) if min_y <= 0 else 0
    
    max_x = np.ceil(offset_x+max_x )
    max_y = np.ceil( offset_y+max_y)
    
    print('Calculated global canvas size: (max_x, max_y) =', max_x, max_y)
    
    global_canvas = np.zeros((int(np.floor(max_y)), int(np.floor(max_x)), 3), dtype=np.uint8)
    
    offset_matrix = np.array([[1, 0, offset_x], [0, 1, offset_y], [0, 0, 1]], dtype=np.float32)
    
    return offset_matrix, global_canvas

	
def apply_homography(bounding_boxes, homography_matrix=None, offset_matrix=None):
    transformed_bounding_boxes = []
    
    for box in bounding_boxes:
        x_min, y_min, x_max, y_max = box
        
        corners = np.array([
            [x_min, y_min, 1],
            [x_max, y_min, 1],
            [x_max, y_max, 1],
            [x_min, y_max, 1]
        ])
        
        if homography_matrix is not None:
            transformed_corners = homography_matrix @ corners.T
            transformed_corners = transformed_corners.T
            transformed_corners = transformed_corners[:, :2] / transformed_corners[:, 2].reshape(-1, 1)
        else:
            transformed_corners = corners[:, :2]        
        if offset_matrix is not None:
            transformed_corners = np.dot(transformed_corners, offset_matrix[:2, :2].T) + offset_matrix[:2, 2]
        
        x_min_new = np.min(transformed_corners[:, 0])
        y_min_new = np.min(transformed_corners[:, 1])
        x_max_new = np.max(transformed_corners[:, 0])
        y_max_new = np.max(transformed_corners[:, 1])
        bbw = (x_max_new - x_min_new).astype(int)
        bbh = (y_max_new - y_min_new).astype(int)
        cx = x_min_new 
        cy = y_min_new
        transformed_bounding_boxes.append([cx, cy, bbw, bbh])
        
    return np.array(transformed_bounding_boxes)

	
def mosaic_label(yolo_path, homography, save_path):
    pred_mosaic = []
    all_img = sorted([f for f in os.listdir(yolo_path) if f.endswith('.png')])
    img = cv2.imread(os.path.join(yolo_path,all_img[0]))
    h, w, c = img.shape
    #h, w, c = 4890,1404,3 
    print(h,w)
    label_final_all = []
    label_final = []
    label_dir = os.path.join(yolo_path, 'labels')
    
    all_txt = sorted([f for f in os.listdir(label_dir) if f.endswith('.txt')])

    homography_accum = homography_manager(homography)
    offset_matrix, global_canvas = calculate_offset(yolo_path, homography_accum)
    #offset_matrix = np.array([[1,0,9982],[0,1,0],[0,0,1]])
    #global_canvas = np.zeros((11375, 5576, 3))
    print(offset_matrix)
    gh, gw, c = global_canvas.shape
    for i in range(len(all_txt)):
       img = cv2.imread(os.path.join(yolo_path,all_img[i]))
       h, w, c = img.shape
       txt_path = os.path.join(label_dir, all_txt[i])

       with open(txt_path, 'r') as file:
         reader = csv.reader(file, delimiter=' ')
         pred = np.array([[float(val) for val in row] for row in reader])
       #print(pred)
       if pred.size==0:
         continue;
          
       pred[:, 1] *= w
       pred[:, 3] *= w
       pred[:, 2] *= h
       pred[:, 4] *= h

       #print('================================================================')
       squares_coord = np.zeros((pred.shape[0], 4))
       squares_coord[:, 0] = np.round(pred[:, 1] - pred[:, 3] / 2)
       squares_coord[:, 1] = np.round(pred[:, 2] - pred[:, 4] / 2)
       squares_coord[:, 2] = np.round(pred[:, 1] + pred[:, 3] /2)
       squares_coord[:, 3] = np.round(pred[:, 2] + pred[:, 4] /2)
       #print('squares coord', squares_coord)
       #print('shape squares coord', squares_coord.shape)

       '''
       squares = np.vstack([
           squares_coord[:, 0], squares_coord[:, 1], np.ones(squares_coord.shape[0]),
           squares_coord[:, 2], squares_coord[:, 1], np.ones(squares_coord.shape[0]),
           squares_coord[:, 0], squares_coord[:, 3], np.ones(squares_coord.shape[0]),
           squares_coord[:, 2], squares_coord[:, 3], np.ones(squares_coord.shape[0])
       ])
       print('squares coordinates', squares)
       '''
       

       '''
       square2 = np.reshape(squares, (3, 4, -1))
       print('shape of square2: ', square2.shape)
       squares_warp = np.zeros_like(square2)

       #offset_matrix = offset_matrix.astype(float)
       #homography = homography.astype(float)
       #square2 = square2.astype(float)
       '''
       if i == 0:
           transformed_bounding_boxes = apply_homography(squares_coord, offset_matrix)
       else:
           transformed_bounding_boxes = apply_homography(squares_coord,  homography_accum[i-1], offset_matrix)
       
       '''
       #print(squares_warp[:, :, j])
       squares_warp[0] /= squares_warp[2]
       squares_warp[1] /= squares_warp[2]
       squares_warp[2] /= squares_warp[2]
       print('shape of squares warp: ', squares_warp)
       max_w = np.max(squares_warp[0], axis=0)
       max_h = np.max(squares_warp[1], axis=0)
       min_w = np.min(squares_warp[0], axis=0)
       min_h = np.min(squares_warp[1], axis=0)
       print(max_w, min_w)
       #new coordinate
       bbw = (max_w - min_w).astype(int)
       bbh = max_h - min_h.astype(int)
       #print(bbw, bbh)
       c_x = min_w.astype(int)
       c_y = min_h.astype(int)
       print('c_x ', c_x)
       '''
       #print(transformed_bounding_boxes)
       pred_warped = pred.copy()
       pred_warped[:, 1] = transformed_bounding_boxes[:,0]
       pred_warped[:, 2] = transformed_bounding_boxes[:,1]
       pred_warped[:, 3] = transformed_bounding_boxes[:,2]
       pred_warped[:, 4] = transformed_bounding_boxes[:,3]
  
       pred_mosaic.append(pred_warped)
  
    pred_mosaic = np.vstack(pred_mosaic)
    print('shape of the predicted mosaic bbox',pred_mosaic.shape)

    return pred_mosaic
    
