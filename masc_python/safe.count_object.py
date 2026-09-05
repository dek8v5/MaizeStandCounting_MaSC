import cv2
import numpy as np
from scipy.io import savemat
import csv

def count_object(img, range_separator, row_separator, label, to_right, num_border, save_path):
    h_ori, w_ori, c = img.shape

    #idx_h = np.where(range_separator == 1)[0]
    range_separator_coord = np.concatenate(([0], range_separator, [w_ori]))

    counter = 1
    final_pred = []

    rowsz = []
    std_count = []
    print('range_separator: ',  range_separator_coord)
    for i in range(len(range_separator_coord) - 1):
        print(i)
        h_sep = np.concatenate(([1], row_separator[i], [h_ori]))

        diff_h_sep_min = np.min(np.abs(h_sep[:-1] - h_sep[1:]))
        if diff_h_sep_min > 200:
            diff_h_sep_min = 200

        row_actual = 1
        std = []

        for k in range(len(h_sep) - 1):
            rowsz.append([h_sep[k], h_sep[k + 1], range_separator_coord[i], range_separator_coord[i + 1]])

            corn_in_row = label[
                (range_separator_coord[i] <= label[:, 1] + label[:, 3] / 2) & (label[:, 1] + label[:, 3] / 2 <= range_separator_coord[i + 1]) &
                ( h_sep[k] <= label[:, 2] + label[:, 4] / 2) & (label[:, 2] + label[:, 4] / 2 <= h_sep[k + 1]), 0
            ]

            if len(corn_in_row) == 0:
                std_count.append(0)
                std.append(0)
            else:
                corn_in_row = corn_in_row + 1
                std_count.append(np.sum(corn_in_row))
                std.append(np.sum(corn_in_row))
            #print((range_separator_coord[i + 1] - range_separator_coord[i]) / 2)
            #print((diff_h_sep_min / 2))
            text_x = int((range_separator_coord[i] + ((range_separator_coord[i + 1] - range_separator_coord[i]) / 2) - (diff_h_sep_min / 2)))
            text_y = int((h_sep[k + 1] - ((h_sep[k + 1] - h_sep[k]) / 2) + (diff_h_sep_min * 0.3) / (20 * 2)))
            #print('txt_x and y: ', text_x, text_y)
            #print(img.shape[1])
            #print(img.shape[0])
            if 0 <= text_x < img.shape[1] and 0 <= text_y < img.shape[0]:
                #print('test')
                img = cv2.putText(img,
                                  str(int(std_count[-1])),
                                  (text_x, text_y),
                                  cv2.FONT_HERSHEY_SIMPLEX,
                                  int((diff_h_sep_min * 0.02)),
                                  (0, 0, 255),
                                  10,
                                  lineType=cv2.LINE_AA,
                                  bottomLeftOrigin=False)								
            '''
            img = cv2.putText(img,
                              str(std_count[-1].astype(int)),
                              (round(range_separator_coord[i + 1] - ((range_separator_coord[i + 1] - range_separator_coord[i]) / 2) - round(diff_h_sep_min / 2)),
                               round(h_sep[k + 1] - ((h_sep[k + 1] - h_sep[k]) / 2) + round(diff_h_sep_min * 0.3) / (20*2))),
                              cv2.FONT_HERSHEY_SIMPLEX,
                              100,#round(diff_h_sep_min * 0.3),
                              (0, 0, 255),
                              thickness=10,
                              lineType=cv2.LINE_AA,
                              bottomLeftOrigin=False)
            '''
            counter += 1
            row_actual += 1

        row_actual -= 1
        
        if to_right:
            if i % 2 == 0:
                row_counter_final = np.column_stack((np.arange(num_border + 1, row_actual - num_border + 1), np.flip(std[num_border:row_actual - num_border + 1])))
            else:
                row_counter_final = np.column_stack((np.arange(num_border + 1, row_actual - num_border + 1), std[num_border:row_actual - num_border + 1]))
        else:
            if i % 2 == 0:
                row_counter_final = np.column_stack((np.arange(num_border + 1, row_actual - num_border + 1), std[num_border:row_actual - num_border + 1]))
            else:
                row_counter_final = np.column_stack((np.arange(num_border + 1, row_actual - num_border + 1), np.flip(std[num_border:row_actual - num_border + 1])))
        
        final_pred.append(row_counter_final)

    rowsz = np.column_stack((rowsz, std_count))

    savemat(f'{save_path}/rows_coordinate_and_count.mat', {'rowsz': rowsz})

    with open(f'{save_path}/prediction_on_each_row.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        for row in final_pred:
            writer.writerows(row)

    #cv2.imshow('Image', img)
    #cv2.imwrite(f'{save_path}/webodm_yolo_detection.png', img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return img, rowsz, final_pred
