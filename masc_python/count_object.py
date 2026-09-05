import cv2
import numpy as np
from scipy.io import savemat
import csv


def count_object(img, range_separator, row_separator, label, to_right, num_border, save_path):

    h_ori, w_ori, c = img.shape

    label_center_x = label[:, 1] + label[:, 3] / 2

    border_margin = 40

    left_border = max(0, int(np.min(label_center_x) - border_margin))
    right_border = min(w_ori, int(np.max(label_center_x) + border_margin))

    range_separator_coord = np.concatenate(([left_border], range_separator, [right_border]))

    counter = 1

    final_pred = []

    rowsz = []

    std_count = []

    print('range_separator: ', range_separator_coord)

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

            corn_in_row = label[(range_separator_coord[i] <= label[:, 1] + label[:, 3] / 2) & (label[:, 1] + label[:, 3] / 2 <= range_separator_coord[i + 1]) & (h_sep[k] <= label[:, 2] + label[:, 4] / 2) & (label[:, 2] + label[:, 4] / 2 <= h_sep[k + 1]), 0]

            if len(corn_in_row) == 0:

                std_count.append(0)

                std.append(0)

            else:

                corn_in_row = corn_in_row + 1

                std_count.append(np.sum(corn_in_row))

                std.append(np.sum(corn_in_row))

            row_height = h_sep[k + 1] - h_sep[k]

            font_scale = max(0.8, min(1.5, row_height * 0.012))

            thickness = max(2, int(font_scale * 2))

            text = str(int(std_count[-1]))

            text_size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

            text_x = int(range_separator_coord[i + 1] - text_size[0] - 25)

            text_y = int((h_sep[k] + h_sep[k + 1]) / 2 + text_size[1] / 2)

            if 0 <= text_x < img.shape[1] and 0 <= text_y < img.shape[0]:

               img = cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness + 4, lineType=cv2.LINE_AA)

               img = cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness, lineType=cv2.LINE_AA)


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

    return img, rowsz, final_pred
