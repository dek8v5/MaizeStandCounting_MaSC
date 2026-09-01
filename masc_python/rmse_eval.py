import csv
import numpy as np

############## ground truth
def read_csv(file_path, delimiter=',', skip_header=False):
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter=delimiter)
        if skip_header:
            next(reader, None)  # Skip the header row
        return [row for row in reader]


txt_name = '/data/e/stand_counts/MATLAB/ground_truth/gt_fix/19r_30_400-405.csv'
ground_truth_data = read_csv(txt_name, skip_header=True)


rowss = [row[0] for row in ground_truth_data]
row_int = np.array([int(row[-3:]) for row in rowss])

labels = np.array([[row_int[i], int(ground_truth_data[i][1]) - int(ground_truth_data[i][3])] for i in range(len(rowss))])
label_sorted = labels[labels[:, 0].argsort()]


output_path = '/data/e/stand_counts/MATLAB/ground_truth/gt_final/19r_30_gt_final.txt'
with open(output_path, 'w', newline='') as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerows(label_sorted)

#### predictionssss
txt_name_pred = '/data/e/stand_counts/MATLAB/IMAGES/19r_30/pred_34_32.txt'
pred_data = read_csv(txt_name_pred, delimiter=',')


pred_stand_count = np.array([[int(row[0]), int(row[1])] for row in pred_data])


def rmse_count(pred, actual):
    return np.sqrt(np.mean((pred - actual) ** 2))


rmse = rmse_count(pred_stand_count, label_sorted[:, 1:])
print(f'RMSE: {rmse}')


evaluation_path = '/data/e/stand_counts/MATLAB/IMAGES/19r_30/rmse_result.txt'
with open(evaluation_path, 'w') as f:
    f.write(f'RMSE: {rmse}\n')
