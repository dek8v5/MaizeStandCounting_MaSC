import csv
import numpy as np
import argparse
from datetime import datetime
from matplotlib import pyplot as plt

def read_csv(file_path):
    actual_counts = []
    predicted_counts = []
    raw_counts = []
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            actual_counts.append(float(row[1]))
            predicted_counts.append(float(row[2]))
            raw_counts.append(float(row[3]))
    print(np.array(actual_counts), np.array(predicted_counts), np.array(raw_counts))
    return np.array(actual_counts), np.array(predicted_counts), np.array(raw_counts)

def compute_metrics(actual_counts, predicted_counts):
    mae = np.mean(np.abs(predicted_counts - actual_counts))
    mse = np.mean((predicted_counts - actual_counts) ** 2)
    rmse = np.sqrt(mse)
    ss_total = np.sum((actual_counts - np.mean(actual_counts)) ** 2)
    ss_res = np.sum((actual_counts - predicted_counts) ** 2)
    r_squared = 1 - (ss_res / ss_total)
    
    return mae, mse, rmse, r_squared

def plot_r2(actual_counts, predicted_counts, r_squared, save_path, fname):
    plt.figure(figsize=(8, 6))
    plt.scatter(actual_counts, actual_counts, color='green', alpha=0.5, label='Actual Counts')
    plt.scatter(actual_counts, predicted_counts, color='blue', alpha=0.5, label=f'{fname} Counts')
    plt.plot([actual_counts.min(), actual_counts.max()], [actual_counts.min(), actual_counts.max()], 'k--', lw=2)
    
    slope, intercept = np.polyfit(actual_counts, predicted_counts, 1)
    regression_line = slope * actual_counts + intercept
    plt.plot(actual_counts, regression_line, color='red', label='Regression Line')

    plt.xlim(0, actual_counts.max()+1)
    plt.ylim(0, actual_counts.max()+1)
    plt.xlabel('actual', fontsize=20)
    plt.ylabel(f'predicted', fontsize=20)
    plt.title(f'Stand Counts from {fname}' , fontsize=20)

    plt.text(0.1 * actual_counts.max(), 0.6 * actual_counts.max(), f'$R^2 = {r_squared:.3f}$', fontsize=30, color='red', weight='bold')

    plt.grid(True)
    plt.savefig(f'{save_path}/{fname}.png')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate stand count detection.')
    parser.add_argument('-csv_file', type=str, help='Path to the CSV file')
    parser.add_argument('-save_path', dest='save_path', default="results/mosaic_"+datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), type=str, help="path to save result")
    parser.add_argument('-fname', '--fname', dest='fname', default='raw', help='filename')
    args = parser.parse_args()
		
    actual_counts, mosaic_counts, raw_counts = read_csv(args.csv_file)
    mae, mse, rmse, r_squared = compute_metrics(actual_counts, raw_counts)

    plot_r2(actual_counts, raw_counts, r_squared, args.save_path, args.fname)
    
    print(f"MAE: {mae}")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"R-squared: {r_squared}")
    
