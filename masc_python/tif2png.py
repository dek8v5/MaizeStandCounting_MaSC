#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
tif2png: converted tif image from webODM to png with 0 as background and
rotated the image to vertical row orientation with radon transform
   input img (tif)
   output img (png)
'''

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
from skimage.transform import radon
from scipy.signal import find_peaks



def radon_transform(img, theta_range):
    sinogram = []
    center = (img.shape[1] // 2, img.shape[0] // 2)
    for theta in theta_range:
        print(theta)
        M = cv2.getRotationMatrix2D(center, theta, 1.0)
        rotated_img = cv2.warpAffine(img.astype(np.float32), M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        print('after image is  rotated')
        projection = np.sum(rotated_img, axis=0)
        sinogram.append(projection)
    return np.array(sinogram)


#def rotate_image(img, angle):
#    center = (img.shape[1] // 2, img.shape[0] // 2)
#    M = cv2.getRotationMatrix2D(center, angle, 1.0)
#    rotated_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
#    return rotated_img



def rotation_matrix(img, angle):
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos = abs(M[0, 0])
    sin = abs(M[0, 1])

    new_w = int(np.ceil(h * sin + w * cos))
    new_h = int(np.ceil(h * cos + w * sin))

    M[0, 2] += new_w / 2.0 - center[0]
    M[1, 2] += new_h / 2.0 - center[1]

    return M, new_w, new_h


def rotate_image(img, angle):
    M, new_w, new_h = rotation_matrix(img, angle)

    rotated_img = cv2.warpAffine(
        img,
        M,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    return rotated_img


def label_rotation(img, labels, angle):
    M, new_w, new_h = rotation_matrix(img, angle)

    rotated_labels = labels.copy().astype(float)

    for i in range(len(labels)):
        x = labels[i, 1]
        y = labels[i, 2]
        bw = labels[i, 3]
        bh = labels[i, 4]

        corners = np.array([
            [x,      y,      1.0],
            [x + bw, y,      1.0],
            [x + bw, y + bh, 1.0],
            [x,      y + bh, 1.0]
        ])

        rotated = corners @ M.T

        x_min = rotated[:, 0].min()
        y_min = rotated[:, 1].min()
        x_max = rotated[:, 0].max()
        y_max = rotated[:, 1].max()

        rotated_labels[i, 1] = x_min
        rotated_labels[i, 2] = y_min
        rotated_labels[i, 3] = x_max - x_min
        rotated_labels[i, 4] = y_max - y_min

    return rotated_labels


def circular_mask(image):
    rows, cols = image.shape
    center = (int(cols / 2), int(rows / 2))
    radius = min(center[0], center[1], cols - center[0], rows - center[1])
    Y, X = np.ogrid[:rows, :cols]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    mask = dist_from_center <= radius

    masked_image = np.zeros_like(image)
    masked_image[mask] = image[mask]
    return masked_image


def tif2png(img_path):
    
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    img_rgb = img[:, :, :3]
    alpha = img[:, :, 3]

    
    img_resized = cv2.resize(img_rgb, (img_rgb.shape[1], img_rgb.shape[0]))
    alpha_resized = cv2.resize(alpha, (alpha.shape[1], alpha.shape[0]))

    img_alpha = img_resized * (alpha_resized[:, :, None] / 255.0)
    img_bw = 2 * img_alpha[:, :, 1] - img_alpha[:, :, 0] - img_alpha[:, :, 2]

    _, th = cv2.threshold(img_bw, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_bw = img_bw > th

    
    theta = np.arange(1, 181)
    radon_image = radon_transform(img_bw, theta)

    
    variance = np.var(radon_image, axis=0)
    max_idx = np.argmax(variance)
    angl = theta[max_idx]

    plt.figure()
    plt.plot(theta, variance)
    plt.plot(theta[max_idx], variance[max_idx], 'rx')
    plt.text(theta[max_idx] + 5, variance[max_idx] - 5, str(max_idx))
    plt.title('Radon transform variance 0-180 degrees')
    plt.show()

    
    angl = 360 - angl
    img_bw_rot = rotate_image(img_bw, angl)
    img_rot = rotate_image(img_alpha, angl)
    alpha_rot = rotate_image(alpha_resized, angl)

    
    rows, cols = np.where(alpha_rot != 0)
    min_row, max_row = np.min(rows), np.max(rows)
    min_col, max_col = np.min(cols), np.max(cols)
    img_cropped = img_rot[min_row:max_row, min_col:max_col]

    img_cropped_uint8 = (img_cropped * 255).astype(np.uint8)

    plt.figure()
    plt.imshow(cv2.cvtColor(img_cropped_uint8, cv2.COLOR_BGR2RGB))
    plt.show()

    return img_cropped_uint8



def rot_radon(img, target_max_dim=1200):
    if img is None:
        raise ValueError("rot_radon received an empty image.")

    print("Original image shape:", img.shape)

    h, w = img.shape[:2]

    if max(h, w) > target_max_dim:
        scale = target_max_dim / float(max(h, w))
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        img_small = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        scale = 1.0
        img_small = img.copy()

    print("Radon image shape:", img_small.shape)
    print("Radon scale factor:", scale)

    img_float = img_small.astype(np.float32)

    B = img_float[:, :, 0]
    G = img_float[:, :, 1]
    R = img_float[:, :, 2]

    img_bw = 2.0 * G - R - B
    img_bw = cv2.normalize(img_bw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    otsu_threshold, binary = cv2.threshold(img_bw, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    print("Otsu threshold:", otsu_threshold)

    img_bw = binary > 0
    img_mask = circular_mask(img_bw)

    theta = np.arange(0.0, 180.0, 1.0)

    print("Running Radon transform with", len(theta), "angles...")

    sinogram = radon(img_mask.astype(np.float32), theta=theta, circle=True)

    print("Radon transform finished.")

    variance = np.var(sinogram, axis=0)

    first_idx = np.argmax(variance)
    first_angle = theta[first_idx]

    orthogonal_angle = (first_angle + 90.0) % 180.0
    angle_diff = np.abs(((theta - orthogonal_angle + 90.0) % 180.0) - 90.0)
    orthogonal_indices = np.where(angle_diff <= 10.0)[0]

    second_idx = orthogonal_indices[np.argmax(variance[orthogonal_indices])]

    candidate_indices = [first_idx, second_idx]

    best_idx = None
    best_score = -1

    print("Radon candidate orientations:")

    for idx in candidate_indices:
        projection = sinogram[:, idx]

        window = max(3, len(projection) // 100)
        smooth_projection = np.convolve(projection, np.ones(window) / window, mode='same')

        signal_range = np.percentile(smooth_projection, 95) - np.percentile(smooth_projection, 5)
        prominence = max(signal_range * 0.10, 1e-6)
        min_distance = max(1, len(projection) // 100)

        peaks, properties = find_peaks(smooth_projection, prominence=prominence, distance=min_distance)

        if len(peaks) > 0:
            score = len(peaks)
        else:
            score = 0

        print("Angle:", theta[idx], "degrees | repeated peaks:", score)

        if score > best_score:
            best_score = score
            best_idx = idx

    detected_angle = theta[best_idx]

    rotation_angle = 90.0 - detected_angle

    if rotation_angle > 90.0:
        rotation_angle -= 180.0
    elif rotation_angle < -90.0:
        rotation_angle += 180.0

    print("Selected Radon angle:", detected_angle, "degrees")
    print("Rotation applied:", rotation_angle, "degrees")

    img_rot = rotate_image(img, rotation_angle)

    print("Rotated full image shape:", img_rot.shape)

    return img_rot, rotation_angle




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-image_path', type=str, nargs='+', help="paths to one or more images or image directories")
    parser.add_argument("-save_path", dest='save_path', default="RESULTS/global_"+datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), type=str, help="path to save result")
    args = parser.parse_args()

    image_path = args.image_path

    img = cv2.imread(image_path[0])

    rot_radon(img)
