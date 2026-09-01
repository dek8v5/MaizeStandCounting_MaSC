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


def rotate_image(img, angle):
    center = (img.shape[1] // 2, img.shape[0] // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return rotated_img


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


def rot_radon(img):
    
    img_bw = 2 * img[:, :, 1] - img[:, :, 0] - img[:, :, 2]

    _, th = cv2.threshold(img_bw, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_bw = img_bw > th
    print(th)

    img_mask = circular_mask(img_bw)
    
    plt.figure()
    plt.imshow(img_mask)
    plt.title('masked Image')
    plt.axis('off')
    plt.savefig('/data/e/dmc/code/figure.png') #just temp to save

    theta = np.linspace(0., 180., max(img_bw.shape), endpoint=False)
    sinogram = radon(img_mask, theta)
    print('after radon')
    plt.figure()
    plt.plot(theta, np.sum(sinogram, axis=0))
    plt.title('Radon transform sum')
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Sum')
    plt.show()

    variance = np.var(sinogram, axis=0)
    max_angl = np.argmax(variance)
    angl = theta[max_angl]

    plt.figure()
    plt.plot(theta, variance)
    plt.plot(angl, variance[max_angl], 'rx')
    plt.text(angl + 5, variance[max_angl] - 5, str(angl))
    plt.title('Radon transform variance 0-180°')
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Variance')
    plt.show()

    angl = 360 - angl
    ExG_rot = rotate_image(img_bw, angl)
    img_rot = rotate_image(img, angl)

    img_rot = (img_rot * 255).astype(np.uint8)

    plt.figure()
    plt.imshow(img_rot)
    plt.title('Rotated Image')
    plt.axis('off')
    plt.show()

    return img_rot

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-image_path', type=str, nargs='+', help="paths to one or more images or image directories")
    parser.add_argument("-save_path", dest='save_path', default="RESULTS/global_"+datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), type=str, help="path to save result")
    args = parser.parse_args()

    image_path = args.image_path

    img = cv2.imread(image_path[0])

    rot_radon(img)
