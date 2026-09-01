import cv2
import numpy as np
import os
import argparse

def undistort_image(image_path, save_path, K, dist_coeffs):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return

    h, w = image.shape[:2]
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist_coeffs, (w, h), 1, (w, h))
    undistorted_image = cv2.undistort(image, K, dist_coeffs, None, new_K)
    
    x, y, w, h = roi
    undistorted_image_cropped = undistorted_image[y:y+h, x:x+w]

    cv2.imwrite(save_path, undistorted_image_cropped)
    print(f"Undistorted and cropped image saved to {save_path}")

def process_images(image_dir, save_dir, K, dist_coeffs):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    filenames = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff'))])
    
    for filename in filenames:
        image_path = os.path.join(image_dir, filename)
        save_path = os.path.join(save_dir, filename)
        undistort_image(image_path, save_path, K, dist_coeffs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Undistort images using given intrinsic parameters and distortion coefficients")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the directory containing images to be undistorted")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the undistorted images")
    args = parser.parse_args()
    #this parameters are from grace's checkerboard that was automatically  computed witjh python
    f = 563.46403113788631
    cx = 1920
    cy = 1080
    k1 = 0.0017247862120850453

    K = np.array([
        [f, 0, cx],
        [0, f, cy],
        [0, 0, 1]
    ])
    dist_coeffs = np.array([[ 0.06693996, -0.15926691, -0.01767889, -0.00425557,  0.34429158]])

    process_images(args.image_path, args.save_path, K, dist_coeffs)


