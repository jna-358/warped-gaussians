import glob
import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("input_dir", type=str)
parser.add_argument("gt_dir", type=str)
args = parser.parse_args()


output_path = os.path.join(os.path.dirname(args.input_dir), os.path.basename(args.input_dir) + ".mp4")
fps = 10

image_paths = sorted(glob.glob(os.path.join(args.input_dir, "*.png")))
image_paths_gt = sorted(glob.glob(os.path.join(args.gt_dir, "*")))
frame = cv2.imread(image_paths[0])
height, width, _ = frame.shape

# Make circular mask
mask = np.zeros((height, width), np.uint8)
diameter = int((min(width, height)/2) * 0.9)
cv2.circle(mask, (width//2, height//2), diameter, (255, 255, 255), -1)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(output_path, fourcc, fps, (width*2, height))

for image_path, image_path_gt in tqdm(list(zip(image_paths, image_paths_gt))):
    image_gt = cv2.imread(image_path_gt)
    image_pred = cv2.imread(image_path)
    image_gt = cv2.resize(image_gt, (width, height))
    image_pred = cv2.resize(image_pred, (width, height))
    images_both = np.hstack((image_gt, image_pred))
    video.write(images_both)

video.release()