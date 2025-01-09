import argparse
import os
import glob
import cv2
import tqdm

FPS = 30


if __name__ == "__main__":
    # Parse directory
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=str)
    args = parser.parse_args()

    # Get all images in the directory
    images = glob.glob(os.path.join(args.directory, "*.png"))
    images = sorted(images)

    # Write video
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    video = None # cv2.VideoWriter(os.path.join(args.directory, "..", "trajectory.mp4"), fourcc, FPS, (1920, 1080))

    for image in tqdm.tqdm(images):
        frame = cv2.imread(image)

        if video is None:
            height, width, _ = frame.shape
            video = cv2.VideoWriter(os.path.join(args.directory, "..", "trajectory.mp4"), fourcc, FPS, (width, height))
        video.write(frame)

    video.release()
    