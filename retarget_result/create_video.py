import cv2
import os
import numpy as np
import glob
from tqdm import tqdm

# Input directory containing PNG frames
input_dir = "/home/huluwulu/Projects/motion_imitation/retarget_motion/new_data/data/full_seq/shiba/j3d"
# Output directory for the video
output_dir = "/home/huluwulu/Projects/motion_imitation/retarget_result"
output_filename = "shiba_j3d_sequence.mp4"
output_path = os.path.join(output_dir, output_filename)

# Parameters
fps = 24  # Frames per second

def create_video_from_frames():
    # Get all PNG files
    png_files = glob.glob(os.path.join(input_dir, "*.png"))
    
    if not png_files:
        print(f"No PNG files found in {input_dir}")
        return
    
    # Sort files numerically
    png_files.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))
    
    print(f"Found {len(png_files)} PNG files")
    
    # Read the first image to get dimensions
    img = cv2.imread(png_files[0])
    height, width, channels = img.shape
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process each frame
    print("Creating video...")
    for png_file in tqdm(png_files):
        frame = cv2.imread(png_file)
        if frame is not None:
            video_writer.write(frame)
    
    # Release the video writer
    video_writer.release()
    print(f"Video saved to {output_path}")

if __name__ == "__main__":
    create_video_from_frames() 