import cv2
import os

def create_video_from_frames(frames_dir, output_file, fps=30):
    """
    Creates a .mov video file from a directory of frames.

    Args:
        frames_dir (str): Path to the directory containing frames.
        output_file (str): Path to the output .mov file.
        fps (int): Frames per second for the output video.

    Raises:
        ValueError: If no valid frames are found in the directory.
    """
    # Get a sorted list of all image files in the directory
    frame_files = sorted(
        [os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    )

    frame_files = frame_files[:250]
    if not frame_files:
        raise ValueError("No valid image frames found in the directory!")

    # Read the first frame to get video dimensions
    first_frame = cv2.imread(frame_files[0])
    height, width, _ = first_frame.shape

    # Define the video codec and create the VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mov files
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    # Write frames to the video
    for frame_file in frame_files:
        frame = cv2.imread(frame_file)
        if frame is None:
            print(f"Warning: Could not read {frame_file}, skipping.")
            continue
        video_writer.write(frame)

    # Release the VideoWriter object
    video_writer.release()
    print(f"Video saved to {output_file}")

if __name__ == "__main__":
    # Configuration: Set your directory and output file
    frames_directory = "/mnt/hdd/emdb_dataset/P5/40_indoor_walk_big_circle/images"  # Replace with the path to your frames directory
    output_mov_file = "/mnt/hdd/emdb_dataset/P5/40_indoor_walk_big_circle/raw_shorter.mov"  # Replace with the desired output .mov file path
    frames_per_second = 30  # Adjust as needed

    create_video_from_frames(frames_directory, output_mov_file, fps=frames_per_second)
