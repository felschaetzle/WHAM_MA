import glob
import os

def find_images_dirs(root_dir):
    # Use glob to find all directories named 'images'
    pattern = os.path.join(root_dir, "**/images")
    img_dirs = sorted([d for d in glob.glob(pattern, recursive=True) if os.path.isdir(d) and "backup" not in d])
    return img_dirs

# Example usage
root_directory = "/mnt/hdd/emdb_dataset/"
images_directories = find_images_dirs(root_directory)

print("Found 'images' directories:")
for d in images_directories:
    print(d)