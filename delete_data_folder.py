import os
import shutil

# Root of the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths of the folders to delete
DATA_DIR = os.path.join(BASE_DIR, "Data")
VIDEOS_DIR = os.path.join(DATA_DIR, "videos")
KEYPOINTS_DIR = os.path.join(DATA_DIR, "keypoints")
PICKLES_DIR = os.path.join(DATA_DIR, "pickles")

# List of paths to delete
#folders_to_delete = [VIDEOS_DIR, KEYPOINTS_DIR, PICKLES_DIR, DATA_DIR]
folders_to_delete = [KEYPOINTS_DIR]

for folder in folders_to_delete:
    if os.path.exists(folder):
        shutil.rmtree(folder)
        print(f"Deleted folder: {folder}")
    else:
        print(f"Folder does not exist: {folder}")

print("Deletion complete!")