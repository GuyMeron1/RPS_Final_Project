import os
import shutil

DANGER_MODE = False

# Root of the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths of the folders
DATA_DIR = os.path.join(BASE_DIR, "Data")
VIDEOS_DIR = os.path.join(DATA_DIR, "videos")
KEYPOINTS_DIR = os.path.join(DATA_DIR, "keypoints")
PICKLES_DIR = os.path.join(DATA_DIR, "pickles")

# List of paths to delete
folders_to_delete = [KEYPOINTS_DIR, PICKLES_DIR]

print(f"DANGER MODE: {'ON - FOLDERS WILL BE DELETED' if DANGER_MODE else 'OFF - NOTHING WILL BE DELETED'}")

for folder in folders_to_delete:
    if os.path.exists(folder):
        if DANGER_MODE:
            shutil.rmtree(folder)
            print(f"Done. Deleted folder: {folder}")
        else:
            print(f"Folder '{folder}' exists and WOULD NOT be deleted (DANGER_MODE: False).")
    else:
        print(f"Folder does not exist: {folder}")

if not DANGER_MODE:
    print("\nNothing was actually deleted. Set DANGER_MODE = True to execute.")
else:
    print("\nDeletion complete!")