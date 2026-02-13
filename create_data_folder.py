import os

# Root of the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Raw data
DATA_DIR = os.path.join(BASE_DIR, "Data")
VIDEOS_DIR = os.path.join(DATA_DIR, "videos")
KEYPOINTS_DIR = os.path.join(DATA_DIR, "keypoints")
PICKLES_DIR = os.path.join(DATA_DIR, "pickles")

# Ensure directories exist
for path in [DATA_DIR, VIDEOS_DIR, KEYPOINTS_DIR, PICKLES_DIR]:
    os.makedirs(path, exist_ok=True)