import os

# Base project path for portability
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data and output storage paths
DATA_DIR = os.path.join(BASE_DIR, "Data")
VIDEOS_DIR = os.path.join(DATA_DIR, "videos")
KEYPOINTS_DIR = os.path.join(DATA_DIR, "keypoints")
PICKLES_DIR = os.path.join(DATA_DIR, "pickles")
REPORTS_DIR = os.path.join(BASE_DIR, "Reports")
MODELS_DIR = os.path.join(BASE_DIR, "Models")

# Initialize directory structure
for path in [DATA_DIR, VIDEOS_DIR, KEYPOINTS_DIR, PICKLES_DIR, REPORTS_DIR, MODELS_DIR]:
    os.makedirs(path, exist_ok=True)