import cv2
import os
import numpy as np
import mediapipe as mp
import pickle
from sklearn.model_selection import train_test_split


# Process a single frame and return MediaPipe holistic results
def mediapipe_detection(image, model):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = model.process(image_rgb)
    image_rgb.flags.writeable = True
    return results

# Flatten MediaPipe results into a single NumPy array
def extract_keypoints(results):
    # Extract Pose
    pose = np.array([[lmk.x, lmk.y, lmk.z, lmk.visibility] for lmk in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    # Extract Face
    face = np.array([[lmk.x, lmk.y, lmk.z] for lmk in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    # Extract Right Hand
    right_hand = np.array([[lmk.x, lmk.y, lmk.z] for lmk in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros( 21 * 3)

    # Concatenate all into one flat array for the model
    return np.concatenate([pose, face, right_hand])

# Iterate through videos and save landmarks as .npy files
def process_videos_to_keypoints(DATA_PATH, sequence_length=None, skip_rate=1):
    # Set up input (videos) and output (keypoints) directories
    video_dir = os.path.join(DATA_PATH, "videos")
    keypoints_dir = os.path.join(DATA_PATH, "keypoints")
    os.makedirs(keypoints_dir, exist_ok=True)

    # Initialize MediaPipe Holistic model
    with mp.solutions.holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
    ) as holistic:

        # Loop through each action folder (Rock, Paper, Scissors)
        for action in os.listdir(video_dir):
            action_path = os.path.join(video_dir, action)
            if not os.path.isdir(action_path): continue  # Skip if not a directory

            # Create a corresponding folder in the keypoints directory
            action_keypoints_path = os.path.join(keypoints_dir, action)
            os.makedirs(action_keypoints_path, exist_ok=True)

            # Process every video file within the action folder
            for video_file in os.listdir(action_path):
                # Filter for video formats only
                if not video_file.endswith(('.mp4', '.avi', '.mov')): continue

                video_path = os.path.join(action_path, video_file)
                cap = cv2.VideoCapture(video_path)  # Open the video file

                frame_idx = 0  # Counter for saved frames
                video_name = os.path.splitext(video_file)[0]  # Get filename without extension

                # Create a specific folder for this video's frames
                video_npy_path = os.path.join(action_keypoints_path, video_name)
                os.makedirs(video_npy_path, exist_ok=True)

                frame_count = 0  # Counter for total frames read
                while True:
                    ret, frame = cap.read()  # Read the next frame
                    if not ret: break  # Stop if video ends

                    # Reduces data size and focuses on key movements.
                    if frame_count % skip_rate == 0:
                        # Run MediaPipe to find landmarks
                        results = mediapipe_detection(frame, holistic)

                        # Convert landmarks to a flat NumPy array (numbers)
                        keypoints = extract_keypoints(results)

                        # Save the array as a .npy file
                        np.save(os.path.join(video_npy_path, f"{frame_idx}.npy"), keypoints)
                        frame_idx += 1

                    frame_count += 1

                cap.release()  # Close the video file to free memory
                print(f"Processed {action} {video_file} -> {frame_idx} frames keypoints")

    print("All videos processed!")

# Compile individual frame files into Train/Valid/Test datasets
def export_to_pickles(DATA_PATH, test_size=0.2, valid_size=0.1):
    # Set the path to the folder containing the extracted .npy files
    keypoints_dir = os.path.join(DATA_PATH, "keypoints")

    actions = ["Rock", "Paper", "Scissors"]
    label_map = {label: idx for idx, label in enumerate(actions)}

    # Lists to store our data (X) and labels (y)
    X, y = [], []

    # DATA COLLECTION
    for action in actions:
        action_path = os.path.join(keypoints_dir, action)

        # Sort video folders numerically to keep data organized
        for video_folder in sorted(os.listdir(action_path), key=lambda x: int(x)):
            video_path = os.path.join(action_path, video_folder)
            if not os.path.isdir(video_path): continue

            sequence = []  # Hold all frames for one single video clip

            # Loop through each .npy file
            for frame_file in sorted(os.listdir(video_path), key=lambda x: int(x.split('.')[0])):
                frame_path = os.path.join(video_path, frame_file)
                keypoints = np.load(frame_path)  # Load the numeric coordinates
                sequence.append(keypoints)  # Add this frame's data to the video sequence

            # X gets the full sequence, y gets the action label
            X.append(sequence)
            y.append(label_map[action])
            print(f"Added {action} {video_folder} -> {len(sequence)} frames")

    # Convert lists to NumPy arrays
    X = np.array(X)
    y = np.array(y)

    # DATA SPLITTING

    # First split: Separate 'Test' (20%) from everything else.
    X_train_full, X_test, y_train_full, y_test = train_test_split( X, y, test_size=test_size, random_state=42, stratify=y)

    # Second split: Take the remaining 80% and split it into Train and Validation.
    valid_ratio = valid_size / (1 - test_size)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, test_size=valid_ratio, random_state=42, stratify=y_train_full)

    # SAVING TO PICKLE
    pickles_dir = os.path.join(DATA_PATH, "pickles")
    os.makedirs(pickles_dir, exist_ok=True)

    pickle_files = {
        "X_train.pkl": X_train, "y_train.pkl": y_train,
        "X_valid.pkl": X_valid, "y_valid.pkl": y_valid,
        "X_test.pkl": X_test, "y_test.pkl": y_test
    }

    # Save each object to a separate file
    for filename, data in pickle_files.items():
        with open(os.path.join(pickles_dir, filename), "wb") as f:
            pickle.dump(data, f)

    print("Pickle files (train/valid/test) saved successfully!")


if __name__ == "__main__":
    DATA_PATH = "Data"
    process_videos_to_keypoints(DATA_PATH, skip_rate=22)  # Extract data every 22 frames
    export_to_pickles(DATA_PATH)  # Prepare dataset for training