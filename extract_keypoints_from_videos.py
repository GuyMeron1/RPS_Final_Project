import cv2
import os
import numpy as np
import mediapipe as mp
import pickle
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def mediapipe_detection(image, model):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = model.process(image_rgb)
    image_rgb.flags.writeable = True
    return results
def extract_keypoints(results):
    pose = np.array([[lmk.x, lmk.y, lmk.z, lmk.visibility] for lmk in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[lmk.x, lmk.y, lmk.z] for lmk in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    right_hand = np.array([[lmk.x, lmk.y, lmk.z] for lmk in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, right_hand])
def process_videos_to_keypoints(DATA_PATH, sequence_length=None, skip_rate=1):
    video_dir = os.path.join(DATA_PATH, "videos")
    keypoints_dir = os.path.join(DATA_PATH, "keypoints")
    os.makedirs(keypoints_dir, exist_ok=True)

    with mp.solutions.holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:

        for action in os.listdir(video_dir):
            action_path = os.path.join(video_dir, action)
            if not os.path.isdir(action_path):
                continue

            action_keypoints_path = os.path.join(keypoints_dir, action)
            os.makedirs(action_keypoints_path, exist_ok=True)

            for video_file in os.listdir(action_path):
                if not video_file.endswith(('.mp4', '.avi', '.mov')):
                    continue

                video_path = os.path.join(action_path, video_file)
                cap = cv2.VideoCapture(video_path)

                frame_idx = 0
                video_name = os.path.splitext(video_file)[0]
                video_npy_path = os.path.join(action_keypoints_path, video_name)
                os.makedirs(video_npy_path, exist_ok=True)

                frame_count = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if frame_count % skip_rate == 0:
                        results = mediapipe_detection(frame, holistic)
                        keypoints = extract_keypoints(results)
                        np.save(os.path.join(video_npy_path, f"{frame_idx}.npy"), keypoints)
                        frame_idx += 1

                    frame_count += 1
                    if sequence_length and frame_idx >= sequence_length:
                        break

                cap.release()
                print(f"Processed {action} {video_file} -> {frame_idx} frames keypoints")

    print("All videos processed!")
def save_pickles_from_keypoints(DATA_PATH, test_size=0.2, valid_size=0.1):
    keypoints_dir = os.path.join(DATA_PATH, "keypoints")

    # ✅ Explicit class order
    actions = ["Rock", "Paper", "Scissors"]
    label_map = {label: idx for idx, label in enumerate(actions)}

    # Safety check
    available_actions = set(os.listdir(keypoints_dir))
    for a in actions:
        if a not in available_actions:
            raise ValueError(f"Missing action folder: {a}")

    X, y = [], []

    for action in actions:
        action_path = os.path.join(keypoints_dir, action)
        for video_folder in sorted(os.listdir(action_path), key=lambda x: int(x)):
            video_path = os.path.join(action_path, video_folder)
            if not os.path.isdir(video_path):
                continue

            sequence = []
            for frame_file in sorted(os.listdir(video_path), key=lambda x: int(x.split('.')[0])):
                frame_path = os.path.join(video_path, frame_file)
                keypoints = np.load(frame_path)
                sequence.append(keypoints)

            X.append(sequence)
            y.append(label_map[action])
            print(f"Added {action} {video_folder} -> {len(sequence)} frames")

    X = np.array(X)
    y = to_categorical(y).astype(int)

    # Train / Test split
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # Train / Validation split
    valid_ratio = valid_size / (1 - test_size)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full, y_train_full, test_size=valid_ratio, random_state=42, stratify=y_train_full
    )

    pickles_dir = os.path.join(DATA_PATH, "pickles")
    os.makedirs(pickles_dir, exist_ok=True)

    with open(os.path.join(pickles_dir, "X_train.pkl"), "wb") as f:
        pickle.dump(X_train, f)
    with open(os.path.join(pickles_dir, "y_train.pkl"), "wb") as f:
        pickle.dump(y_train, f)
    with open(os.path.join(pickles_dir, "X_valid.pkl"), "wb") as f:
        pickle.dump(X_valid, f)
    with open(os.path.join(pickles_dir, "y_valid.pkl"), "wb") as f:
        pickle.dump(y_valid, f)
    with open(os.path.join(pickles_dir, "X_test.pkl"), "wb") as f:
        pickle.dump(X_test, f)
    with open(os.path.join(pickles_dir, "y_test.pkl"), "wb") as f:
        pickle.dump(y_test, f)

    print("Pickle files (train/valid/test) saved successfully!")


if __name__ == "__main__":
    DATA_PATH = "Data"

    process_videos_to_keypoints(DATA_PATH, skip_rate=22)
    save_pickles_from_keypoints(DATA_PATH)