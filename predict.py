import cv2
import os
import numpy as np
import mediapipe as mp
import pickle

# SETTINGS & CONFIGURATIONS

# Colors for the probability bars (BGR format)
colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]
# Labels for the classes - must be in the same order as training
actions = ['Rock', 'Paper', 'Scissors']
# Number of frames to collect before making a prediction (must match training)
frames_per_action = 3

USE_LAST_MODEL = True # True for last model trained. False for best model ever trained.

def load_model(use_historical=True):
    if use_historical:
        # Settings to last model trained.
        directory = "Models"
        model_name = "best_model.pkl"
    else:
        # Settings to best model ever trained.
        directory = "Models_Trained"
        model_name = "best_model_DT_60P_3F_20Each.pkl"

    path = os.path.join(directory, model_name)

    if not os.path.exists(path):
        print(f"Error: {path} not found!")
        return None

    with open(path, "rb") as f:
        print(f"Loading model: {model_name} from {directory}")
        return pickle.load(f)

# Load the model to be used for predictions
model = load_model()

# PRE-PROCESSING FUNCTIONS
def mediapipe_detection(image, model):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = model.process(image_rgb)
    image_rgb.flags.writeable = True
    return image, results

def draw_landmarks(image, results):
    mp_drawing = mp.solutions.drawing_utils
    # Drawing Face, Pose, and Right Hand with custom styles
    if results.face_landmarks:
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp.solutions.holistic.FACEMESH_TESSELATION)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp.solutions.holistic.POSE_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)

def extract_keypoints(results):
    pose = np.array([[lmk.x, lmk.y, lmk.z, lmk.visibility] for lmk in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[lmk.x, lmk.y, lmk.z] for lmk in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    right_hand = np.array([[lmk.x, lmk.y, lmk.z] for lmk in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, right_hand])

def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        # Draw dynamic rectangles based on the probability score
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    return output_frame

# REAL-TIME DETECTION LOOP
sequence = [] # Buffer to store the last N frames of keypoints
sentence = [] # History of detected actions
threshold = 0.5 # Confidence threshold for display

cap = cv2.VideoCapture(0) # Open webcam

# Access MediaPipe Holistic model
with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Detection
        image, results = mediapipe_detection(frame, holistic)
        draw_landmarks(image, results)

        # Keypoint Collection
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-frames_per_action:] # Keep only the last N frames

        # Prediction Logic
        if len(sequence) == frames_per_action:
            # Flatten & Reshape to match the ML model's expected input
            seq_array = np.array(sequence).flatten().reshape(1, -1)
            res_probs = model.predict_proba(seq_array)[0] # Get probabilities for each class
            res_class = np.argmax(res_probs) # Get index of highest probability

            # Display Logic (Apply Threshold)
            if res_probs[res_class] > threshold:
                # Add to sentence only if it's a new gesture
                if len(sentence) == 0 or actions[res_class] != sentence[-1]:
                    sentence.append(actions[res_class])

            # Keep only the last 5 detected gestures on screen
            if len(sentence) > 5: sentence = sentence[-5:]

            # Visualize the probability bars
            image = prob_viz(res_probs, actions, image, colors)

        # Draw the top status bar and current "sentence"
        cv2.rectangle(image, (0,0), (640,40), (245,117,16), -1)
        cv2.putText(image, ' '.join(sentence), (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

        cv2.imshow("OpenCV Feed", image)

        # Press 'q' to exit
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()