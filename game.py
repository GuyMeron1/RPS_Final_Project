import cv2
import cvzone
import os
import numpy as np
import mediapipe as mp
import pickle
import time
from collections import deque

# Constants and Configuration
ACTIONS = ["rock", "paper", "scissors"] # possible gestures
THRESHOLD = 0.5 # minimum confidence for a prediction
FRAMES_PER_ACTION = 3 # frames needed for a stable gesture
RESOURCE_DIR = "Resources" # directory for UI elements

USE_LAST_MODEL = True # True for last model trained. False for best model ever trained.

def load_model(use_historical=True):
    """
    Loads the model based on the USE_HISTORICAL_MODEL flag.
    """
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

# Mediapipe Setup
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, holistic):
    """Converts image to RGB and processes it with Mediapipe"""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False # Optimization
    results = holistic.process(image_rgb)
    image_rgb.flags.writeable = True
    return image, results

def extract_keypoints(results):
    """Flattens pose, face, and hand landmarks into a single numpy array"""
    pose = np.array([[l.x, l.y, l.z, l.visibility] for l in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[l.x, l.y, l.z] for l in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    right_hand = np.array([[l.x, l.y, l.z] for l in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, face, right_hand])

# VIDEO & GUI SETUP
cap = cv2.VideoCapture(0)
imgBG_original = cv2.imread(f"{RESOURCE_DIR}/BG.jpg")
imgBG = imgBG_original.copy()

# Dimensions for the camera feed inside the UI
player_box_x, player_box_y = 1427, 418
player_box_width = 2143 - player_box_x
player_box_height = 1170 - player_box_y

# Set window to fullscreen mode
cv2.namedWindow("BG", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("BG", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Scaling calculations for the camera feed
success, img = cap.read()
cam_h, cam_w = img.shape[:2]
scale_factor = player_box_height / cam_h
scaled_width = int(cam_w * scale_factor)
crop_start = (scaled_width - player_box_width) // 2
crop_end = crop_start + player_box_width

# GAME STATE
sequence = deque(maxlen=FRAMES_PER_ACTION)
current_player_action = None
last_player_action = None
ai_current_action = None
player_score = 0
ai_score = 0

# COOLDOWN SETUP
last_score_time = 0
COOLDOWN_SECONDS = 0.2 # Minimum time between changes of score

# HAND DETECTION TIMEOUT
NO_HAND_CONFIRM_FRAMES = 5
no_hand_counter = 0

# MAIN LOOP
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while True:
        success, frame = cap.read()
        if not success:
            break

        imgDisplay = imgBG.copy()
        # Scale and crop camera frame to fit the UI box
        imgScaled = cv2.resize(frame, (0, 0), None, scale_factor, scale_factor)
        imgScaled = imgScaled[:, crop_start:crop_end]

        image, results = mediapipe_detection(imgScaled, holistic)

        # CHECK IF HAND IS IN FRAME
        hand_in_frame = results.right_hand_landmarks is not None

        keypoints = extract_keypoints(results)
        sequence.append(keypoints)

        # PREDICTION LOGIC
        if hand_in_frame:
            no_hand_counter = 0

            cv2.putText(imgDisplay, "Play!", (1005, 760), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 255), 10)

            if len(sequence) == FRAMES_PER_ACTION:
                seq_array = np.array(sequence).flatten().reshape(1, -1)
                probs = model.predict_proba(seq_array)[0]
                pred_class = np.argmax(probs)

                # Only act if model is confident
                if probs[pred_class] > THRESHOLD:
                    current_player_action = ACTIONS[pred_class]

                    # ACTION CHANGE DETECTED
                    if current_player_action != last_player_action and current_player_action is not None:

                        last_player_action = current_player_action
                        print(f"Player changed to: {current_player_action}")

                        # AI determines winning counter-move
                        winning_move = {
                            "rock": "paper",
                            "paper": "scissors",
                            "scissors": "rock"
                        }

                        ai_current_action = winning_move[current_player_action]
                        print(f"AI mirrors with: {ai_current_action}")

                        # Load and display the AI's move
                        imgAI = cv2.imread(f"{RESOURCE_DIR}/{ai_current_action}.png", cv2.IMREAD_UNCHANGED)
                        if imgAI is not None:
                            imgAI = cv2.resize(imgAI, (0, 0), None, fx=1.3, fy=1.3)
                            imgBG = cvzone.overlayPNG(imgBG.copy(), imgAI, (230, 510))

                        # Update AI Score with cooldown check
                        current_time = time.time()
                        if (current_time - last_score_time) > COOLDOWN_SECONDS:
                            ai_score += 1
                            last_score_time = current_time
                            print(f"Score updated! AI Score: {ai_score}")
        else:
            # Handle hand removal from frame
            no_hand_counter += 1
            if no_hand_counter >= NO_HAND_CONFIRM_FRAMES:
                current_player_action = None

                # RESET MEMORY AND VISUAL RESET
                imgBG = imgBG_original.copy()
                last_player_action = None

                cv2.putText(imgDisplay, "No Hand", (1010, 710), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 5)
                cv2.putText(imgDisplay, "Detected", (1010, 780), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 5)

        # GUI RENDERING
        cv2.putText(imgDisplay, f"{ai_score:02d}", (715, 375), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 7)
        cv2.putText(imgDisplay, f"{player_score:02d}", (1980, 375), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 7)

        # Place the camera feed into the background image
        imgDisplay[player_box_y:player_box_y + player_box_height,
        player_box_x:player_box_x + player_box_width] = imgScaled

        cv2.imshow("BG", imgDisplay)

        # INPUT HANDLING
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'): # Reset game
            ai_score = 0
            player_score = 0
            imgBG = imgBG_original.copy()
            last_player_action = None
            print("Reset Scores!")
        elif key == 27: # Exit on ESC
            break

cap.release()
cv2.destroyAllWindows()