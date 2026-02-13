import cv2
import cvzone
import os
import numpy as np
import mediapipe as mp
import pickle
from collections import deque

# =========================
# CONFIG
# =========================
ACTIONS = ["rock", "paper", "scissors"]
THRESHOLD = 0.5
FRAMES_PER_ACTION = 3

MODEL_DIR = "Models"
MODELS_TRAINED_DIR = "Models_Trained"

RESOURCE_DIR = "Resources"

# =========================
# LOAD MODEL
# =========================
"""with open(os.path.join(MODEL_DIR, "best_model.pkl"), "rb") as f:
    model = pickle.load(f)"""

with open(os.path.join(MODELS_TRAINED_DIR, "best_model_DT_60P_3F_20Each.pkl"), "rb") as f:
    model = pickle.load(f)

# =========================
# MEDIAPIPE SETUP
# =========================
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, holistic):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = holistic.process(image_rgb)
    image_rgb.flags.writeable = True
    return image, results

def extract_keypoints(results):
    pose = np.array([[l.x, l.y, l.z, l.visibility]
                     for l in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[l.x, l.y, l.z]
                     for l in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    right_hand = np.array([[l.x, l.y, l.z]
                           for l in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, right_hand])

# =========================
# VIDEO & GUI SETUP
# =========================
cap = cv2.VideoCapture(0)
imgBG = cv2.imread(f"{RESOURCE_DIR}/BG.jpg")

player_box_x, player_box_y = 1427, 418
player_box_width = 2143 - player_box_x
player_box_height = 1170 - player_box_y

cv2.namedWindow("BG", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("BG", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

success, img = cap.read()
cam_h, cam_w = img.shape[:2]
scale_factor = player_box_height / cam_h
scaled_width = int(cam_w * scale_factor)
crop_start = (scaled_width - player_box_width) // 2
crop_end = crop_start + player_box_width

# =========================
# GAME STATE
# =========================
sequence = deque(maxlen=FRAMES_PER_ACTION)
current_player_action = None
last_player_action = None
ai_current_action = None

# =========================
# MAIN LOOP
# =========================
with mp_holistic.Holistic(min_detection_confidence=0.5,
                          min_tracking_confidence=0.5) as holistic:

    while True:
        success, frame = cap.read()
        if not success:
            break

        imgDisplay = imgBG.copy()

        # Resize & crop camera
        imgScaled = cv2.resize(frame, (0, 0), None, scale_factor, scale_factor)
        imgScaled = imgScaled[:, crop_start:crop_end]

        image, results = mediapipe_detection(imgScaled, holistic)
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)

        # =========================
        # PREDICTION LOGIC
        # =========================
        if len(sequence) == FRAMES_PER_ACTION:
            seq_array = np.array(sequence).flatten().reshape(1, -1)
            probs = model.predict_proba(seq_array)[0]
            pred_class = np.argmax(probs)

            if probs[pred_class] > THRESHOLD:
                current_player_action = ACTIONS[pred_class]


        # =========================
        # ACTION CHANGE DETECTED
        # =========================
        if current_player_action != last_player_action and current_player_action is not None:
            last_player_action = current_player_action
            print(f"Player changed to: {current_player_action}")

            # AI mirrors the player
            #ai_current_action = current_player_action #this line for same action as the player.
            # AI chooses the winning move against the player
            winning_move = {
                "rock": "paper",
                "paper": "scissors",
                "scissors": "rock"
            }

            ai_current_action = winning_move[current_player_action]

            print(f"AI mirrors with: {ai_current_action}")

            # Update AI image
            imgAI = cv2.imread(f"{RESOURCE_DIR}/{ai_current_action}.png", cv2.IMREAD_UNCHANGED)
            if imgAI is not None:
                imgAI = cv2.resize(imgAI, (0, 0), None, fx=1.3, fy=1.3)
                imgBG = cvzone.overlayPNG(imgBG, imgAI, (230, 510))

        # =========================
        # GUI RENDERING
        # =========================
        imgDisplay[player_box_y:player_box_y + player_box_height,
                   player_box_x:player_box_x + player_box_width] = imgScaled

        cv2.imshow("BG", imgDisplay)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()