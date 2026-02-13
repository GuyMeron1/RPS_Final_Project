import cv2
import os
import numpy as np
import mediapipe as mp
import pickle

colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]
actions = ['Rock', 'Paper', 'Scissors']  # must match the order used in training
frames_per_action = 3

MODEL_DIR = "Models"
MODELS_TRAINED_DIR = "Models_Trained"
"""with open(os.path.join(MODEL_DIR, "best_model.pkl"), "rb") as f:
    model = pickle.load(f)"""
with open(os.path.join(MODELS_TRAINED_DIR, "best_model_DT_60P_3F_20Each.pkl"), "rb") as f:
    model = pickle.load(f)

def mediapipe_detection(image, model):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = model.process(image_rgb)
    image_rgb.flags.writeable = True
    return image, results
def draw_landmarks(image, results):
    mp_drawing = mp.solutions.drawing_utils
    # Face
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            image, results.face_landmarks, mp.solutions.holistic.FACEMESH_TESSELATION,
            mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
        )
    # Pose
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp.solutions.holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
        )
    # Right hand
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )
def extract_keypoints(results):
    pose = np.array([[lmk.x, lmk.y, lmk.z, lmk.visibility] for lmk in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[lmk.x, lmk.y, lmk.z] for lmk in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    right_hand = np.array([[lmk.x, lmk.y, lmk.z] for lmk in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, right_hand])
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    return output_frame

# ----------------- Real-time Detection -----------------
sequence = []
sentence = []
threshold = 0.5  # minimum probability to accept prediction

cap = cv2.VideoCapture(0)

with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image, results = mediapipe_detection(frame, holistic)
        draw_landmarks(image, results)

        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-frames_per_action:]  # use last 8 frames

        if len(sequence) == frames_per_action:
            # Flatten sequence to match training
            seq_array = np.array(sequence).flatten().reshape(1, -1)
            res_probs = model.predict_proba(seq_array)[0]  # probabilities
            res_class = np.argmax(res_probs)

            # Append to sentence if threshold passed
            if res_probs[res_class] > threshold:
                if len(sentence) == 0 or actions[res_class] != sentence[-1]:
                    sentence.append(actions[res_class])

            if len(sentence) > 5:
                sentence = sentence[-5:]

            # Visualize probabilities
            image = prob_viz(res_probs, actions, image, colors)

        # Display recognized actions
        cv2.rectangle(image, (0,0), (640,40), (245,117,16), -1)
        cv2.putText(image, ' '.join(sentence), (3,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

        cv2.imshow("OpenCV Feed", image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
