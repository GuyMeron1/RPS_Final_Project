import cv2
import os
import mediapipe as mp

# Process image for MediaPipe detection
def mediapipe_detection(image, model):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = model.process(image_rgb)
    image_rgb.flags.writeable = True
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    return image, results

# Draw visual feedback for face, pose, and hands
def draw_landmarks(image, results):
    mp_drawing = mp.solutions.drawing_utils
    # Face mesh
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            image, results.face_landmarks, mp.solutions.holistic.FACEMESH_TESSELATION,
            mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
        )
    # Pose skeleton
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp.solutions.holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
        )
    # Right hand connections
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )

# Core function to record videos for each action
def collect_videos(DATA_PATH, actions, videos_per_action, frames_per_video=64, fps=30, pre_roll=15):
    video_dir = os.path.join(DATA_PATH, "videos")
    os.makedirs(video_dir, exist_ok=True)

    cap = cv2.VideoCapture(0) # Open webcam

    # Initialize MediaPipe Holistic model
    with mp.solutions.holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:

        for action in actions:
            action_path = os.path.join(video_dir, action)
            os.makedirs(action_path, exist_ok=True)

            # Detect existing files to avoid overwriting
            existing = len(os.listdir(action_path))

            for i in range(existing + 1, existing + videos_per_action + 1):
                video_path = os.path.join(action_path, f"{i}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Video codec
                out = cv2.VideoWriter(video_path, fourcc, fps, (640, 480))

                print(f"Recording {action} video {i}")

                # Pre-roll phase: allow user to prepare and stabilize camera
                for _ in range(pre_roll):
                    ret, frame = cap.read()
                    if not ret: break
                    frame_disp, results = mediapipe_detection(frame, holistic)
                    draw_landmarks(frame_disp, results)
                    cv2.putText(frame_disp, "GET READY...", (120, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 4, cv2.LINE_AA)
                    cv2.imshow("Recording", frame_disp)
                    cv2.waitKey(100)

                # Capture recording phase
                for frame_num in range(frames_per_video):
                    ret, frame = cap.read()
                    if not ret: break

                    # Store clean frame (no landmarks) for training
                    clean_frame = frame.copy()

                    # Provide real-time visual feedback
                    frame_disp, results = mediapipe_detection(frame, holistic)
                    draw_landmarks(frame_disp, results)
                    cv2.putText(frame_disp, f'{action} Video {i}/{existing + videos_per_action}',(15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
                    cv2.imshow("Recording", frame_disp)

                    out.write(clean_frame) # Save frame to file

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                out.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    DATA_PATH = "Data"
    actions = ["Rock", "Paper", "Scissors"]
    collect_videos(DATA_PATH, actions, 1)