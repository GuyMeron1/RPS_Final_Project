import cv2
import os
import mediapipe as mp

def mediapipe_detection(image, model):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = model.process(image_rgb)
    image_rgb.flags.writeable = True
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
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

def collect_videos(DATA_PATH, actions, videos_per_action, frames_per_video, fps=30, pre_roll=10):
    """
    Collect videos from webcam, display landmarks and text,
    but save only clean videos without drawings or text.
    The first 'pre_roll' frames are discarded to ensure fresh frames.
    """
    video_dir = os.path.join(DATA_PATH, "videos")
    os.makedirs(video_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)

    with mp.solutions.holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:

        for action in actions:
            action_path = os.path.join(video_dir, action)
            os.makedirs(action_path, exist_ok=True)

            existing = len(os.listdir(action_path))

            for i in range(existing + 1, existing + videos_per_action + 1):
                video_path = os.path.join(action_path, f"{i}.mp4")

                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(video_path, fourcc, fps, (640, 480))

                print(f"Recording {action} video {i}")

                # --- PRE-ROLL: discard first few frames to flush camera ---
                for _ in range(pre_roll):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    # optional: display pre-roll frames
                    frame_disp, results = mediapipe_detection(frame, holistic)
                    draw_landmarks(frame_disp, results)
                    cv2.putText(frame_disp, "GET READY...", (120, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 4, cv2.LINE_AA)
                    cv2.imshow("Recording", frame_disp)
                    cv2.waitKey(100)

                # --- RECORD ACTUAL FRAMES ---
                for frame_num in range(frames_per_video):
                    ret, frame = cap.read()
                    if not ret:
                        break

                    clean_frame = frame.copy()  # save clean frame

                    # display landmarks for feedback
                    frame_disp, results = mediapipe_detection(frame, holistic)
                    draw_landmarks(frame_disp, results)
                    cv2.putText(frame_disp, f'{action} Video {i}/{existing + videos_per_action}',
                                (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
                    cv2.imshow("Recording", frame_disp)

                    # write only clean frame
                    out.write(clean_frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                out.release()  # save video file

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    DATA_PATH = "Data"
    actions = ["Rock", "Paper", "Scissors"]

    collect_videos(
        DATA_PATH,
        actions,
        videos_per_action=5,    # number of videos per action
        frames_per_video=64,    # frames per video
        pre_roll=15             # discard first 15 frames for clean start
    )
