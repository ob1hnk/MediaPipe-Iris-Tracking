import sys
import time

import cv2
import mediapipe as mp


def main(camera_index: int = 0):
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="face_landmarker.task"),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"웹캠을 열 수 없습니다. camera_index={camera_index}", file=sys.stderr)
        return

    base_timestamp = time.time()

    iris_indices = {
        "left": [468, 469, 470, 471, 472], # 왼쪽 홍채에 해당하는 랜드마크 인덱스
        "right": [473, 474, 475, 476, 477], # 오른쪽 홍채에 해당하는 랜드마크 인덱스
    }

    with FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            success, frame = cap.read()
            if not success:
                print("프레임을 읽을 수 없습니다.", file=sys.stderr)
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            timestamp_ms = int((time.time() - base_timestamp) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.face_landmarks:
                h, w, _ = frame.shape
                landmarks = result.face_landmarks[0]

                for eye, indices in iris_indices.items():
                    coords = []
                    for idx in indices:
                        lm = landmarks[idx]
                        x_px = int(lm.x * w)
                        y_px = int(lm.y * h)
                        coords.append((x_px, y_px))
                        cv2.circle(frame, (x_px, y_px), 2, (0, 255, 0), -1)

                    if coords:
                        avg_x = sum(pt[0] for pt in coords) // len(coords)
                        avg_y = sum(pt[1] for pt in coords) // len(coords)
                        cv2.circle(frame, (avg_x, avg_y), 3, (0, 0, 255), -1)
                        cv2.putText(
                            frame,
                            f"{eye} iris: ({avg_x}, {avg_y})",
                            (avg_x + 6, avg_y - 6),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (0, 0, 255),
                            1,
                            cv2.LINE_AA,
                        )

            cv2.imshow("MediaPipe Iris Live", frame)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        try:
            cam_idx = int(sys.argv[1])
        except ValueError:
            print("카메라 인덱스는 정수여야 합니다.", file=sys.stderr)
            sys.exit(1)
    else:
        cam_idx = 0
    main(cam_idx)

