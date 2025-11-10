import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import cv2
import mediapipe as mp
import numpy as np


CALIBRATION_POINTS: List[Tuple[str, Tuple[float, float]]] = [
    ("center", (0.5, 0.5)),
    ("left", (0.2, 0.5)),
    ("right", (0.8, 0.5)),
    ("up", (0.5, 0.25)),
    ("down", (0.5, 0.75)),
]

PREP_SECONDS = 1.5
COLLECT_SECONDS = 2.0
MIRROR_PREVIEW = True

IRIS_INDICES = {
    "left": [468, 469, 470, 471, 472],
    "right": [473, 474, 475, 476, 477],
}


@dataclass
class CalibrationSample:
    name: str
    target: Tuple[float, float]
    samples: List[Tuple[float, float]] = field(default_factory=list)

    def add_sample(self, x: float, y: float) -> None:
        self.samples.append((x, y))

    @property
    def mean(self) -> Tuple[float, float]:
        if not self.samples:
            return 0.0, 0.0
        xs, ys = zip(*self.samples)
        return float(np.mean(xs)), float(np.mean(ys))


def compute_iris_center(landmarks):
    coords = []
    for eye_indices in IRIS_INDICES.values():
        eye_coords = []
        for idx in eye_indices:
            if idx >= len(landmarks):
                continue
            lm = landmarks[idx]
            eye_coords.append((lm.x, lm.y))
        if eye_coords:
            xs, ys = zip(*eye_coords)
            coords.append((float(np.mean(xs)), float(np.mean(ys))))
    if not coords:
        return None
    xs, ys = zip(*coords)
    return float(np.mean(xs)), float(np.mean(ys))


def solve_affine_transform(samples: List[CalibrationSample]) -> Dict[str, List[float]]:
    inputs = []
    targets_x = []
    targets_y = []
    for sample in samples:
        mean_x, mean_y = sample.mean
        inputs.append([mean_x, mean_y, 1.0])
        targets_x.append(sample.target[0])
        targets_y.append(sample.target[1])

    A = np.array(inputs, dtype=np.float64)
    target_x_arr = np.array(targets_x, dtype=np.float64)
    target_y_arr = np.array(targets_y, dtype=np.float64)

    coeff_x, *_ = np.linalg.lstsq(A, target_x_arr, rcond=None)
    coeff_y, *_ = np.linalg.lstsq(A, target_y_arr, rcond=None)

    transform = {
        "matrix": [
            [coeff_x[0], coeff_x[1]],
            [coeff_y[0], coeff_y[1]],
        ],
        "bias": [coeff_x[2], coeff_y[2]],
    }
    return transform


def run_calibration(camera_index: int, output_path: str) -> None:
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

    samples = [CalibrationSample(name, target) for name, target in CALIBRATION_POINTS]
    point_index = 0
    prep_start = time.time()
    collecting = False

    with FaceLandmarker.create_from_options(options) as landmarker:
        while point_index < len(samples):
            success, frame = cap.read()
            if not success:
                print("프레임을 읽을 수 없습니다.", file=sys.stderr)
                break

            if MIRROR_PREVIEW:
                frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            timestamp_ms = int(time.time() * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            sample = samples[point_index]
            h, w, _ = frame.shape
            target_px = (int(sample.target[0] * w), int(sample.target[1] * h))

            now = time.time()
            elapsed = now - prep_start

            if not collecting and elapsed >= PREP_SECONDS:
                collecting = True
                collect_start = now
                elapsed = 0.0

            if collecting:
                collect_elapsed = now - collect_start
                if collect_elapsed <= COLLECT_SECONDS:
                    if result.face_landmarks:
                        iris = compute_iris_center(result.face_landmarks[0])
                        if iris:
                            sample.add_sample(*iris)
                            cv2.circle(frame, target_px, 16, (0, 200, 0), 2)
                    cv2.putText(
                        frame,
                        f"{sample.name} 수집 중... {COLLECT_SECONDS - collect_elapsed:.1f}s",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                else:
                    point_index += 1
                    prep_start = now
                    collecting = False
                    continue
            else:
                cv2.putText(
                    frame,
                    f"{sample.name} 지점을 바라봐 주세요...",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"{PREP_SECONDS - elapsed:.1f}s 후 수집 시작",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (200, 200, 200),
                    1,
                    cv2.LINE_AA,
                )

            cv2.circle(frame, target_px, 12, (0, 0, 255), -1)
            cv2.imshow("Iris Calibration", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                print("사용자에 의해 종료되었습니다.")
                break

    cap.release()
    cv2.destroyAllWindows()

    if point_index < len(samples):
        print("캘리브레이션이 완료되지 않았습니다. 결과를 저장하지 않습니다.", file=sys.stderr)
        return

    transform = solve_affine_transform(samples)

    output = {
        "timestamp": time.time(),
        "camera_index": camera_index,
        "prep_seconds": PREP_SECONDS,
        "collect_seconds": COLLECT_SECONDS,
        "points": [
            {
                "name": sample.name,
                "target": sample.target,
                "mean": sample.mean,
                "sample_count": len(sample.samples),
            }
            for sample in samples
        ],
        "transform": transform,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"캘리브레이션 결과를 '{output_path}'에 저장했습니다.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MediaPipe 홍채 캘리브레이션 도구")
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="사용할 카메라 인덱스 (기본값: 0)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="iris_calibration_profile.json",
        help="결과를 저장할 JSON 파일 경로",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_calibration(args.camera, args.output)


if __name__ == "__main__":
    main()

