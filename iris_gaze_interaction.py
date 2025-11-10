"""실시간으로 사용자 시선을 보정하며 타깃과 정렬하는 피드백 루프."""

import argparse
import json
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


# 화면 좌표계(0~1)에 정의된 시선 타깃 목록.
TARGET_POINTS: List[Tuple[str, Tuple[float, float]]] = [
    ("center", (0.5, 0.5)),
    ("left", (0.2, 0.5)),
    ("right", (0.8, 0.5)),
    ("up", (0.5, 0.2)),
    ("down", (0.5, 0.8)),
    ("top-left", (0.2, 0.2)),
    ("top-right", (0.8, 0.2)),
    ("bottom-left", (0.2, 0.8)),
    ("bottom-right", (0.8, 0.8)),
    ("corner-tl-outer", (0.05, 0.05)),
    ("corner-tr-outer", (0.95, 0.05)),
    ("corner-bl-outer", (0.05, 0.95)),
    ("corner-br-outer", (0.95, 0.95)),
    ("corner-tl-inner", (0.15, 0.15)),
    ("corner-tr-inner", (0.85, 0.15)),
    ("corner-bl-inner", (0.15, 0.85)),
    ("corner-br-inner", (0.85, 0.85)),
]

# 양쪽 홍채를 추적하기 위한 MediaPipe 랜드마크 인덱스.
IRIS_INDICES = {
    "left": [468, 469, 470, 471, 472],
    "right": [473, 474, 475, 476, 477],
}

DEFAULT_DWELL_SECONDS = 4.5
DEFAULT_BLEND_RATIO = 0.4
DEFAULT_MAX_SAMPLES = 2000
GROUND_TRUTH_WINDOW_SECONDS = 0.5
SMOOTHING_ALPHA = 0.6
MIRROR_PREVIEW_DEFAULT = True


@dataclass
class FeedbackPoint:
    name: str
    target: Tuple[float, float]
    iris_samples: List[Tuple[float, float]] = field(default_factory=list)
    prediction_samples: List[Tuple[float, float]] = field(default_factory=list)
    distances: List[float] = field(default_factory=list)

    def add_sample(
        self,
        iris_raw: Tuple[float, float],
        prediction: Tuple[float, float],
        distance: float,
    ) -> None:
        self.iris_samples.append(iris_raw)
        self.prediction_samples.append(prediction)
        self.distances.append(distance)

    @property
    def sample_count(self) -> int:
        return len(self.iris_samples)

    @property
    def mean_distance(self) -> float:
        if not self.distances:
            return float("nan")
        return float(np.mean(self.distances))


@dataclass
class FeedbackDataset:
    maxlen: int
    iris_samples: Deque[Tuple[float, float]] = field(init=False)
    target_samples: Deque[Tuple[float, float]] = field(init=False)

    def __post_init__(self) -> None:
        self.iris_samples = deque(maxlen=self.maxlen)
        self.target_samples = deque(maxlen=self.maxlen)

    def add(self, iris_xy: Tuple[float, float], target_xy: Tuple[float, float]) -> None:
        self.iris_samples.append(iris_xy)
        self.target_samples.append(target_xy)

    @property
    def sample_count(self) -> int:
        return len(self.iris_samples)

    def estimate_transform(self) -> Optional[Dict[str, List[List[float]]]]:
        if len(self.iris_samples) < 3:
            return None

        A = np.array([[x, y, 1.0] for x, y in self.iris_samples], dtype=np.float64)
        targets = np.array(self.target_samples, dtype=np.float64)

        target_x = targets[:, 0]
        target_y = targets[:, 1]

        coeff_x, *_ = np.linalg.lstsq(A, target_x, rcond=None)
        coeff_y, *_ = np.linalg.lstsq(A, target_y, rcond=None)

        return {
            "matrix": [
                [float(coeff_x[0]), float(coeff_x[1])],
                [float(coeff_y[0]), float(coeff_y[1])],
            ],
            "bias": [float(coeff_x[2]), float(coeff_y[2])],
        }

    def recent_samples(self, limit: int = 200) -> List[Dict[str, Tuple[float, float]]]:
        data = []
        for iris_xy, target_xy in list(zip(self.iris_samples, self.target_samples))[-limit:]:
            data.append(
                {
                    "iris": (float(iris_xy[0]), float(iris_xy[1])),
                    "target": (float(target_xy[0]), float(target_xy[1])),
                }
            )
        return data


def create_identity_transform() -> Dict[str, List[List[float]]]:
    """정규화된 화면 좌표계에서 항등 원근 변환을 생성한다."""
    return {
        "matrix": [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        "bias": [0.0, 0.0],
    }


def load_initial_transform(profile_path: Optional[str]) -> Tuple[Dict[str, List[List[float]]], Optional[Dict]]:
    """디스크에서 저장된 변환을 읽고, 실패 시 항등 변환을 반환한다."""
    if profile_path and os.path.exists(profile_path):
        try:
            with open(profile_path, "r", encoding="utf-8") as f:
                profile = json.load(f)
        except json.JSONDecodeError as exc:
            print(f"프로파일 JSON을 해석하지 못했습니다: {exc}", file=sys.stderr)
            return create_identity_transform(), None

        transform = profile.get("transform")
        if (
            transform
            and isinstance(transform, dict)
            and "matrix" in transform
            and "bias" in transform
        ):
            matrix = transform["matrix"]
            bias = transform["bias"]
            if (
                isinstance(matrix, list)
                and len(matrix) == 2
                and all(isinstance(row, list) and len(row) == 2 for row in matrix)
                and isinstance(bias, list)
                and len(bias) == 2
            ):
                print(f"프로파일 '{profile_path}'을(를) 불러왔습니다.")
                return {
                    "matrix": [[float(v) for v in row] for row in matrix],
                    "bias": [float(v) for v in bias],
                }, profile
        print(f"프로파일 형식이 올바르지 않습니다. '{profile_path}'을(를) 무시합니다.", file=sys.stderr)
    else:
        if profile_path:
            print(f"프로파일을 찾을 수 없습니다. '{profile_path}' 대신 초기값을 사용합니다.", file=sys.stderr)
    return create_identity_transform(), None


def compute_iris_center(landmarks):
    """양쪽 눈의 홍채 랜드마크를 평균 내어 정규화된 시선 좌표를 구한다."""
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


def apply_transform(transform: Dict[str, List[List[float]]], iris_xy: Tuple[float, float]) -> Tuple[float, float]:
    """홍채 좌표에 현재 변환을 적용해 정규화된 화면 좌표를 반환한다."""
    matrix = np.array(transform["matrix"], dtype=np.float64)
    bias = np.array(transform["bias"], dtype=np.float64)
    iris_vec = np.array([iris_xy[0], iris_xy[1]], dtype=np.float64)
    output = matrix @ iris_vec + bias
    return float(output[0]), float(output[1])


def clip_point(point: Tuple[float, float]) -> Tuple[float, float]:
    """정규화된 좌표를 0~1 범위로 잘라 안정적으로 사용한다."""
    return float(np.clip(point[0], 0.0, 1.0)), float(np.clip(point[1], 0.0, 1.0))


def blend_transforms(
    original: Dict[str, List[List[float]]],
    updated: Dict[str, List[List[float]]],
    blend: float,
) -> Dict[str, List[List[float]]]:
    """두 개의 원근 변환을 선형 보간해 부드럽게 갱신한다."""
    blend = float(np.clip(blend, 0.0, 1.0))
    orig_matrix = np.array(original["matrix"], dtype=np.float64)
    orig_bias = np.array(original["bias"], dtype=np.float64)
    upd_matrix = np.array(updated["matrix"], dtype=np.float64)
    upd_bias = np.array(updated["bias"], dtype=np.float64)

    new_matrix = blend * upd_matrix + (1.0 - blend) * orig_matrix
    new_bias = blend * upd_bias + (1.0 - blend) * orig_bias

    return {
        "matrix": new_matrix.tolist(),
        "bias": new_bias.tolist(),
    }


def draw_targets(frame, active_index: int) -> None:
    """모든 타깃 점을 그리고 현재 바라보는 타깃을 강조 표시한다."""
    h, w, _ = frame.shape
    for idx, (name, coords) in enumerate(TARGET_POINTS):
        center = (int(coords[0] * w), int(coords[1] * h))
        if idx == active_index:
            color = (0, 255, 0)
            radius = 20
            thickness = 3
        else:
            color = (120, 120, 120)
            radius = 12
            thickness = 2
        cv2.circle(frame, center, radius, color, thickness)
        if idx == active_index:
            cv2.putText(
                frame,
                name,
                (center[0] + 10, center[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )


def draw_prediction(frame, prediction_px: Optional[Tuple[int, int]]) -> None:
    """추정된 시선 좌표를 십자선으로 표시한다."""
    if prediction_px is None:
        return
    cv2.circle(frame, prediction_px, 10, (255, 128, 0), 2)
    cv2.line(frame, (prediction_px[0] - 12, prediction_px[1]), (prediction_px[0] + 12, prediction_px[1]), (255, 128, 0), 1)
    cv2.line(frame, (prediction_px[0], prediction_px[1] - 12), (prediction_px[0], prediction_px[1] + 12), (255, 128, 0), 1)


def draw_status(
    frame,
    target_name: str,
    dwell_remaining: float,
    dataset_samples: int,
    target_samples: int,
    total_targets: int,
    blend_ratio: float,
    error: Optional[float],
) -> None:
    """현재 상태와 수집 지표를 화면 구석에 텍스트로 출력한다."""
    lines = [
        f"Target: {target_name}",
        f"Next switch in: {dwell_remaining:.1f}s",
        f"Total samples: {dataset_samples}",
        f"Active target samples: {target_samples}",
        f"Blend ratio: {blend_ratio:.2f}",
        f"Total targets: {total_targets}",
    ]
    if error is not None:
        lines.append(f"Current error (norm): {error:.4f}")

    for idx, text in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (20, 40 + idx * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    instruction = "[SPACE] Next target  |  [S] Save  |  [ESC/Q] Quit"
    cv2.putText(
        frame,
        instruction,
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )


def save_profile(
    path: Optional[str],
    transform: Dict[str, List[List[float]]],
    dataset: FeedbackDataset,
    feedback_points: Dict[str, FeedbackPoint],
    camera_index: int,
) -> None:
    """현재 변환과 주요 통계를 JSON 파일로 저장한다."""
    if not path:
        print("프로파일 저장 경로가 지정되지 않아 저장을 건너뜁니다.")
        return

    timestamp = time.time()
    stats = []
    all_distances = []
    for point in feedback_points.values():
        if point.sample_count == 0:
            continue
        stats.append(
            {
                "name": point.name,
                "target": point.target,
                "sample_count": point.sample_count,
                "mean_distance": point.mean_distance,
            }
        )
        all_distances.extend(point.distances)

    payload = {
        "timestamp": timestamp,
        "camera_index": camera_index,
        "samples": dataset.sample_count,
        "targets": TARGET_POINTS,
        "transform": transform,
        "stats": stats,
        "mean_distance_overall": float(np.mean(all_distances)) if all_distances else None,
        "recent_samples": dataset.recent_samples(200),
    }

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"프로파일을 '{path}'에 저장했습니다.")
    except OSError as exc:
        print(f"프로파일 저장에 실패했습니다: {exc}", file=sys.stderr)


def run_feedback_loop(
    camera_index: int,
    profile_path: Optional[str],
    save_profile_path: Optional[str],
    mirror_preview: bool,
    dwell_seconds: float,
    blend_ratio: float,
    max_samples: int,
) -> None:
    """타깃을 띄우고 시선을 추적하면서 변환을 꾸준히 보정하는 메인 루프."""
    transform, loaded_profile = load_initial_transform(profile_path)
    feedback_points = {
        name: FeedbackPoint(name=name, target=coords) for name, coords in TARGET_POINTS
    }
    dataset = FeedbackDataset(maxlen=max_samples)
    error_history: List[float] = []
    # 이전 세션에서 로드한 프로파일 정보가 있으면 참고용으로 보관한다.
    smoothed_iris: Optional[np.ndarray] = None

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
        print(f"카메라를 열 수 없습니다. camera_index={camera_index}", file=sys.stderr)
        return

    active_index = 0
    last_switch = time.time()

    with FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            success, frame = cap.read()
            if not success:
                print("프레임을 읽지 못했습니다.", file=sys.stderr)
                break

            if mirror_preview:
                frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(time.time() * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            now = time.time()
            dwell_elapsed = now - last_switch
            time_remaining = max(0.0, dwell_seconds - dwell_elapsed)

            target_name, target_norm = TARGET_POINTS[active_index]
            h, w, _ = frame.shape

            prediction_px = None
            current_error = None

            if result.face_landmarks:
                iris = compute_iris_center(result.face_landmarks[0])
                if iris:
                    # 노이즈를 줄이기 위해 지수 이동 평균을 적용한다.
                    iris_np = np.array(iris, dtype=np.float64)
                    if smoothed_iris is None:
                        smoothed_iris = iris_np
                    else:
                        smoothed_iris = (
                            SMOOTHING_ALPHA * iris_np
                            + (1.0 - SMOOTHING_ALPHA) * smoothed_iris
                        )
                    iris_smoothed = (float(smoothed_iris[0]), float(smoothed_iris[1]))

                    prediction_norm = apply_transform(transform, iris_smoothed)
                    prediction_clipped = clip_point(prediction_norm)
                    prediction_px = (
                        int(prediction_clipped[0] * w),
                        int(prediction_clipped[1] * h),
                    )
                    # 변환 후 결과가 타깃과 얼마나 차이나는지 실시간 오차를 계산한다.
                    current_error = float(
                        np.linalg.norm(np.array(prediction_clipped) - np.array(target_norm))
                    )

                    should_record = time_remaining <= GROUND_TRUTH_WINDOW_SECONDS

                    if should_record:
                        # 사용자가 바라본다고 가정한 타깃 좌표와 홍채 좌표를 쌍으로 저장한다.
                        dataset.add(iris_smoothed, target_norm)
                        error_history.append(current_error)

                        feedback_point = feedback_points[target_name]
                        feedback_point.add_sample(
                            iris_smoothed,
                            prediction_clipped,
                            current_error,
                        )

                        # 샘플이 충분할 때마다 새 변환을 추정하고 기존 변환과 섞어 점진적으로 보정한다.
                        updated_transform = dataset.estimate_transform()
                        if updated_transform:
                            transform = blend_transforms(transform, updated_transform, blend_ratio)

            if dwell_elapsed >= dwell_seconds:
                # 일정 시간마다 다음 타깃으로 전환하여 화면 전체 영역을 순회한다.
                active_index = (active_index + 1) % len(TARGET_POINTS)
                last_switch = now
                dwell_elapsed = 0.0
                target_name, target_norm = TARGET_POINTS[active_index]
                time_remaining = dwell_seconds
                smoothed_iris = None

            time_remaining = max(0.0, dwell_seconds - dwell_elapsed)
            draw_targets(frame, active_index)
            draw_prediction(frame, prediction_px)
            draw_status(
                frame,
                target_name=target_name,
                dwell_remaining=max(0.0, dwell_seconds - dwell_elapsed),
                dataset_samples=dataset.sample_count,
                target_samples=feedback_points[target_name].sample_count,
                total_targets=len(TARGET_POINTS),
                blend_ratio=blend_ratio,
                error=current_error,
            )

            cv2.imshow("Iris Gaze Feedback Loop", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                print("사용자 입력으로 세션을 종료합니다.")
                break
            if key == ord(" "):
                # 스페이스바를 누르면 즉시 다음 타깃으로 이동한다.
                active_index = (active_index + 1) % len(TARGET_POINTS)
                last_switch = time.time()
                smoothed_iris = None
                target_name, target_norm = TARGET_POINTS[active_index]
            if key == ord("s"):
                # 'S' 키 입력 시 현재까지의 결과를 중간 저장한다.
                save_profile(save_profile_path or profile_path, transform, dataset, feedback_points, camera_index)

    cap.release()
    cv2.destroyAllWindows()

    print("\n=== 세션 요약 ===")
    for point in feedback_points.values():
        if point.sample_count == 0:
            print(f"- {point.name:12s}: 수집된 샘플 없음")
            continue
        print(
            f"- {point.name:12s}: 샘플 {point.sample_count:4d}, 평균 오차 {point.mean_distance:.4f}"
        )

    if error_history:
        print(f"\n전체 평균 오차: {np.mean(error_history):.4f}")
    else:
        print("\n오차 샘플을 수집하지 못했습니다.")

    save_profile(save_profile_path or profile_path, transform, dataset, feedback_points, camera_index)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time feedback-driven MediaPipe iris alignment loop")
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index to open (default: 0)",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="iris_calibration_profile.json",
        help="Optional JSON path containing an initial transform",
    )
    parser.add_argument(
        "--save-profile",
        type=str,
        default=None,
        help="Output path to save the updated profile (default: --profile path)",
    )
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="Disable mirrored preview",
    )
    parser.add_argument(
        "--dwell-seconds",
        type=float,
        default=DEFAULT_DWELL_SECONDS,
        help="Seconds to dwell on each target",
    )
    parser.add_argument(
        "--blend",
        type=float,
        default=DEFAULT_BLEND_RATIO,
        help="Blend ratio for combining new and existing transforms (0.0~1.0)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=DEFAULT_MAX_SAMPLES,
        help="Maximum number of feedback samples to retain in memory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mirror_preview = MIRROR_PREVIEW_DEFAULT
    if args.no_mirror:
        mirror_preview = False

    save_profile_path = args.save_profile or args.profile

    run_feedback_loop(
        camera_index=args.camera,
        profile_path=args.profile,
        save_profile_path=save_profile_path,
        mirror_preview=mirror_preview,
        dwell_seconds=args.dwell_seconds,
        blend_ratio=args.blend,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()

