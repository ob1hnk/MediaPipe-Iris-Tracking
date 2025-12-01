## MediaPipe Iris Tracking Calibration

이 프로젝트는 **MediaPipe Face Landmarker**로부터 추출한 홍채(iris) 랜드마크를 이용해  
사용자의 **시선을 화면 좌표로 보정·추정**하는 Python 기반 데모입니다.

- `iris_live_stream.py` : 웹캠 영상에서 홍채 위치를 실시간으로 시각화
- `iris_gaze_interaction.py` : 화면 타깃을 이용해 시선 **캘리브레이션(보정)** 을 수행하고  
  홍채 좌표 → 화면 좌표 변환 행렬을 학습해 JSON 파일로 저장

---

## 주요 기능

- **홍채 실시간 스트리밍 (`iris_live_stream.py`)**
  - MediaPipe Face Landmarker를 사용해 얼굴 랜드마크를 검출
  - 양쪽 눈의 홍채 주변 랜드마크(468~477)를 이용해 홍채 중심을 계산
  - 각 눈의 홍채 위치를 프레임 위에 텍스트와 점으로 표시

- **시선 캘리브레이션 & 피드백 루프 (`iris_gaze_interaction.py`)**
  - 화면 좌표계(0~1) 상에 정의된 여러 타깃 포인트(`TARGET_POINTS`)를 순차적으로 보여줌
  - 사용자가 타깃을 바라보는 동안 (iris 좌표, target 좌표) 쌍을 수집
  - 최소자승법(Least Squares)으로 **affine transform**을 추정해 홍채 좌표 → 화면 좌표 변환 행렬 학습
  - 학습된 변환을 `iris_calibration_profile.json`에 저장/로드하여 세션 간 재사용
  - 실시간으로 새 데이터를 받으면 기존 변환과 선형 보간(`blend_transforms`)하여 부드럽게 보정

- **캘리브레이션 프로파일(JSON) 관리**
  - 변환 행렬(`matrix`, `bias`), 타깃별 통계(`stats`), 최근 샘플(`recent_samples`) 등을 JSON으로 관리
  - `load_initial_transform()`으로 기존 프로파일을 로드하고,  
    `save_profile()`로 업데이트된 결과를 저장

---

## 파일 구조

- `iris_live_stream.py`  
  간단한 MediaPipe Face Landmarker 데모. 홍채 위치를 픽셀 좌표로 시각화하는 스크립트입니다.

- `iris_gaze_interaction.py`  
  시선 캘리브레이션 및 피드백 루프 메인 스크립트입니다.
  - `TARGET_POINTS` : 화면 상의 타깃 목록 (center, left, right, corners 등)
  - `FeedbackDataset` : (iris, target) 쌍을 저장하고 변환 행렬을 추정하는 데이터셋 클래스
  - `FeedbackPoint` : 타깃별 샘플/오차를 관리하는 클래스
  - `create_identity_transform()` : 기본 항등 변환 생성
  - `apply_transform()` : 홍채 좌표에 변환을 적용해 화면 좌표를 반환
  - `blend_transforms()` : 새 변환과 기존 변환을 섞어 점진적으로 보정
  - `run_feedback_loop()` : 전체 캘리브레이션/시각화를 담당하는 메인 루프

- `iris_calibration_profile.json`  
  캘리브레이션 결과(변환 행렬, 타깃별 통계, 최근 샘플 등)를 저장하는 JSON 파일입니다.  
  최초 실행 시 없을 수 있으며, 세션이 끝나면 자동으로 생성/갱신됩니다.

- `CALIBRATION_BLOG_CONTENT.md`  
  캘리브레이션 알고리즘, JSON 구조, 수학적 배경을 설명하는 기술 문서(블로그 초안)입니다.

---

## 사전 준비

### Python 환경

- **Python**: 3.8 이상 권장
- **필수 패키지**
  - `mediapipe`
  - `opencv-python`
  - `numpy`

예시:

```bash
pip install mediapipe opencv-python numpy
```

또는 `requirements.txt`를 직접 만들어 관리해도 좋습니다.

### 모델 파일 (`face_landmarker.task`)

두 스크립트 모두 MediaPipe Face Landmarker 모델 파일을 필요로 합니다.

- 프로젝트 루트에 `face_landmarker.task` 파일을 위치시키고,
- `iris_live_stream.py`, `iris_gaze_interaction.py`에서 `model_asset_path="face_landmarker.task"` 경로를 사용합니다.

> 모델 파일은 MediaPipe 공식 리포지토리 또는 배포 페이지에서 다운로드할 수 있습니다.

---

## 사용 방법

### 1. 홍채 라이브 스트리밍 데모 (`iris_live_stream.py`)

웹캠에서 홍채 위치를 실시간으로 추적하며, 결과를 화면에 렌더링합니다.

```bash
cd /Users/ob1hnk/Projects/Unity/MediaPipe-Iris-Tracking

# 기본 카메라(인덱스 0)
python iris_live_stream.py

# 다른 카메라 인덱스를 사용하고 싶을 때
python iris_live_stream.py 1
```

- 창 제목: `MediaPipe Iris Live`
- 종료: `Esc` 또는 `q` 키

### 2. 시선 캘리브레이션 & 피드백 루프 (`iris_gaze_interaction.py`)

화면에 타깃 점을 순차적으로 띄우고, 사용자가 각 타깃을 바라보는 동안  
(홍채 좌표, 타깃 좌표) 쌍을 수집해 변환 행렬을 학습합니다.

```bash
cd /Users/ob1hnk/Projects/Unity/MediaPipe-Iris-Tracking

python iris_gaze_interaction.py \
  --camera 0 \
  --profile iris_calibration_profile.json
```

- **주요 옵션**
  - `--camera` : 사용할 카메라 인덱스 (기본값: `0`)
  - `--profile` : 초기 캘리브레이션 프로파일 JSON 경로 (기본값: `iris_calibration_profile.json`)
  - `--save-profile` : 세션 종료 시 저장할 프로파일 경로  
    (지정하지 않으면 `--profile` 경로에 덮어쓰기)
  - `--no-mirror` : 미러(좌우 반전) 프리뷰 비활성화
  - `--dwell-seconds` : 각 타깃에 머무는 시간 (기본: `4.5`초)
  - `--blend` : 새 변환과 기존 변환을 섞는 비율 (기본: `0.4`)
  - `--max-samples` : 메모리에 유지할 최대 피드백 샘플 수 (기본: `2000`)

- **런타임 조작 키**
  - `SPACE` : 즉시 다음 타깃으로 전환
  - `S` : 현재까지의 결과를 JSON 파일로 저장
  - `Esc` 또는 `q` : 세션 종료 및 최종 결과 저장

- **출력**
  - 화면: 타깃 원, 예측 시선 십자선, 좌상단 상태 텍스트
  - 터미널: 타깃별 샘플 수/평균 오차, 전체 평균 오차, 저장 경로 로그
  - JSON: `iris_calibration_profile.json` (또는 지정 경로)에 변환 및 통계 저장

---

## 캘리브레이션 프로파일(JSON) 개요

`iris_calibration_profile.json`에는 대략 다음과 같은 정보가 들어갑니다.

- `transform`
  - `matrix` : 2x2 변환 행렬 (홍채 좌표 → 화면 좌표)
  - `bias` : 2D 오프셋 벡터
  - 수식: `screen_coord = matrix @ iris_coord + bias`

- `stats`
  - 타깃별 샘플 수(`sample_count`)
  - 평균 오차(`mean_distance`)
  - 캘리브레이션 품질 평가에 사용

- `recent_samples`
  - 최근 (iris, target) 샘플 일부 (디버깅/시각화용)

- `mean_distance_overall`
  - 전체 평균 오차

구체적인 JSON 예시는 `CALIBRATION_BLOG_CONTENT.md`에 포함되어 있습니다.

---

## 참고 및 확장

- 알고리즘/수학적 배경, JSON 구조, FAQ 등은  
  `CALIBRATION_BLOG_CONTENT.md`에서 자세히 설명합니다.
- 향후에는
  - 다른 모니터 해상도/레이아웃에 맞춰 타깃 포인트를 커스터마이즈하거나
  - 학습된 변환을 다른 애플리케이션(UI 제스처, 포인터 제어 등)에 적용하는 등  
    다양한 확장이 가능합니다.

---

## 라이선스

현재 별도의 라이선스 파일은 포함되어 있지 않습니다.  
MediaPipe 및 기타 외부 라이브러리의 라이선스를 준수하면서,  
필요에 따라 이 저장소에 맞는 라이선스를 `LICENSE` 파일로 추가해 사용하시길 권장합니다.


