# MediaPipe Iris Tracking Calibration 기술 블로그 내용

## 1. Calibration 개요

### Calibration이란?

Calibration(보정)은 MediaPipe로 추출한 홍채 좌표를 실제 화면 좌표로 정확하게 매핑하기 위한 변환 행렬을 학습하는 과정입니다. 

사용자의 얼굴 구조, 카메라 위치, 모니터 거리 등에 따라 홍채의 상대적 위치와 실제 시선이 향하는 화면 좌표 간에는 개인별 차이가 존재합니다. Calibration은 이러한 개인차를 보정하여 정확한 시선 추적을 가능하게 합니다.

### 왜 필요한가?

1. **개인차 보정**: 사람마다 얼굴 구조, 눈의 위치, 카메라와의 거리가 다르기 때문에 범용적인 변환이 불가능합니다.
2. **정확도 향상**: 보정 없이는 홍채 좌표만으로는 실제 시선 위치를 정확히 예측할 수 없습니다.
3. **실시간 적응**: 사용자가 움직이거나 환경이 변해도 점진적으로 보정을 업데이트할 수 있습니다.

### 어떻게 활용되는가?

1. **초기 Calibration**: 사용자가 화면의 여러 타깃 포인트를 바라보는 동안 (iris 좌표, target 좌표) 쌍을 수집합니다.
2. **변환 행렬 계산**: 수집된 데이터로 최소자승법(Least Squares)을 사용해 affine transformation을 계산합니다.
3. **실시간 적용**: 이후 프레임마다 홍채 좌표에 변환 행렬을 적용하여 화면 좌표를 예측합니다.
4. **점진적 업데이트**: 사용자가 계속 사용하는 동안 새로운 데이터로 변환을 부드럽게 업데이트합니다.

---

## 2. Calibration 핵심 코드 발췌

### 2.1 변환 행렬 추정 (Least Squares)

```python
def estimate_transform(self) -> Optional[Dict[str, List[List[float]]]]:
    """수집된 홍채 좌표와 타깃 좌표로부터 affine transformation을 추정한다."""
    if len(self.iris_samples) < 3:
        return None

    # 입력: [x, y, 1] 형태의 homogeneous 좌표
    A = np.array([[x, y, 1.0] for x, y in self.iris_samples], dtype=np.float64)
    targets = np.array(self.target_samples, dtype=np.float64)

    target_x = targets[:, 0]
    target_y = targets[:, 1]

    # 최소자승법으로 변환 계수 계산
    # Ax = b 형태에서 x를 구함
    coeff_x, *_ = np.linalg.lstsq(A, target_x, rcond=None)
    coeff_y, *_ = np.linalg.lstsq(A, target_y, rcond=None)

    return {
        "matrix": [
            [float(coeff_x[0]), float(coeff_x[1])],  # x 변환 계수
            [float(coeff_y[0]), float(coeff_y[1])],  # y 변환 계수
        ],
        "bias": [float(coeff_x[2]), float(coeff_y[2])],  # 오프셋
    }
```

**수학적 설명**: 
- Affine transformation: `[x', y'] = M * [x, y] + b`
- 여기서 M은 2x2 행렬, b는 2x1 벡터입니다.
- 최소자승법으로 실제 타깃 좌표와 예측 좌표 간 오차를 최소화하는 계수를 찾습니다.

### 2.2 변환 적용

```python
def apply_transform(transform: Dict[str, List[List[float]]], iris_xy: Tuple[float, float]) -> Tuple[float, float]:
    """홍채 좌표에 현재 변환을 적용해 정규화된 화면 좌표를 반환한다."""
    matrix = np.array(transform["matrix"], dtype=np.float64)
    bias = np.array(transform["bias"], dtype=np.float64)
    iris_vec = np.array([iris_xy[0], iris_xy[1]], dtype=np.float64)
    
    # 행렬 곱셈과 덧셈으로 변환
    output = matrix @ iris_vec + bias
    return float(output[0]), float(output[1])
```

### 2.3 점진적 업데이트 (Blending)

```python
def blend_transforms(
    original: Dict[str, List[List[float]]],
    updated: Dict[str, List[List[float]]],
    blend: float,
) -> Dict[str, List[List[float]]]:
    """두 개의 변환을 선형 보간해 부드럽게 갱신한다."""
    blend = float(np.clip(blend, 0.0, 1.0))
    
    orig_matrix = np.array(original["matrix"], dtype=np.float64)
    orig_bias = np.array(original["bias"], dtype=np.float64)
    upd_matrix = np.array(updated["matrix"], dtype=np.float64)
    upd_bias = np.array(updated["bias"], dtype=np.float64)

    # 선형 보간으로 부드러운 전환
    new_matrix = blend * upd_matrix + (1.0 - blend) * orig_matrix
    new_bias = blend * upd_bias + (1.0 - blend) * orig_bias

    return {
        "matrix": new_matrix.tolist(),
        "bias": new_bias.tolist(),
    }
```

**설명**: 
- `blend` 비율(기본 0.4)로 새 변환과 기존 변환을 섞어 급격한 변화를 방지합니다.
- 실시간으로 변환이 업데이트되어도 부드러운 전환이 가능합니다.

### 2.4 실시간 피드백 루프

```python
# 타깃을 바라보는 마지막 0.5초 동안만 샘플 수집
should_record = time_remaining <= GROUND_TRUTH_WINDOW_SECONDS

if should_record:
    # (홍채 좌표, 타깃 좌표) 쌍 저장
    dataset.add(iris_smoothed, target_norm)
    
    # 충분한 샘플이 모이면 변환 추정 및 업데이트
    updated_transform = dataset.estimate_transform()
    if updated_transform:
        transform = blend_transforms(transform, updated_transform, blend_ratio)
```

**설명**:
- 각 타깃에 4.5초 동안 머무르며, 마지막 0.5초 동안만 샘플을 수집합니다.
- 이는 사용자가 타깃을 정확히 바라보고 있다고 가정할 수 있는 시간입니다.

---

## 3. JSON 파일에 저장되는 정보와 활용

### 3.1 JSON 파일 구조

```json
{
  "timestamp": 1764128018.6808372,        // 캘리브레이션 수행 시간
  "camera_index": 0,                       // 사용된 카메라 인덱스
  "samples": 142,                          // 수집된 총 샘플 수
  "targets": [                             // 타깃 포인트 목록
    ["center", [0.5, 0.5]],
    ["left", [0.2, 0.5]],
    ...
  ],
  "transform": {                           // 핵심: 변환 행렬
    "matrix": [
      [43.28, 12.17],                      // x 변환 계수
      [10.97, 35.52]                       // y 변환 계수
    ],
    "bias": [-22.47, -12.50]              // 오프셋
  },
  "stats": [                               // 각 타깃별 통계
    {
      "name": "center",
      "target": [0.5, 0.5],
      "sample_count": 14,
      "mean_distance": 0.2135              // 평균 오차
    },
    ...
  ],
  "mean_distance_overall": 0.2154,         // 전체 평균 오차
  "recent_samples": [                      // 최근 200개 샘플 (디버깅용)
    {
      "iris": [0.4733, 0.2101],
      "target": [0.5, 0.5]
    },
    ...
  ]
}
```

### 3.2 각 필드의 의미

#### `transform` (핵심 데이터)
- **matrix**: 2x2 변환 행렬로, 홍채 좌표를 화면 좌표로 변환하는 선형 변환 계수
- **bias**: 평행 이동 벡터로, 변환 후 추가 오프셋
- **수식**: `screen_coord = matrix @ iris_coord + bias`

#### `stats`
- 각 타깃 포인트별로 수집된 샘플 수와 평균 오차를 기록
- 캘리브레이션 품질을 평가하는 데 사용
- 특정 방향에서 오차가 크면 해당 영역의 보정이 부족함을 의미

#### `recent_samples`
- 최근 수집된 (iris, target) 쌍 데이터
- 디버깅이나 재학습 시 활용 가능
- 시각화하여 데이터 분포를 확인할 수 있음

### 3.3 저장 및 로드

#### 저장 (`save_profile`)
```python
def save_profile(
    path: Optional[str],
    transform: Dict[str, List[List[float]]],
    dataset: FeedbackDataset,
    feedback_points: Dict[str, FeedbackPoint],
    camera_index: int,
) -> None:
    """현재 변환과 주요 통계를 JSON 파일로 저장한다."""
    payload = {
        "timestamp": time.time(),
        "camera_index": camera_index,
        "samples": dataset.sample_count,
        "targets": TARGET_POINTS,
        "transform": transform,              # 핵심 변환 데이터
        "stats": stats,
        "mean_distance_overall": ...,
        "recent_samples": dataset.recent_samples(200),
    }
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
```

#### 로드 (`load_initial_transform`)
```python
def load_initial_transform(profile_path: Optional[str]) -> Tuple[Dict, Optional[Dict]]:
    """디스크에서 저장된 변환을 읽고, 실패 시 항등 변환을 반환한다."""
    if profile_path and os.path.exists(profile_path):
        with open(profile_path, "r", encoding="utf-8") as f:
            profile = json.load(f)
        
        transform = profile.get("transform")
        if transform and "matrix" in transform and "bias" in transform:
            return transform, profile
    
    # 파일이 없거나 형식이 잘못된 경우 항등 변환 반환
    return create_identity_transform(), None
```

### 3.4 활용 방법

1. **세션 간 지속성**: 한 번 캘리브레이션한 결과를 저장하여 다음 실행 시 바로 사용
2. **점진적 개선**: 기존 변환을 로드하고 새로운 데이터로 업데이트하여 정확도 향상
3. **품질 평가**: `mean_distance_overall`로 캘리브레이션 품질 확인
4. **디버깅**: `recent_samples`로 데이터 분포 확인 및 문제 진단
5. **다중 사용자 지원**: 사용자별로 다른 프로파일 파일 저장 가능

### 3.5 활용 예시

```python
# 프로그램 시작 시 기존 캘리브레이션 로드
transform, profile = load_initial_transform("iris_calibration_profile.json")

# 변환 적용
prediction = apply_transform(transform, iris_coord)

# 실시간 업데이트 후 저장
updated_transform = dataset.estimate_transform()
if updated_transform:
    transform = blend_transforms(transform, updated_transform, 0.4)
    save_profile("iris_calibration_profile.json", transform, ...)
```

### 3.6 캘리브레이션 파일 관리 FAQ

- **Q. 캘리브레이션을 한 번 돌릴 때마다 JSON이 새로 생기나요?**  
  기본 설정에서는 아닙니다. `--profile` 인자로 전달한 동일한 경로를 계속 사용하므로, 세션이 끝날 때마다 같은 파일을 덮어씁니다. 새로운 파일을 원하면 실행할 때 `--save-profile my_profile_YYYYMMDD.json`처럼 다른 경로를 지정하세요.

- **Q. 과거 캘리브레이션 기록이 하나의 JSON에 누적되나요?**  
  자동으로 누적되지 않습니다. JSON에는 항상 현재 세션에서 학습한 최신 변환과 그 통계만 들어갑니다. 과거 기록을 보존하려면 세션마다 다른 파일명으로 저장하거나 버전 관리(Git/S3 등)를 활용해 별도로 보관해야 합니다.

- **Q. 앱이 실행될 때 어떤 파일을 로드하나요?**  
  `load_initial_transform()`이 `--profile`로 지정된 단일 파일을 읽어 초기 변환으로 사용합니다. 여러 기록 중에서 하나를 선택해 로드하고 싶다면 실행 시 원하는 파일 경로를 명시하면 됩니다.

---

## 요약

MediaPipe Iris Tracking의 Calibration은:
1. **개인별 차이를 보정**하여 정확한 시선 추적을 가능하게 합니다.
2. **최소자승법**으로 홍채 좌표와 화면 좌표 간의 변환 행렬을 학습합니다.
3. **JSON 파일**로 변환 행렬과 통계를 저장하여 세션 간 지속성을 제공합니다.
4. **점진적 업데이트**로 실시간으로 정확도를 개선할 수 있습니다.

이러한 방식으로 사용자 친화적이고 정확한 시선 추적 시스템을 구현할 수 있습니다.

