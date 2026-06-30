### 📂 Repository Structure

이 저장소는 운전자 상태 모니터링(DMS)을 위한 AI 모델 학습 및 평가 코드를 포함하고 있습니다. 각 폴더는 파이프라인의 주요 기능별로 나뉘어 있습니다.

```text
Lab_ai_train/
├── L00_face_detect_mobilenetv3/ # MobileNetV3 기반 얼굴 탐지 모델 
├── L01_face_detect_yolov8/      # YOLOv8 기반 얼굴 탐지 모델 (학습 및 가중치)
├── L02_pose_detect/             # 운전자 자세 추정(Pose Detection) 데이터셋 및 모델 학습
├── L03_eye_state_comparison/    # 눈 상태(감김/열림 등) 분류 모델 비교 및 분석
├── L04_eye_roll_aug_finetune/   # 눈동자 움직임 데이터 증강(Augmentation) 및 파인튜닝
├── L05_benchmark/               # 전체 파이프라인 및 모델 성능 벤치마크(평가) 코드
└── L06_eye_position_detection/  # 눈동자 위치 탐지(Eye position) 모델 및 학습 코드
```

**📌 주요 구성 요소:**

* **Face & Pose Detection (`L00` ~ `L02`)**
  * 운전자의 얼굴 영역 및 신체 특징점(Keypoints)을 추출하여 기본 자세를 인식합니다.
* **Eye State & Tracking (`L03`, `L04`, `L06`)**
  * 졸음운전 판단의 핵심이 되는 눈의 상태(감김 정도)와 눈동자의 위치/움직임을 정밀하게 추적합니다.
* **Evaluation (`L05`)**
  * 각 모델의 추론 속도 및 정확도를 종합적으로 평가합니다.