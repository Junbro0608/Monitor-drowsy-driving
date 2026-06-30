from ultralytics import YOLO

def main():
    # 1. 모델 로드 (가장 가벼운 Nano 모델 사용)
    # 처음 실행 시 가중치 파일(yolo26n-pose.pt)을 자동으로 다운로드합니다.
    model = YOLO("yolo26n-pose.pt")

    # 2. 모델 학습 (Training)
    print("🚀 학습을 시작합니다...")
    results = model.train(
        data="datasets/data.yaml",   # 데이터셋 설정 파일 경로 (매우 중요)
        epochs=1000,                # 총 학습 반복 횟수
        imgsz=320,                 # 입력 이미지 해상도 (차량용 Edge 기기 타겟이면 320~640 권장)
        batch=16,                  # 배치 사이즈 (GPU VRAM에 맞춰 8, 16, 32 등으로 조절)
        device=0,                  # 0: 첫 번째 GPU 사용 ('cpu' 입력 시 CPU 사용)
        project="dms_project",     # 결과물이 저장될 최상위 폴더명
        name="yolo26_nano_run",    # 현재 학습 세션의 폴더명
        patience=20,               # Early Stopping (20 에포크 동안 성능 개선이 없으면 조기 종료)
        
        # 차량 내부 악조건을 극복하기 위한 데이터 증강(Augmentation) 옵션
        hsv_v=0.4,                 # 명도 변화 (주간/야간, 터널 환경 대비)
        hsv_s=0.2,                 # 채도 변화
        degrees=50.0,              # 이미지 회전 (카메라 설치 각도 미세 오차 대비)
        translate=0.1,             # 이미지 상하좌우 이동
        fliplr=0.5                 # 좌우 반전 (운전석/조수석 좌우 반전 대비 확률 50%)
    )

    # 3. 검증 (Validation)
    print("✅ 검증을 진행합니다...")
    metrics = model.val()

    # 4. 테스트 추론 (Inference)
    print("🎥 샘플 영상으로 테스트합니다...")
    # 테스트할 이미지나 영상 경로를 입력하면 결과물이 저장됩니다.
    model("test_video.mp4", save=True, project="dms_project", name="predict_result")

if __name__ == '__main__':
    main()