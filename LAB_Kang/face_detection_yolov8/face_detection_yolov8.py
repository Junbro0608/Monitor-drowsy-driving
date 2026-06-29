# !pip install ultralytics

import os
import torch
from ultralytics import YOLO

def main():
    # 1. 'device' 변수 정의 (GPU가 있으면 0, 없으면 'cpu')
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"사용 중인 장치: {device}")

    # 2. 데이터 설정 파일 경로 명시 (실제 파일 경로로 반드시 수정해야 함)
    subset_yaml_path = './files/face_detection.v2i.yolov8/data.yaml'  # 예: 'C:/yolo_data/dataset.yaml'

    if not os.path.exists(subset_yaml_path):
        print(f"오류: '{subset_yaml_path}' 파일을 찾을 수 없습니다. 경로를 확인하세요.")
        return

    # 3. 모델 로드 및 학습
    model = YOLO('yolov8n.pt')

    # results = model.train(
    #     data=subset_yaml_path,      # 정의된 데이터 설정 파일
    #     epochs=80,                  # 충분한 학습 횟수
    #     batch=32,                   # 배치 크기
    #     imgsz=320,                  # 이미지 크기
    #     device='cpu',              
    #     workers=4,                  # 데이터 로딩에 사용할 CPU 코어 수
    #     degrees=15.0,               # 15도까지 랜덤 회전
    #     flipud=0.5,                 # 50% 확률로 상하 반전
    #     fliplr=0.5,                 # 50% 확률로 좌우 반전
    #     mosaic=1.0,                 # 모자이크 데이터 증강 활성화
    #     name='face_yolov8',         # 결과 저장 폴더 이름
    #     plots=True
    # )

    # 최적화가 적용된 학습 코드 예시
    results = model.train(
        data=subset_yaml_path,
        epochs=50,          # 최대 횟수 축소 (조기 종료를 믿고 과감히 줄임)
        patience=10,        # 10번 동안 개선 없으면 스톱
        batch=32,
        imgsz=320,          
        device='cpu',
        workers=4,
        degrees=15.0,
        fliplr=0.5,         # 상하 반전 제거 (얼굴 데이터셋 특성상 불필요하다고 판단), 모자이크 증강도 제거
        freeze=10,          # Backbone 파라미터 동결 (학습 파라미터 대폭 감소)
        mosaic=0.0,         # 불필요한 증강 제거 (빠른 수렴 유도)
        name='face_yolov8',
        plots=True
    )

# Windows 환경에서 workers > 0 일 때 발생하는 무한 루프 오류를 방지하는 핵심 구문
if __name__ == '__main__':
    main()