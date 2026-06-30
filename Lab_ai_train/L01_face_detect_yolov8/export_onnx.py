# from ultralytics import YOLO

# def export_onnx_fp32():
#     # 1. 최적화된 학습이 완료된 모델 로드 (경로는 실제 환경에 맞게 확인 필요)
#     model_path = 'runs/detect/face_yolov8-2/weights/best.pt'
#     model = YOLO(model_path)

#     print("FP32 (Non-Quantized) ONNX 변환을 시작합니다...")

#     # 2. ONNX 포맷으로 내보내기
#     export_path = model.export(
#         format='onnx',
#         imgsz=256,       # 이전 최적화 단계에서 제안한 해상도 적용
#         half=False,      # FP16 양자화 미적용 (기본값)
#         int8=False,      # INT8 양자화 미적용 (기본값)
#         simplify=True,   # ONNX 그래프 구조 단순화 (추론 속도 향상에 필수)
#         dynamic=False    # 동적 해상도 비활성화 (고정 크기가 성능에 더 유리함)
#     )

#     print(f"\n변환 완료! 저장된 경로: {export_path}")

# if __name__ == '__main__':
#     export_onnx_fp32()

import os
from ultralytics import YOLO

def export_all_onnx():
    # 1. 원본 모델 경로 설정
    model_path = 'runs/detect/face_yolov8-2/weights/best.pt'
    
    if not os.path.exists(model_path):
        print(f"오류: '{model_path}' 경로에 파일이 없습니다.")
        return
        
    model = YOLO(model_path)
    base_dir = os.path.dirname(model_path)
    
    # 변환할 옵션 리스트 (이름, half 옵션)
    configs = [
        ('yolov8_face_detector_NQ.onnx', False),
        ('yolov8_face_detector_Q.onnx', True)
    ]

    print("=== ONNX 모델 통합 변환 시작 ===")

    for filename, is_half in configs:
        print(f"\n변환 중: {filename} (Half={is_half}) ...")
        
        # 2. 변환 실행
        exported_path = model.export(
            format='onnx',
            imgsz=320,
            half=is_half,    # True면 FP16 양자화, False면 FP32
            int8=False,      
            simplify=True,   
            dynamic=False    
        )
        
        # 3. 파일 이름 변경
        new_path = os.path.join(base_dir, filename)
        
        if os.path.exists(new_path):
            os.remove(new_path)
        
        os.rename(exported_path, new_path)
        print(f"-> 완료: {new_path}")

    print("\n==============================================")
    print("모든 변환 작업이 성공적으로 종료되었습니다.")
    print("==============================================")

if __name__ == '__main__':
    export_all_onnx()
