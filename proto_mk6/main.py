import cv2
from ultralytics import YOLO
import time
from collections import deque
import numpy as np
import math
from speaker.mic_speaker_bluetooth import *
import threading
import time


#--------------------------------------#
engine_path = './engine'
face_engine = 'face_yolov8n_fp16.engine'
eyes_engine = 'eye_detect_yolov8n_fp16.engine'
close_engine = 'close_yolov8n_fp16.engine'
pose_engine = 'pose_yolov26n_rev0.pt'
#--------------------------------------#
ollama_model = 'gemma3:4b'
camera = 0
width = 320
height = 240
#--------------------------------------#

# 1. 모델 로드
face_model = YOLO(f"{engine_path}/{face_engine}", task="detect")
eyes_model = YOLO(f"{engine_path}/{eyes_engine}", task="detect")
eyes_close_model = YOLO(f"{engine_path}/{close_engine}", task="classify") 
pose_model = YOLO(f"{engine_path}/{pose_engine}", task="pose")

# 카메라인 경우 (0번 웹캠)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# 윈도우 설정
cv2.namedWindow('cam', cv2.WINDOW_NORMAL)
cv2.resizeWindow('cam', 640, 480)

# FPS 측정 초기값
prev_time = time.time()
fps = 0.0

# 양안 개별 데이터 저장소 (큐)
left_eye_history = deque(maxlen=6)
right_eye_history = deque(maxlen=6)

# 졸음판단 데이터 저장소 (큐)
close_history = deque(maxlen=24)
agree_history = deque(maxlen=24)

last_alarm_time = 0.0  # 마지막으로 알람이 울린 시간
ALARM_COOLDOWN = 5.0   # 알람이 다시 울리기까지의 대기 시간 (5초)
total_sleep_warnings = 0
total_pose_warnings = 0

check_voice = 0

def calculate_angle(pt1, pt2):
    '''두 점을 연결한 축이 기울어진 각도(Roll)를 직관적으로 반환'''
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    radian = math.atan2(-dy, dx)
    degree = math.degrees(radian)
    
    if degree < -90:
        degree += 180
    elif degree > 90:
        degree -= 180
        
    return degree

def ai_detect(current_frame):
    '''얼굴, 눈, 포즈 데이터 수집하는 함수'''
    pose_results = None
    eyes_results = {"left": "NO EYES", "right": "NO EYES", "left_score": 0.0, "right_score": 0.0} 
    face_box = None 
    target_keypoints = None
    head_angle = None
    rotation_angle = 0
    
    detected_eye_boxes = [] 
    found_left = False
    found_right = False

    # 2. 추론 실행 
    face_results = face_model(current_frame, imgsz=320, conf=0.5, iou=0.45, device=0, verbose=False)
    pose_results = pose_model(current_frame, imgsz=320, conf=0.5, iou=0.45, device=0, verbose=False)

    # 포즈 결과 추출 (2번 코, 3번 입술 점)
    if pose_results and len(pose_results[0].keypoints) > 0:
        keypoints = pose_results[0].keypoints.xy.cpu().numpy()[0]
        if len(keypoints) >= 4: 
            pt2 = keypoints[2]
            pt3 = keypoints[3]
            target_keypoints = (pt2, pt3) 

            if pt2[0] > 0 and pt2[1] > 0 and pt3[0] > 0 and pt3[1] > 0:
                head_angle = calculate_angle(pt2, pt3)
    
    # face를 찾을 시 eyes 탐색
    if face_results and len(face_results[0].boxes) > 0:
        boxes = face_results[0].boxes.xyxy.cpu().numpy().astype(int)
        centers = [((b[0]+b[2])/2, (b[1]+b[3])/2) for b in boxes]

        best_idx = min(range(len(boxes)), key=lambda i: (centers[i][0] - 160)**2 + (centers[i][1] - 120)**2)
        x1, y1, x2, y2 = boxes[best_idx]
        
        face_box = (x1, y1, x2, y2) 
        face_crop = current_frame[max(0, y1):min(240, y2), max(0, x1):min(320, x2)]

        # --- face_crop 이미지를 회전시켜 수평 맞추기 ---
        if head_angle is not None and face_crop.size > 0:
            h, w = face_crop.shape[:2]
            center = (w / 2.0, h / 2.0)
            
            if head_angle < 0:
                rotation_angle = head_angle + 90
            else:
                rotation_angle = head_angle - 90
                
            M = cv2.getRotationMatrix2D(center, -1*rotation_angle, 1.0)
            aligned_face = cv2.warpAffine(face_crop, M, (w, h))
        else:
            aligned_face = face_crop
            center = (0, 0)

        # 💡 [수정됨] 얼굴 전체를 흑백으로 바꾸는 코드 삭제. 
        # 객체 탐지는 원본 '컬러' 이미지(aligned_face)를 그대로 사용합니다!
        if aligned_face.size > 0:
            eye_preds = eyes_model(aligned_face, imgsz=416, conf=0.25, iou=0.45, device=0, verbose=False)
            
            if len(eye_preds[0].boxes) > 0:
                eye_boxes_data = eye_preds[0].boxes.xyxy.cpu().numpy().astype(int)
                face_center_x = aligned_face.shape[1] / 2.0
                
                for i in range(len(eye_boxes_data)):
                    ex1, ey1, ex2, ey2 = eye_boxes_data[i]
                    
                    # --- 눈 박스 좌표 원상복구 로직 ---
                    if head_angle is not None:
                        eye_cx = (ex1 + ex2) / 2.0
                        eye_cy = (ey1 + ey2) / 2.0
                        M_inv = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
                        orig_cx = M_inv[0, 0] * eye_cx + M_inv[0, 1] * eye_cy + M_inv[0, 2]
                        orig_cy = M_inv[1, 0] * eye_cx + M_inv[1, 1] * eye_cy + M_inv[1, 2]
                    else:
                        orig_cx = (ex1 + ex2) / 2.0
                        orig_cy = (ey1 + ey2) / 2.0
                        
                    ew = ex2 - ex1
                    eh = ey2 - ey1
                    final_ex = int(orig_cx - (ew / 2) + x1)
                    final_ey = int(orig_cy - (eh / 2) + y1)
                    
                    detected_eye_boxes.append((final_ex, final_ey, ew, eh))
                    
                    # --- 눈 상태(OPEN/CLOSE) 분류 ---
                    # 💡 1. 우선 '컬러' 얼굴 이미지에서 눈 바운딩 박스만큼 잘라냅니다.
                    single_eye_crop = aligned_face[max(0, ey1):min(aligned_face.shape[0], ey2), 
                                                   max(0, ex1):min(aligned_face.shape[1], ex2)]
                    
                    if single_eye_crop.size > 0:
                        
                        # 💡 2. [추가됨] 잘라낸 눈 이미지만 '흑백(3채널 유지)'으로 변환합니다!
                        gray_eye = cv2.cvtColor(single_eye_crop, cv2.COLOR_BGR2GRAY)
                        gray_eye_3ch = cv2.cvtColor(gray_eye, cv2.COLOR_GRAY2BGR)

                        # 왼쪽/오른쪽 눈을 구분해서 창 이름을 다르게 설정합니다.
                        debug_side = "Left" if (ex1 + ex2) / 2.0 < face_center_x else "Right"
                        
                        # 흑백으로 잘 변환되었는지 디버깅 창 띄우기 (128x128 뻥튀기)
                        debug_eye_img = cv2.resize(gray_eye_3ch, (128, 128))
                        cv2.imshow(f"Debug {debug_side} Eye", debug_eye_img)

                        # 💡 3. 분류 추론 모델에는 흑백으로 변환된 눈 이미지(gray_eye_3ch)를 넣습니다.
                        close_preds = eyes_close_model(gray_eye_3ch, imgsz=128, device=0, verbose=False)
                        
                        if close_preds and hasattr(close_preds[0], 'probs') and close_preds[0].probs is not None:
                            top1_class_id = close_preds[0].probs.top1 
                            top1_conf = float(close_preds[0].probs.top1conf)
                            
                            # 터미널 디버깅 출력
                            side_str = "Left" if (ex1 + ex2) / 2.0 < face_center_x else "Right"
                            status_str = "CLOSE" if top1_class_id == 1 else "OPEN"
                            # print(f"[DEBUG] {side_str:5s} Eye | 예측: {status_str:5s} (Class {top1_class_id}) | 신뢰도: {top1_conf*100:.1f}%")

                            score = 1.0 if top1_class_id == 1 else 0.0
                            
                            if (ex1 + ex2) / 2.0 < face_center_x: 
                                left_eye_history.append(score)
                                found_left = True
                            else:                      
                                right_eye_history.append(score)
                                found_right = True
                        else:
                            print("[에러] 분류 모델이 예측 결과를 반환하지 않았습니다.")
                            
            # --- 큐 계산 및 상태 업데이트 ---
            eye_threshold = 0.5
            
            if found_left and left_eye_history:
                avg_l = float(np.mean(left_eye_history))
                eyes_results["left"] = "CLOSE" if avg_l >= eye_threshold else "OPEN"
                eyes_results["left_score"] = avg_l 
                
            if found_right and right_eye_history:
                avg_r = float(np.mean(right_eye_history))
                eyes_results["right"] = "CLOSE" if avg_r >= eye_threshold else "OPEN"
                eyes_results["right_score"] = avg_r 

    return face_results, target_keypoints, eyes_results, face_box, head_angle, detected_eye_boxes

def check_sleep(eyes_results, head_angle):
    '''눈 상태와 머리 각도를 큐에 저장하고, 졸음(sleep) 및 자세 불량(pose)을 판별하는 함수'''
    global last_alarm_time  # 타이머 변수를 끌어다 씁니다.
    global last_alarm_time, total_sleep_warnings
    sleep = 0
    pose = 0

    # 1. 눈(Sleep) 상태 큐 업데이트
    if eyes_results["left"] == "CLOSE" or eyes_results["right"] == "CLOSE":
        close_history.append(1)
    else:
        close_history.append(0)

    # 2. 머리 각도(Pose) 상태 큐 업데이트
    if head_angle is not None:
        if abs(head_angle) < 50:
            agree_history.append(1)
        else:
            agree_history.append(0)
    # 3. 빈도수 체크 및 상태 트리거 
    if len(close_history) > 15:
        if sum(close_history) >= 16:
            sleep = 1

    if len(agree_history) > 15:
        if sum(agree_history) >= 16:
            pose = 1

    # 위험도(눈 감음 비율) 수치화 계산 (%)
    if len(close_history) > 0:
        close_ratio = int((sum(close_history) / len(close_history)) * 100)
    else:
        close_ratio = 0

    # 4. 🚨 알람 재생 (쿨다운 필터 적용)
    current_time = time.time()
    
    # 졸음이나 자세 불량 중 하나라도 걸렸고 AND 마지막 알람 후 5초가 지났을 때만 실행!
    if (sleep == 1 or pose == 1) and (current_time - last_alarm_time > ALARM_COOLDOWN):
        
        last_alarm_time = current_time # 방금 알람이 울렸으니 현재 시간으로 덮어씌움
        
        # 카메라 영상이 멈추지 않도록 스레드(백그라운드)로 mp3를 재생합니다.
        if sleep == 1 and pose == 1:
            print("🚨 [위험] 2단계 졸음 감지! (눈+고개)")
            total_sleep_warnings += 1
            total_pose_warnings += 1
            threading.Thread(target=play_mp3_rev2, args=('save_mp3/sleep_lv2.mp3',), daemon=True).start()
        elif sleep == 1:
            total_sleep_warnings += 1
            print("🚨 [경고] 1단계 졸음 감지! (눈)")
            threading.Thread(target=play_mp3_rev2, args=('save_mp3/sleep_lv1.mp3',), daemon=True).start()
        elif pose == 1:
            print("⚠️ [경고] 자세 불량 감지! (고개)")
            threading.Thread(target=play_mp3_rev2, args=('save_mp3/bad_pose.mp3',), daemon=True).start()

    return close_ratio
        



def render_frame(frame, face_results, target_keypoints, eyes_results, face_box, head_angle, current_fps, eye_rects):
    '''화면에 결과를 그리는 전용 함수'''
    if face_results: 
        frame = face_results[0].plot(img=frame)
        
    if eye_rects:
        for ex, ey, ew, eh in eye_rects:
            cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (255, 0, 255), 2)
            
    if target_keypoints:
        pt2, pt3 = target_keypoints
        if pt2[0] > 0 and pt2[1] > 0:
            cv2.circle(frame, (int(pt2[0]), int(pt2[1])), 4, (255, 0, 0), -1)
            cv2.putText(frame, "2", (int(pt2[0]) + 5, int(pt2[1]) - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            
        if pt3[0] > 0 and pt3[1] > 0:
            cv2.circle(frame, (int(pt3[0]), int(pt3[1])), 4, (255, 0, 0), -1)
            cv2.putText(frame, "3", (int(pt3[0]) + 5, int(pt3[1]) - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

    # 양쪽 눈 깜빡임 결과 상태
    if eyes_results and face_box:
        x1, y1, x2, y2 = face_box 
        
        l_status = eyes_results["left"]
        l_score = eyes_results["left_score"]
        l_color = (0, 0, 255) if l_status == "CLOSE" else ((0, 255, 0) if l_status == "OPEN" else (0, 255, 255))
        l_text = f"L: {l_status} ({l_score:.2f})" 
        
        r_status = eyes_results["right"]
        r_score = eyes_results["right_score"]
        r_color = (0, 0, 255) if r_status == "CLOSE" else ((0, 255, 0) if r_status == "OPEN" else (0, 255, 255))
        r_text = f"R: {r_status} ({r_score:.2f})"
        
        cv2.putText(frame, l_text, (x1, max(40, y1 - 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, l_color, 2, cv2.LINE_AA)
        cv2.putText(frame, r_text, (x1, max(15, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, r_color, 2, cv2.LINE_AA)

    if head_angle is not None:
        angle_text = f"Angle: {head_angle:.1f} deg"
        color = (0, 0, 255) if abs(head_angle) > 30 else (0, 255, 255)
        cv2.putText(frame, angle_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

    if current_fps is None:
        current_fps = 0.0
    cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    
    frame = cv2.resize(frame, (640, 480))
    
    return frame

def process_frame(cap, prev_time):
    success, frame = cap.read()
    if not success:
        return False, prev_time

    face_results, target_keypoints, eyes_results, face_box, head_angle, eye_rects = ai_detect(frame)

    close_ratio = check_sleep(eyes_results, head_angle)
    
    cur_time = time.time()
    dt = cur_time - prev_time
    fps = 1.0 / dt if dt > 0 else 0.0
    updated_time = cur_time 

    frame = render_frame(frame, face_results, target_keypoints, eyes_results, face_box, head_angle, fps, eye_rects)
    cv2.imshow("cam", frame)

    return True, updated_time, close_ratio


def check_mic():
    global check_voice
    while True:
        if check_voice == 0:
            record_audio(5)
            trigger_text = transcribe_audio()
            print("\n트리거 인식 결과:", trigger_text)

            if not is_wake_word(trigger_text):
                print("'안녕' 트리거가 인식되지 않았습니다. 다시 대기합니다.")
                continue

            check_voice = 1
        else:
            time.sleep(0.1)

def llm(close_ratio=0, total_sleep_warnings=0, total_pose_warnings=0):
    '''로컬 LLM으로 실행, 음성->LLM->출력'''
    user_data_txt = (
        f"- 최근 3초간 눈 감음 비율: {close_ratio}%\n"
        f"- 누적 졸음 경고: {total_sleep_warnings}회\n"
        f"- 누적 자세 경고: {total_pose_warnings}회"
    )
    print('LLM input 정보')
    print(user_data_txt)
    print("\n질문을 10초 동안 말하세요.")
    record_audio(seconds=10)
    user_text = transcribe_audio()

    print("\n==============================")
    print("질문 인식 결과")
    print("==============================")
    print(user_text)
    print("==============================")

    if not user_text or user_text.strip() in ["(끝)", "[BLANK_AUDIO]"]:
        speak_text("질문을 인식하지 못했습니다. 다시 말씀해주세요.")
        return False

    answer = get_ollama_text(user_text,user_data_txt)
    print("\n==============================")
    print("Ollama 응답")
    print("==============================")
    print(answer)
    print("==============================")

    speak_text(answer)


if __name__ == "__main__":

    check_mic_thread = threading.Thread(target=check_mic, daemon=True)
    check_mic_thread.start()
    
    while cap.isOpened():
        if check_voice == 1:
            print('LLM 모드 활성화')
            play_mp3_rev2('save_mp3/yes.mp3')
            llm(close_ratio, total_sleep_warnings, total_pose_warnings)
            check_voice = 0
        else:
            success, prev_time, close_ratio = process_frame(cap, prev_time)

            if not success:
                break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()