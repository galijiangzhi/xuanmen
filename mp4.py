import cv2
import mediapipe as mp

# 初始化MediaPipe手部检测模块
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 读取视频
cap = cv2.VideoCapture('1.mp4')  # 替换为你的视频路径

# 处理每一帧
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 将帧从BGR转换为RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 进行手部检测
    results = hands.process(frame_rgb)

    # 如果检测到手部，绘制关键点
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # 显示帧
    cv2.imshow('Hand Landmarks', frame)

    # 按下 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()