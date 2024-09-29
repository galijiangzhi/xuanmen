import cv2
import mediapipe as mp

# 初始化MediaPipe手部检测模块
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 读取图像
image = cv2.imread('2.jpg')  # 替换为你的图像路径

# 将图像从BGR转换为RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 进行手部检测
results = hands.process(image_rgb)

# 如果检测到手部，绘制关键点
if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

# 显示图像
cv2.imshow('Hand Landmarks', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 处理检测结果
if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        for id, lm in enumerate(hand_landmarks.landmark):
            h, w, c = image.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            print(f'ID: {id}, X: {cx}, Y: {cy}')