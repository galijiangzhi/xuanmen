import cv2
import mediapipe as mp
from pathlib import Path

# 初始化MediaPipe手部检测模块
def video2txt(videopath,txtpath):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,  # 视频流模式（跟踪优先）
        max_num_hands=2,  # 最多检测2只手
        model_complexity=1,  # 中等复杂度模型
        min_detection_confidence=0.4,  # 检测置信度阈值较高
        min_tracking_confidence=0.3  # 跟踪置信度阈值
    )
    mp_drawing = mp.solutions.drawing_utils

    # 读取视频
    cap = cv2.VideoCapture(videopath)  # 替换为你的视频路径
    txtpath = Path(txtpath)
    txtpath.parent.mkdir(parents=True, exist_ok=True)
    # 打开文件用于保存相对坐标
    output_file = open(txtpath, 'w')

    # 处理每一帧
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 将帧从BGR转换为RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 进行手部检测
        results = hands.process(frame_rgb)

        # 初始化手的相对坐标列表
        hand_relative_coords = []  # 动态存储每只手的相对坐标

        # 如果检测到手部，获取关键点坐标
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 获取根点（手腕）的坐标
                root_x = hand_landmarks.landmark[0].x
                root_y = hand_landmarks.landmark[0].y
                root_z = hand_landmarks.landmark[0].z

                # 计算相对坐标
                relative_coords = []  # 存储当前手的相对坐标
                for landmark in hand_landmarks.landmark:
                    relative_x = landmark.x - root_x
                    relative_y = landmark.y - root_y
                    relative_z = landmark.z - root_z
                    relative_coords.extend([relative_x, relative_y, relative_z])  # 添加相对坐标
                hand_relative_coords.append(relative_coords)  # 将当前手的相对坐标添加到列表中

                # 绘制关键点
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # 如果检测到的手少于2只，补充0
        while len(hand_relative_coords) < 2:
            hand_relative_coords.append([0] * 21 * 3)  # 补充0

        # 将左右手相对坐标写入文件
        output_file.write(' '.join(map(str, hand_relative_coords[0])) + ' ')  # 左手相对坐标
        output_file.write(' '.join(map(str, hand_relative_coords[1])) + '\n')  # 右手相对坐标 + 换行

        # # 显示帧
        # cv2.imshow('Hand Landmarks', frame)
        #
        # # 按下 'q' 键退出
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    output_file.close()