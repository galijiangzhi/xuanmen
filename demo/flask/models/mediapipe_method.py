import mediapipe as mp
import cv2


def process_video_with_mediapipe(
        video_path,
        output_path='./provideo.mp4',
        save_bool=False,
        sample_interval=1  # 新增参数：采样间隔（默认每帧处理）
):
    """Process video with MediaPipe, return hand landmarks and optionally save annotated video.

    Args:
        video_path: 输入视频路径
        output_path: 输出视频路径（仅在 save_bool=True 时有效）
        save_bool: 是否保存带标记的视频
        sample_interval: 采样间隔（例如 5 表示每5帧处理1帧）
    """
    print("Starting video processing...")
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.3
    )

    landmarks_data = []

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("无法打开视频文件")

        # 获取视频属性（仅在需要保存时使用）
        if save_bool:
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        processed_count = 0  # 新增：记录实际处理的帧数

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 采样逻辑：跳过非目标帧
            if frame_count % sample_interval != 0:
                frame_count += 1
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if save_bool:
                annotated_frame = frame.copy()

            frame_landmarks = []

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    if save_bool:
                        mp_drawing.draw_landmarks(
                            annotated_frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                            mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                        )

                    # 计算相对坐标（与原逻辑一致）
                    root = hand_landmarks.landmark[0]
                    hand_data = []
                    for landmark in hand_landmarks.landmark:
                        relative_x = landmark.x - root.x
                        relative_y = landmark.y - root.y
                        relative_z = landmark.z - root.z
                        hand_data.extend([relative_x, relative_y, relative_z])
                    frame_landmarks.append(hand_data)

            # 不足2只手时补零
            while len(frame_landmarks) < 2:
                frame_landmarks.append([0.0] * 21 * 3)

            landmarks_data.append(frame_landmarks)

            if save_bool:
                out.write(annotated_frame)

            processed_count += 1
            frame_count += 1

            if processed_count % 30 == 0:
                print(f"Processed {processed_count} frames (sampled from {frame_count} total frames)...")

        print(f"Finished processing. Sampled {processed_count} frames from {frame_count} total frames.")
        return landmarks_data

    except Exception as e:
        print(f"Error during processing: {str(e)}")
        raise
    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        if save_bool and 'out' in locals():
            out.release()
        hands.close()
        print("Released resources")
