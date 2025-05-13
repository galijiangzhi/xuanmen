import mediapipe as mp
import cv2
def process_video_with_mediapipe(video_path, output_path='./provideo.mp4', save_bool=False):
    """Process video with MediaPipe, return hand landmarks and optionally save annotated video"""
    print("Starting video processing...")
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils  # 用于绘制标记
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

        # 获取原始视频的帧率和尺寸（仅在需要保存时使用）
        if save_bool:
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或者使用 'avc1'
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if save_bool:
                annotated_frame = frame.copy()  # 创建用于绘制的帧副本

            frame_landmarks = []

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    if save_bool:
                        # 绘制手部标记和连接线
                        mp_drawing.draw_landmarks(
                            annotated_frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                            mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                        )

                    # Get wrist (root) coordinates
                    root = hand_landmarks.landmark[0]
                    hand_data = []

                    for landmark in hand_landmarks.landmark:
                        # Calculate relative coordinates
                        relative_x = landmark.x - root.x
                        relative_y = landmark.y - root.y
                        relative_z = landmark.z - root.z
                        hand_data.extend([relative_x, relative_y, relative_z])

                    frame_landmarks.append(hand_data)

            # Pad with zeros if less than 2 hands detected
            while len(frame_landmarks) < 2:
                frame_landmarks.append([0.0] * 21 * 3)

            landmarks_data.append(frame_landmarks)

            # 仅在需要时写入带标记的帧
            if save_bool:
                out.write(annotated_frame)

            frame_count += 1

            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames...")

        print(f"Finished processing. Total frames: {frame_count}")
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
