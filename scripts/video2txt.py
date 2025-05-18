import cv2
import mediapipe as mp
from pathlib import Path
from scripts.config.config import get_config
import os
os.environ['GLOG_minloglevel'] = '2'  # 禁用 MediaPipe 的 INFO/WARNING 日志
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 彻底关闭TensorFlow日志
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from itertools import islice


def process_video_batch(video_txt_batch):
    """处理一个批次的视频，共用一个 MediaPipe 实例"""
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=get_config(['mediapipe', 'static_image_mode']),
        max_num_hands=get_config(['mediapipe', 'max_num_hands']),
        model_complexity=get_config(['mediapipe', 'model_complexity']),
        min_detection_confidence=get_config(['mediapipe', 'min_detection_confidence']),
        min_tracking_confidence=get_config(['mediapipe', 'min_tracking_confidence'])
    )

    try:
        for video_path, txt_path in video_txt_batch:
            print(f"Processing: {txt_path}")
            video2txt(video_path, txt_path, hands)
    finally:
        hands.close()


def batch_iterable(iterable, batch_size):
    """将可迭代对象分成批次（生成器）"""
    it = iter(iterable)
    while True:
        batch = list(islice(it, batch_size))
        if not batch:
            return
        yield batch


def parallel_process(video_txt_pairs, batch_size=4, max_workers=None):
    """
    并行处理视频，每个 MediaPipe 实例处理 `batch_size` 个视频
    :param video_txt_pairs: 列表，每个元素是 (video_path, txt_path)
    :param batch_size: 每个 MediaPipe 实例处理的视频数量
    :param max_workers: 最大进程数（默认 CPU 核心数）
    """
    if max_workers is None or max_workers > multiprocessing.cpu_count():
        max_workers = multiprocessing.cpu_count()

    # 按 batch_size 分组，生成批次任务
    batches = list(batch_iterable(video_txt_pairs, batch_size))

    # 使用进程池处理
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        executor.map(process_video_batch, batches)

# 初始化MediaPipe手部检测模块
def video2txt(videopath, txtpath, hands=None):
    """处理单个视频文件，可接受外部初始化的hands对象"""
    # 如果没有传入hands对象，则初始化一个新的
    local_hands = False
    if hands is None:
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=get_config(['mediapipe', 'static_image_mode']),
            max_num_hands=get_config(['mediapipe', 'max_num_hands']),
            model_complexity=get_config(['mediapipe', 'model_complexity']),
            min_detection_confidence=get_config(['mediapipe', 'min_detection_confidence']),
            min_tracking_confidence=get_config(['mediapipe', 'min_tracking_confidence'])
        )
        local_hands = True  # 标记为本地初始化，需要本地释放

    mp_drawing = mp.solutions.drawing_utils

    # 确保输出目录存在
    txtpath = Path(txtpath)
    txtpath.parent.mkdir(parents=True, exist_ok=True)

    with open(txtpath, 'w') as output_file:
        cap = cv2.VideoCapture(videopath)
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)
                hand_relative_coords = []

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        root_x = hand_landmarks.landmark[0].x
                        root_y = hand_landmarks.landmark[0].y
                        root_z = hand_landmarks.landmark[0].z

                        relative_coords = []
                        for landmark in hand_landmarks.landmark:
                            relative_x = landmark.x - root_x
                            relative_y = landmark.y - root_y
                            relative_z = landmark.z - root_z
                            relative_coords.extend([relative_x, relative_y, relative_z])
                        hand_relative_coords.append(relative_coords)

                        # 如果需要可视化，取消下面注释
                        # mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                while len(hand_relative_coords) < 2:
                    hand_relative_coords.append([0] * 21 * 3)

                output_file.write(' '.join(map(str, hand_relative_coords[0])) + ' ')
                output_file.write(' '.join(map(str, hand_relative_coords[1])) + '\n')

                del frame
        finally:
            cap.release()

    # 只有本地初始化的hands对象才需要关闭
    if local_hands:
        hands.close()