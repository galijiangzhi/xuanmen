# Copyright (C) 2024 "白稹" (GitHub: @galijiangzhi)
# SPDX-License-Identifier: AGPL-3.0-only

import os
from scripts.config.config import get_config

path_list = [
    get_config(['path','dataset_video_path']),
    get_config(['path','dataset_seq_path']),
    get_config(['path','kmeans_root_path']),
    get_config(['path','kmeseq_root_path'])

]

def ensure_path_exists(path_list=path_list):
    """
    确保列表中的路径存在，如果不存在则创建该文件夹

    参数:
    path_list (list): 包含路径字符串的列表，可以是绝对路径或相对路径

    返回:
    None
    """
    for path in path_list:
        normalized_path = os.path.normpath(path)

        if not os.path.exists(normalized_path):
            try:
                os.makedirs(normalized_path)
                print(f"目录已创建: {normalized_path}")
            except OSError as e:
                print(f"创建目录失败: {normalized_path}, 错误: {e}")
        else:
            print(f"目录已存在: {normalized_path}")

