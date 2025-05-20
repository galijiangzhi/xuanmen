# Copyright (C) 2024 "白稹" (GitHub: @galijiangzhi)
# SPDX-License-Identifier: AGPL-3.0-only

import os
import numpy as np
from joblib import dump, load
from scripts.zero_ratio_filter import list_proLR,left_right_hand

import os
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from itertools import chain
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
import concurrent.futures


def predict_with_outlier(X, threshold=0.1):
    # distances = kmeans_loaded.transform(X)  # 计算到所有中心的距离
    # min_distances = np.min(distances, axis=1)

    # 分类结果（离群点设为0）
    labels = kmeans_loaded.predict(X)
    # labels[min_distances > threshold] = 999  # 阈值需根据数据分布调整

    return labels

def kmeans_processing_data(model_path,data_root_path,data_save_root_path,hand_model):
    '''
    :param model_path: kmeans的模型路径 '../model/kmeans/kmeans_40_双手合并.joblib'
    :param data_root_path:  全部数据的根路径 '../SLR_dataset/seq_txt/'
    :param data_save_root_path:  数据的保存路径 '../SLR_dataset/kmeans_40_seq_双手合并_多头/'
    :param hand_model: 双手是否合并
    :return:
    '''
    kmeans_loaded = load(model_path)  # 分类文件夹根目录
    all_path = data_root_path
    save_path = data_save_root_path
    print(save_path)
    all_folder = [i for i in os.listdir(all_path)] #获取全部的数据文件夹0-99
    quchong = False  # 去重

    def predict_with_outlier(X, threshold=0.1):
        # distances = kmeans_loaded.transform(X)  # 计算到所有中心的距离
        # min_distances = np.min(distances, axis=1)

        # 分类结果（离群点设为0）
        labels = kmeans_loaded.predict(X)
        # labels[min_distances > threshold] = 999  # 阈值需根据数据分布调整

        return labels

    maxlen = 0
    for i in all_folder: #循环每个文件夹
        all_new_labels = [] #全部新数据
        folder = os.path.join(all_path, i) #获取当前处理的文件夹的路径
        # print(f'正在处理文件夹{folder}')
        file_name = [file for file in os.listdir(folder) if file.endswith('txt')]
        for j in file_name: #遍历每个文件
            file_path = os.path.join(folder, j) #获取当前文件的路径
            datalist = []
            with open(file_path, 'r') as file:  # 以读取模式打开文件
                for line in file:  # 逐行读取文件内容
                    # 按空格分割每一行，并将每个部分转换为浮点型
                    if hand_model:
                        float_values = [float(x) for x in line.split()]
                        datalist.append(float_values)  # 将转换后的浮点型列表添加到 datalist
                    else:
                        float_values = [float(x) for x in line.split()]
                        datalist.append(float_values[:63])  # 将转换后的浮点型列表添加到 datalist
                        datalist.append(float_values[63:])  # 将转换后的浮点型列表添加到 datalist
            new_labels = predict_with_outlier(datalist)
            # print(new_labels)
            all_new_labels.append(new_labels)#视频数量*视频帧数*每一帧的数据量
        # all_new_labels = np.array(all_new_labels)

        # print(f'当前文件夹下的视频数量:{len(all_new_labels)},当前文件夹下第一个视频的帧数：{len(all_new_labels[0])}')
        # print(f'当前文件夹下第一个视频第一帧的数据：{len(all_new_labels[0])}')

        save_file_name = os.path.join(save_path, i) + '.npy'
        save_data = np.array(all_new_labels, dtype=object)
        os.makedirs(os.path.dirname(save_file_name), exist_ok=True)
        np.save(save_file_name, save_data)
        print(save_file_name)
    # print(maxlen)

def kmeans_train(datapath,kmeans_n,kmeans_batch,save_path,random_state=160916):
    # 以内存映射方式加载 .npy 文件
    data = np.load(datapath, mmap_mode='r')

    # 初始化 MiniBatchKMeans
    n_clusters = int(kmeans_n) #分的类别数
    batch_size = int(kmeans_batch)  # 每批读取和处理的样本数
    print(f'开始进行mini batch kmeans训练，类别数：{n_clusters}，批次大小：{batch_size}')
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, random_state=random_state)

    # 分批读取数据并拟合模型
    num_samples = data.shape[0]  # 总样本数
    for i in range(0, num_samples, batch_size):
        batch = data[i:i + batch_size]  # 读取当前批次的数据
        kmeans.partial_fit(batch)  # 使用当前批次的数据更新模型

        # 每处理 30 个批次打印一次信息
        if (i // batch_size + 1) % 30 == 0:
            print(f"已处理 {i // batch_size + 1} 个批次，当前批次形状: {batch.shape}")

    # 获取聚类结果
    labels = kmeans.labels_  # 每个样本的聚类标签
    centers = kmeans.cluster_centers_  # 聚类中心

    # 输出结果
    print("聚类标签:", labels)
    print("聚类中心:", centers)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    dump(kmeans, save_path)

def process_file(file):
    """处理单个文件，返回两种格式的数据"""
    with open(file, 'r') as f:
        data = []
        for line in f:
            float_values = [float(x) for x in line.split()]
            data.append(float_values)

    # fenli_data = list_proLR(data, dimension=1)
    hebing_data = list_proLR(data, True, dimension=1)
    # return fenli_data, hebing_data
    return hebing_data

def process_folder(folder):
    """处理单个文件夹中的所有文件"""
    print('正在处理文件夹',folder)
    data_list = [os.path.join(folder, filename)
                 for filename in os.listdir(folder)
                 if filename.endswith('.txt')]

    # folder_fenli = []
    folder_hebing = []
    for file in data_list:
        # fenli, hebing = process_file(file)
        # folder_fenli.extend(fenli)
        hebing = process_file(file)
        # folder_fenli.extend(fenli)
        folder_hebing.extend(hebing)
    # return folder_fenli, folder_hebing
    return folder_hebing


def qualified_data_organization(seq_path, hebing_save_path, workers=None):
    """使用多进程优化的数据组织函数"""
    all_path = seq_path
    folder_list = [os.path.join(all_path, i) for i in os.listdir(all_path)]

    # 设置工作进程数，默认为CPU核心数
    if workers is None or workers > cpu_count():
        workers = cpu_count()

    # 使用进程池并行处理文件夹
    with Pool(processes=workers) as pool:
        results = pool.map(process_folder, folder_list)

    # 合并所有结果
    # fenli_data = []
    hebing_data = []
    for folder_hebing in results:
        # fenli_data.extend(folder_fenli)
        hebing_data.extend(folder_hebing)

    # # 保存分离数据
    # fenli_array = np.array(fenli_data)
    # np.save(fenli_save_path, fenli_array)
    # print(f"分离数据形状: {fenli_array.shape}")

    # 保存合并数据
    hebing_array = np.array(hebing_data)
    np.save(hebing_save_path, hebing_array)
    print(f"合并数据形状: {hebing_array.shape}")

def one_hand_to_two_hand(one_hand_path, two_hand_path):
    data = np.load(one_hand_path)  # 形状 [10000, 126]
    # 2. 计算新的行数 x = (总元素数) / 63
    x = data.size // 63  # 10000*126 / 63 = 20000
    print(x)
    # # 3. 重塑为 [x, 63]
    reshaped_data = data.reshape(x, 63)

    # # 4. 保存新数据（可选）
    np.save(two_hand_path, reshaped_data)