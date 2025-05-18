import os
import numpy as np
import matplotlib.pyplot as plt

def left_right_hand(data):
    left = data[0:63]
    right = data [63:]
    return left,right

def list_proLR(data,merging=False,dimension=1):
    #该函数用于将读取到的数据变成左右手格式
    result = []
    if merging: #如果合并则之间添加
        for i in data:
            result.append(i)
    else:
        for i in data: #否则双手分离添加
            if dimension == 1:
                left, right = left_right_hand(i)
                result.append(left)
                result.append(right)
            else:
                left,right = left_right_hand(i)
                result.append([left,right])

    return result


def file_statistics_data(data):
    # 该函数输入分好左右手的数据，输出每个视频的有效率
    yes_data = 0
    no_data = 0
    for i in data:
        left = i[0]
        right = i[1]
        # print(left,right)
        if (len(set(left)) == 1) and (len(set(right)) == 1):
            no_data += 1
        else:
            yes_data += 1

    return yes_data / (yes_data + no_data)

def zero_ratio_filter(data_path,ratio = 0.74):
    folder_list = os.listdir(data_path)
    folder_list = [os.path.join(data_path, i) for i in folder_list]
    for folder in folder_list:
        data_list = [os.path.join(folder,filename) for filename in os.listdir(folder) if filename.endswith('.avi')]
        for file in data_list:
            # print(file)
            data = []
            with open(file,'r') as f:
                for line in f:
                    float_values = [float(x) for x in line.split()]
                    data.append(float_values)
            # print(len(data)) #二维数组 帧*126
            # print(data[0:10])
            data = list_proLR(data)
            statistice=file_statistics_data(data)
            if statistice < ratio:
                # os.remove(file)
                print('删除零值过高数据：',file,'零值比例：',statistice)