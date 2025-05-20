# Copyright (C) 2024 "白稹" (GitHub: @galijiangzhi)
# SPDX-License-Identifier: AGPL-3.0-only

from scripts.config.config import get_config
import os
os.environ['GLOG_minloglevel'] = '2'
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from scripts.video2txt import parallel_process,video2txt,process_video_batch,batch_iterable
from scripts.utils import ensure_path_exists
from scripts.zero_ratio_filter import left_right_hand,list_proLR,file_statistics_data,zero_ratio_filter
from scripts.kmeans_train import qualified_data_organization,one_hand_to_two_hand,kmeans_train,kmeans_processing_data
from scripts.xuanmen_train import xuanmen_train
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)  # 只显示 ERROR 级别日志



if __name__ == '__main__':
    #读取配置文件中的关键参数
    root_path = get_config(['path', 'dataset_video_path'])
    seq_path = get_config(['path', 'dataset_seq_path'])
    max_woekers = get_config(['计算参数', 'max_workers'])
    data_cleaningzero_ratio = get_config(['数据清洗', '零值比例阈值'])
    all_data_hebing = get_config(['path', '双手合并全部数据npy_path'])
    all_data_fenli = get_config(['path', '双手分离全部数据npy_path'])
    kmeseq_root_path = get_config(['path', 'kmeseq_root_path'])
    kmeans_n = get_config(['model', 'kmeans类别'])
    kmeans_train_batch = get_config(['model', 'kmeans训练批次大小'])
    hand_model = get_config(['model', '双手合并'])
    multi_head_bool = get_config(['model', '多头'])
    # '../SLR_dataset/kmeseq_root_path/'+ 'kmeans40_双手合并/'

    ensure_path_exists() #检查相关目录是否存在，如果不存在则创建

    name_hand_model = '_双手合并' if get_config(['model', '双手合并']) else '_双手分离'
    name_multi_head_model = '_多头' if get_config(['model', '多头']) else '_标准'
    name_kmeans_n = 'kmeans'+str(get_config(['model', 'kmeans类别']))
    name_emb = '_emb'+str(get_config(['model', '词向量维度']))
    name_hid = '_hid' + str(get_config(['model', 'LSTM隐藏层维度']))
    '''关键路径'''
    kmeans_process_save_path = os.path.join(kmeseq_root_path, name_kmeans_n + name_hand_model) #kmeans处理之后的数据的保存地址
    # './SLR_dataset/kme_seq',kmeans40_双手合并
    train_data_path = kmeans_process_save_path #训练数据集的路径
    train_data_label_path = get_config(['path', '数据文件夹标签path']) #训练数据集的标签文件路径
    xuanmen_model_root_path = get_config(['path', 'xuanmen_model_root_path']) #模型根路径 './SLR_dataset/model/'
    xuanmen_model_save_path = os.path.join(xuanmen_model_root_path,
                                           'xuanmen'+name_kmeans_n,
                                           name_kmeans_n+name_emb+name_hid+name_hand_model+name_multi_head_model+'.pth')
    # './SLR_dataset/model/','xuanmenkmean40',name_kmeans_n+name_emb+name_hid+name_hand_model+name_multi_head_model+'.pth'


    kmeans_model_path =  os.path.join(*[
        get_config(['path', 'kmeans_root_path']),
        name_kmeans_n,
        name_kmeans_n + name_hand_model + '.joblib'
    ])

    '''模型相关参数'''

    model_input_dim = kmeans_n + 2 #输入词汇表大小，
    model_emb_dim = get_config(['model', '词向量维度'])
    model_hid_dim = get_config(['model', 'LSTM隐藏层维度'])
    model_n_layer = get_config(['model', 'n_layers'])
    model_num_head = get_config(['model', 'num_heads'])
    model_seq_len = get_config(['model', '不抽帧双手分离seq最大长度'])
    model_sampling_interval = get_config(['model', '取样区间'])
    model_hand_model = hand_model
    model_epoch = get_config(['train', 'epoch'])
    model_batch_size = get_config(['train', 'batch_size'])

    if get_config(['流程','video2txt']):
        # print(type(get_config(['path','dataset_video_path'])))
        vidlist = []
        txtlist = []
        dir_list = os.listdir(root_path)
        file_list = os.listdir(os.path.join(root_path, dir_list[1]))
        print(os.path.join(root_path, dir_list[1], dir_list[1]))

        for i in dir_list:
            file_list = os.listdir(os.path.join(root_path, i))
            for j in file_list:
                vidlist.append(os.path.join(root_path, i, j))
                txtlist.append(os.path.join(seq_path, i, j) + '.txt')

        video_txt_pairs = list(zip(vidlist, txtlist))
        parallel_process(video_txt_pairs, batch_size=len(video_txt_pairs)//max_woekers,max_workers=max_woekers)
        print('视频序列转数字序列完成')
    if get_config(['流程','数据零值比例过滤']):
        zero_ratio_filter(seq_path,data_cleaningzero_ratio)
        # 整合合理数据
    if get_config(['流程', '整合合理数据']):
        if os.path.exists(all_data_hebing):
            print('双手合并全部数据以整理为:',all_data_hebing)
        else:
            print('双手合并全部数据未整理，正在整理请稍后')
            qualified_data_organization(seq_path,all_data_hebing,max_woekers)
        if os.path.exists(all_data_fenli):
            print('双手分离全部数据以整理为:',all_data_fenli)
        else:
            print('双手分离全部数据未整理，正在整理请稍后')
            one_hand_to_two_hand(all_data_hebing,all_data_fenli)
    if get_config(['流程', 'kmeans模型训练']):
        kmeans_save_path = [
            get_config(['path', 'kmeans_root_path']),
            'kmeans'+str(get_config(['model', 'kmeans类别']))
        ]
        if get_config(['model', '双手合并']):
            all_data_path = all_data_hebing
            # kmeans_save_path.append('kmeans'+str(get_config(['model', 'kmeans类别']))+name_hand_model+'.joblib')
        else:
            all_data_path = all_data_fenli
            # kmeans_save_path.append('kmeans'+str(get_config(['model', 'kmeans类别']))+name_hand_model+'.joblib')
        # kmeans_save_path=os.path.join(*kmeans_save_path)
        kmeans_train(all_data_path,kmeans_n,kmeans_train_batch,kmeans_model_path)
    if get_config(['流程', 'kmeans对原数据进行处理']):
        print(kmeans_process_save_path)
        print(kmeans_model_path)
        kmeans_processing_data(kmeans_model_path,seq_path,kmeans_process_save_path,hand_model)
    if get_config(['流程','xuanmen网络训练']):
       datapath =  "../SLR_dataset/kmeans_80_seq_frame5_双手合并/"
       xuanmen_train(train_data_path=train_data_path,
                     train_data_label_path=train_data_label_path,
                     multi_head_bool=multi_head_bool,
                     model_path=xuanmen_model_save_path,
                     input_dim=model_input_dim,
                     emb_dim=model_emb_dim,
                     hidden_dim=model_hid_dim,
                     hand_model= model_hand_model,
                     seq_len= model_seq_len,
                     sampling_interval = model_sampling_interval,
                     train_epoch = model_epoch,
                     n_layers=model_n_layer,
                     batch_size= model_batch_size,
                     num_heads=model_num_head)













