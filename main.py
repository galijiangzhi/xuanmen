from scripts.config.config import get_config
import os
os.environ['GLOG_minloglevel'] = '2'
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from scripts.video2txt import parallel_process,video2txt,process_video_batch,batch_iterable
from scripts.utils import ensure_path_exists
from scripts.zero_ratio_filter import left_right_hand,list_proLR,file_statistics_data,zero_ratio_filter
from scripts.kmeans_train import qualified_data_organization,one_hand_to_two_hand,kmeans_train,kmeans_processing_data
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
    '../SLR_dataset/kmeseq_root_path/'+ 'kmeans40_双手合并/'

    ensure_path_exists() #检查相关目录是否存在，如果不存在则创建

    name_hand_model = '_双手合并' if get_config(['model', '双手合并']) else '_双手分离'
    name_multi_head_model = '_多头' if get_config(['model', '多头']) else '_标准'
    name_kmeans_n = 'kmeans'+str(get_config(['model', 'kmeans类别']))


    kmeans_model_path =  os.path.join(*[
        get_config(['path', 'kmeans_root_path']),
        name_kmeans_n,
        name_kmeans_n + name_hand_model + '.joblib'
    ])

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
        kmeans_process_save_path = os.path.join(kmeseq_root_path,name_kmeans_n+name_hand_model)
        print(kmeans_process_save_path)
        print(kmeans_model_path)
        kmeans_processing_data(kmeans_model_path,seq_path,kmeans_process_save_path,hand_model)









