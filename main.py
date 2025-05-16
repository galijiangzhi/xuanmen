from scripts.config.config import get_config
import os

print(type(get_config(['path','dataset_video_path'])))
vidlist = []
txtlist = []
root_path = get_config(['path','dataset_video_path'])
text_path = get_config(['path','dataset_seq_path'])
dir_list = os.listdir(root_path)
file_list = os.listdir(os.path.join(root_path,dir_list[1]))
print(os.path.join(root_path,dir_list[1],dir_list[1]))

for i in dir_list:
    file_list = os.listdir(os.path.join(root_path,i))
    for j in file_list:
        vidlist.append(os.path.join(root_path, i, j))
        txtlist.append(os.path.join(text_path, i, j)+'.txt')


# print(vidlist)
# print(txtlist)
#
# def go(start_data):
#     for i in range(len(vidlist)):
#         if i >= start_data:
#             video2txt.video2txt(vidlist[i], txtlist[i])
#         print(i,vidlist[i])
#
# go(0)
