import os
import video2txt

root_path = '../SLR_dataset/color'
text_path = '../SLR_dataset/seq_txt'
dir_list = os.listdir(root_path)
file_list = os.listdir(os.path.join(root_path,dir_list[1]))
print(os.path.join(root_path,dir_list[1],dir_list[1]))

for i in dir_list:
    file_list = os.listdir(os.path.join(root_path, i))
    for j in file_list:
        print(os.path.join(root_path, i, j))
        video2txt.video2txt(os.path.join(root_path, i, j), os.path.join(text_path, i, j))