import os
import video2txt
vidlist = []
txtlist = []
root_path = '../SLR_dataset/color'
text_path = '../SLR_dataset/seq_txt'
dir_list = os.listdir(root_path)
file_list = os.listdir(os.path.join(root_path,dir_list[1]))
print(os.path.join(root_path,dir_list[1],dir_list[1]))

for i in dir_list[2:]:
    file_list = os.listdir(os.path.join(root_path,i))
    for j in file_list:
        vidlist.append(os.path.join(root_path, i, j))
        txtlist.append(os.path.join(text_path, i, j))

def go(start_data):
    for i in range(len(vidlist)):
        if i >= start_data:
            video2txt.video2txt(vidlist[i], txtlist[i])
        print(i,vidlist[i])

go(0)
