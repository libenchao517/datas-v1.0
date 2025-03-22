################################################################################
# 本文件用于整理KTH数据集至Grassmann空间
################################################################################
# 导入模块
import os
import numpy as np
from DATA.utils import def_path
from DATA.utils import extract_frames
from DATA.utils import save_data
from DATA.utils import save_grassmann_paras
from Grassmann import GrassmannSubSpace
from Grassmann import GrassmannDimensionality
################################################################################
# 定义基本变量
root_path = "KTH"
target_dict = dict()
new_width = 20
new_height = 20
data = []
target = []
eta_list = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]
GS = GrassmannSubSpace()
GD = GrassmannDimensionality()
action = []
avi_name = []
frames_list = []
frames = np.zeros((600, 8), dtype=int)
################################################################################
# 整理起始和结束视频帧的坐标
with open("KTH/KTH.txt") as kth:
    for txt in kth.readlines():
        name = txt.split()
        action.append(name[0].split("_")[1])
        avi_name.append(name[0])
        frames_list.append(name[2:])
for i, frame in enumerate(frames_list):
    for j, txt in enumerate(frame):
        temp = txt.split("-")
        if "," in temp[1]:
            temp[1] = temp[1].rstrip(",")
        frames[i, 2*j + 0] = int(temp[0])
        frames[i, 2*j + 1] = int(temp[1])
frames = frames - 1
################################################################################
# 建模数据
for k, act in enumerate(np.unique(action)):
    target_dict[act] = k
for idx, element in enumerate(zip(action, avi_name, frames)):
    file_path = os.path.join(root_path, element[0], element[1] + "_uncomp.avi")
    for seq in range(4):
        start_frame, end_frame = frames[idx, 2*seq], frames[idx, 2*seq+1]
        if start_frame < end_frame:
            S = extract_frames(file_path, start_frame=start_frame, end_frame=end_frame, new_width=new_width, new_height=new_height)
            data.append(S)
            target.append(target_dict.get(element[0]))
################################################################################
# 建模导Grassmann流形
images = data
for p in range(5, 16):
    print("Modeling Grassmann", p)
    data_name = "KTH-" + str(p)
    data_path, target_path, target_dict_path = def_path(data_name, option="GRASSMANN")
    data = GS.compute_subspace(images, p=p)
    target = np.array(target)
    save_data(data, target, target_dict, data_path, target_path, target_dict_path)
    save_grassmann_paras(data_name=data_name, grassmann_p=p, picture_height=new_height, picture_width=new_width)
    for eta in eta_list:
        GD.ratio = eta
        GD.determine_dimensionality(data)
        GD.save_low_dimensions(data_name=data_name)
