################################################################################
## 本代码用于将UT-Kinect数据集建模至Grassmann空间
################################################################################
## 导入模块
import os
import numpy as np
from DATA.utils import def_path
from DATA.utils import collect_images
from DATA.utils import save_data
from DATA.utils import save_grassmann_paras
from Grassmann import GrassmannSubSpace
from Grassmann import GrassmannDimensionality
################################################################################
# 定义基本变量
root_path = "UT-Kinect"
target_dict = {"walk" : 0, "sitDown" : 1, "standUp" : 2, "pickUp" : 3, "carry" : 4, "throw" : 5, "push" : 6, "pull" : 7, "waveHands" : 8, "clapHands" : 9}
new_width = 20
new_height = 20
data = []
target = []
eta_list = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]
GS = GrassmannSubSpace()
GD = GrassmannDimensionality()
frame_list = []
################################################################################
# 整理起始和结束视频帧的坐标
with open("UT-Kinect/UT-Kinect-labels.txt") as file:
    for txt in file.readlines():
        txt = txt.rstrip("\n")
        frame_list.append(txt.split())
################################################################################
# 建模数据
for i in range(20):
    indexs = frame_list[i * 11][0]
    avi_path = os.path.join(root_path, indexs)
    for j in range(10):
        action = frame_list[i * 11 + j + 1][0][:-1]
        start, end = int(frame_list[i * 11 + j + 1][1]), int(frame_list[i * 11 + j + 1][2])
        files = []
        for k in range(start, end + 1):
            if os.path.exists(os.path.join(avi_path, "colorImg" + str(k) + ".jpg")):
                files.append("colorImg" + str(k) + ".jpg")
        if np.array(files).shape[0] > 0:
            S = collect_images(path=avi_path, file_list=files, new_width=new_width, new_height=new_height)
            data.append(S)
            target.append(target_dict.get(action))
################################################################################
# 建模导Grassmann流形
images = data
for p in range(5, 16):
    print("Modeling Grassmann", p)
    data_name = "UT-Kinect-" + str(p)
    data_path, target_path, target_dict_path = def_path(data_name, option="GRASSMANN")
    data = GS.compute_subspace(images, p=p)
    target = np.array(target)
    save_data(data, target, target_dict, data_path, target_path, target_dict_path)
    save_grassmann_paras(data_name=data_name, grassmann_p=p, picture_height=new_height, picture_width=new_width)
    for eta in eta_list:
        GD.ratio = eta
        GD.determine_dimensionality(data)
        GD.save_low_dimensions(data_name=data_name)
