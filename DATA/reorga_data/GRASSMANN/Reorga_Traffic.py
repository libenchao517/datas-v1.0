################################################################################
# 本文件用于整理Traffic数据集至Grassmann空间
################################################################################
# 导入模块
import numpy as np
from scipy.io import loadmat
from PIL import Image
from DATA.utils import def_path
from DATA.utils import save_data
from DATA.utils import save_grassmann_paras
from Grassmann import GrassmannSubSpace
from Grassmann import GrassmannDimensionality
################################################################################
## 定义基本变量
data_path_from = "trafficdb/traffic_patches_reg.mat"
target_path_from = "trafficdb/ImageMaster.mat"
target_dict = {'heavy' : 2, 'medium' : 1, 'light' : 0}
new_width = 20
new_height = 20
D = []
T = []
eta_list = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]
data = loadmat(data_path_from)
target = loadmat(target_path_from)
GS = GrassmannSubSpace()
GD = GrassmannDimensionality()
################################################################################
## 建模数据
for i in data['imgdb'][0]:
    temp = i.transpose((2, 0, 1))
    pixel_matrix = []
    for j in temp:
        image = Image.fromarray(j)
        image = image.resize((new_width, new_height))
        image = image.convert("L")
        pixel_vector = np.array(image.getdata())
        pixel_matrix.append(pixel_vector)
    D.append(np.array(pixel_matrix).T)
for t in target['imagemaster'][0]:
    T.append(target_dict.get(t[0][0][2][0]))
################################################################################
# 建模导Grassmann流形
images = D
for p in range(5, 16):
    print("Modeling Grassmann", p)
    data_name = "Traffic-" + str(p)
    data_path, target_path, target_dict_path = def_path(data_name, option="GRASSMANN")
    data = GS.compute_subspace(images, p=p)
    target = np.array(T)
    save_data(data, target, target_dict, data_path, target_path, target_dict_path)
    save_grassmann_paras(data_name=data_name, grassmann_p=p, picture_height=new_height, picture_width=new_width)
    for eta in eta_list:
        GD.ratio = eta
        GD.determine_dimensionality(data)
        GD.save_low_dimensions(data_name=data_name)
