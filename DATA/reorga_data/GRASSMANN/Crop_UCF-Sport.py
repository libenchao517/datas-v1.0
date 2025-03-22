################################################################################
## 将UCF-Sport数据集进行切割
################################################################################
## 导入模块
import os
import re
import cv2
import numpy as np
from PIL import Image
################################################################################
## 定义函数
def video_to_frames(video_path, output_dir):
    video_filename = os.path.splitext(os.path.basename(video_path))[0]
    match = re.match(r'^(.*?)(\d+)$', video_filename)
    base_name = match.group(1)
    start_number = int(match.group(2))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        current_number = start_number + frame_count
        frame_filename = os.path.join(output_dir, f'{base_name}{current_number}.jpg')
        cv2.imwrite(frame_filename, frame)
        frame_count += 1
    cap.release()
################################################################################
## 定义基本变量
n=0
root_path = "ucf_sports_actions"
annotation_path = "UCF-Sports-annotations-master"
################################################################################
## 整理图片
categories = [item for item in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, item))]
for cate in categories:
    cate_path = os.path.join(root_path, cate)
    objects = [item for item in os.listdir(cate_path) if os.path.isdir(os.path.join(cate_path, item))]
    for object in objects:
        object_path = os.path.join(cate_path, object)
        gt_path = os.path.join(object_path, "gt")
        jpeg_path = os.path.join(object_path, "jpeg")
        jpg_files = [item for item in os.listdir(object_path) if re.match(r".*.jpg", item)]
        avi_files = [item for item in os.listdir(object_path) if re.match(r".*.avi", item)]
        if len(np.array(jpg_files)) == 0:
            avi_path = os.path.join(object_path, avi_files[0])
            video_to_frames(avi_path, object_path)
        if not os.path.exists(jpeg_path):
            os.makedirs(jpeg_path)
            anno_path = os.path.join(annotation_path, cate, object, "gt")
            jpg_files = [item for item in os.listdir(object_path) if re.match(r".*.jpg", item)]
            txt_files = [item for item in os.listdir(anno_path) if re.match(r".*.txt", item)]
            for jpg, txt in zip(jpg_files, txt_files):
                jpg_path = os.path.join(object_path, jpg)
                txt_path = os.path.join(anno_path, txt)
                gt = list(map(int, open(txt_path).read().split()[:4]))
                box = (gt[0], gt[1], gt[0] + gt[2], gt[1] + gt[3])
                image = Image.open(jpg_path)
                crop_image = image.crop(box)
                save_path = os.path.join(jpeg_path, jpg)
                print(save_path)
                crop_image.save(save_path)
