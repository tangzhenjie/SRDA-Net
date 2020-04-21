import os
from PIL import Image as m
from tqdm import tqdm
import random
import cv2
import numpy as np
from util import util

# the splitted pots val dataset paths
pots_images_path_val = "./potsdam/val_origin/images"
pots_labels_path_val = "./potsdam/val_origin/labels"
pots_image_size_val = 500

class0 = np.array([255, 255, 255])
class1 = np.array([0, 0, 255])
class2 = np.array([0, 255, 255])
class3 = np.array([0, 255, 0])
class4 = np.array([255, 255, 0])
class5 = np.array([255, 0, 0])



# create the valB without overlap
def createSets(image_dir, label_dir, image_size, output_path):
    index = 1
    image_paths = os.listdir(image_dir)
    for path_item in tqdm(image_paths):
        # image = m.open(image_dir + "/" + path_item).convert('RGB')
        image = cv2.imread(image_dir + "/" + path_item, cv2.IMREAD_UNCHANGED)
        # label = m.open(label_dir + "/" + path_item.split(".")[0] + ".tif").convert("L")
        label = cv2.imread(label_dir + "/" + path_item[:-8] + "label.tif", cv2.IMREAD_UNCHANGED)
        X_height, X_width, _ = image.shape
        for row in range(12):
            start_row = row * image_size
            end_row = start_row + image_size
            for colom in range(12):
                start_colom = colom * image_size
                end_colom = start_colom + image_size

                src_roi = image[start_colom: end_colom, start_row: end_row, :]
                label_roi = label[start_colom: end_colom, start_row: end_row, :]
                # 切割图像然后保存
                cv2.imwrite((output_path + "/images/image%d.tif" % index), src_roi)
                cv2.imwrite((output_path + "/labels/label%d.tif" % index), label_roi)
                index += 1


def change_B_label(label_dir):
    image_paths = os.listdir(label_dir)
    for path_item in tqdm(image_paths):
        #label = m.open(label_dir + "/" + path_item).convert("L")
        label = cv2.imread(label_dir + "/" + path_item, cv2.IMREAD_UNCHANGED)
        # Bgr to RGB
        label = label[:, :, ::-1]
        height, width, chanel = label.shape
        label_seg = np.zeros([height, width], dtype=np.int8)
        label_seg[(label == class0).all(axis=2)] = 0
        label_seg[(label == class1).all(axis=2)] = 1
        label_seg[(label == class2).all(axis=2)] = 2
        label_seg[(label == class3).all(axis=2)] = 3
        label_seg[(label == class4).all(axis=2)] = 4
        label_seg[(label == class5).all(axis=2)] = 5
        # change 255 to 1
        #im_point = label.point(lambda x: x // 255)
        
        cv2.imwrite((label_dir + "/" + path_item), label_seg)
        #im_point.save(label_dir + "/" + path_item,'tif')

if __name__ == "__main__":
    util.mkdirs(['./vaih_pots/valB/images', './vaih_pots/valB/labels'])
    pots_output_path_val = "./vaih_pots/valB"
    createSets(pots_images_path_val, pots_labels_path_val, pots_image_size_val, pots_output_path_val)
    valB_label_dir = "./vaih_pots/valB/labels"
    change_B_label(valB_label_dir)