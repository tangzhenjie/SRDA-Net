import os
from PIL import Image as m
from tqdm import tqdm
import random
import cv2
import numpy as np
from util import util

# the splitted pots val dataset paths
inria_images_path_val = "./inria/val/images"
inria_labels_path_val = "./inria/val/labels"
inria_image_size_val = 625

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
        for row in range(8):
            start_row = row * image_size
            end_row = start_row + image_size
            for colom in range(8):
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
        label = m.open(label_dir + "/" + path_item).convert("L")

        # change 255 to 1
        im_point = label.point(lambda x: x // 255)

        im_point.save(label_dir + "/" + path_item, 'tif')


if __name__ == "__main__":
    util.mkdirs(['./mass_inria/valB/images', './mass_inria/valB/labels'])
    inria_output_path_val = "./mass_inria/valB"
    createSets(inria_images_path_val, inria_labels_path_val, inria_image_size_val, inria_output_path_val)
    valB_label_dir = "./mass_inria/valB/labels"
    change_B_label(valB_label_dir)