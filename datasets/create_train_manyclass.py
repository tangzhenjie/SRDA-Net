import os
from PIL import Image as m
from tqdm import tqdm
import random
import cv2
import numpy as np
from util import util

# vaih dataset paths
vai_images_path = "./vaihingen/images"
vai_labels_path = "./vaihingen/labels"
vai_image_size = 500

# the splitted pots training dataset paths
pots_images_path_train = "./potsdam/train_origin/images"
pots_labels_path_train = "./potsdam/train_origin/labels"
pots_image_size = 1000

# map(RGB -> label)
class0 = np.array([255, 255, 255])
class1 = np.array([0, 0, 255])
class2 = np.array([0, 255, 255])
class3 = np.array([0, 255, 0])
class4 = np.array([255, 255, 0])
class5 = np.array([255, 0, 0])


def createSetsA(image_dir, label_dir, image_size, output_path):
    index = 1
    label_paths = os.listdir(label_dir)
    for path_item in tqdm(label_paths):
        # image = m.open(image_dir + "/" + path_item).convert('RGB')
        image = cv2.imread(image_dir + "/" + path_item, cv2.IMREAD_UNCHANGED)
        # label = m.open(label_dir + "/" + path_item.split(".")[0] + ".tif").convert("L")
        label = cv2.imread(label_dir + "/" + path_item, cv2.IMREAD_UNCHANGED)
        X_height, X_width, _ = image.shape
        for key in range(80):
            random_width = random.randint(0, X_width - image_size - 1)
            random_height = random.randint(0, X_height - image_size - 1)
            src_roi = image[random_height: random_height + image_size, random_width: random_width + image_size, :]
            label_roi = label[random_height: random_height + image_size, random_width: random_width + image_size, :]

            cv2.imwrite((output_path + "/images/image%d.tif" % index), src_roi)
            cv2.imwrite((output_path + "/labels/label%d.tif" % index), label_roi)
            index += 1
def createSetsB(image_dir, label_dir, image_size, output_path):
    index = 1
    image_paths = os.listdir(image_dir)
    for path_item in tqdm(image_paths):
        # image = m.open(image_dir + "/" + path_item).convert('RGB')
        image = cv2.imread(image_dir + "/" + path_item, cv2.IMREAD_UNCHANGED)
        # label = m.open(label_dir + "/" + path_item.split(".")[0] + ".tif").convert("L")
        label = cv2.imread(label_dir + "/" + path_item[:-8] + "label.tif", cv2.IMREAD_UNCHANGED)
        X_height, X_width, _ = image.shape
        for row in range(6):
            start_row = row * image_size
            end_row = start_row + image_size
            for colom in range(6):
                start_colom = colom * image_size
                end_colom = start_colom + image_size

                src_roi = image[start_colom: end_colom, start_row: end_row, :]
                label_roi = label[start_colom: end_colom, start_row: end_row, :]

                cv2.imwrite((output_path + "/images/image%d.tif" % index), src_roi)
                cv2.imwrite((output_path + "/labels/label%d.tif" % index), label_roi)
                index += 1


def change_label(label_dir):
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

    # create the paths of created datasets
    util.mkdirs(['./vaih_pots/trainA/images', './vaih_pots/trainA/labels',
                 './vaih_pots/trainB/images', './vaih_pots/trainB/labels'])
    vai_output_path = "./vaih_pots/trainA"
    createSetsA(vai_images_path, vai_labels_path, vai_image_size, vai_output_path)


    pots_output_path_train = "./vaih_pots/trainB"
    createSetsB(pots_images_path_train, pots_labels_path_train, pots_image_size, pots_output_path_train)
      
    trainA_label_dir = "./vaih_pots/trainA/labels"
    change_label(trainA_label_dir)
    trainB_label_dir = "./vaih_pots/trainB/labels"
    change_label(trainB_label_dir)