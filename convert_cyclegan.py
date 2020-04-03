from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util import util
from PIL import Image as m
import numpy as np
import cv2
import os


if __name__ == '__main__':
    # 加载设置
    opt = TestOptions().parse()
    opt.batch_size = 1
    # 加载数据集
    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    print('The number of training images = %d' % dataset_size)

    # 创建模型
    model = create_model(opt)

    # 恢复权重
    model.setup(opt)
    model.eval()
    output_dir = os.path.join(opt.dataroot, "fakeA_ltol")
    img_dir = os.path.join(output_dir, "images")
    label_dir = os.path.join(output_dir, "labels")
    util.mkdirs([output_dir,img_dir,label_dir])
    for i, data in enumerate(dataset):
        image_name_A = data["img_path"][0].split("/")[-1]
        image_name_B = data["label_path"][0].split("/")[-1]
        if image_name_A.split(".")[0][5:] != image_name_B.split(".")[0][5:]:
            print(image_name_A + "is not" + image_name_B)
            break
        model.set_input(data)
        model.forward()
        # result data(image label)
        fake_B = model.fake_B
        imageB_np = util.tensor2im(fake_B)[...,::-1]
        label_data = np.squeeze(data["label"].numpy())

        # result_paths(image, label)

        image_out = output_dir + "/images/" + image_name_A
        label_out = output_dir + "/labels/" + image_name_B
        # save images
        cv2.imwrite(image_out, imageB_np)
        cv2.imwrite(label_out, label_data)
        print(i + 1)


