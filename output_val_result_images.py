from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from util.visualizer import save_segment_result
from util.metrics import RunningScore
from util import util
import time
import os
import numpy as np
import torch.nn as nn

tag = "one_class"

if __name__ == '__main__':
    # 验证设置
    opt_val = TestOptions().parse()

    # 设置显示验证结果存储的设置
    result_dir = os.path.join(opt_val.checkpoints_dir, opt_val.name, 'result')
    gt_dir = os.path.join(result_dir, "gt")
    pre_dir = os.path.join(result_dir, "pre")
    img_dir = os.path.join(result_dir, "img")
    util.mkdirs([gt_dir, pre_dir, img_dir])

    # 设置验证数据集
    opt_val.batch_size = 1
    dataset_val = create_dataset(opt_val)
    dataset_val_size = len(dataset_val)
    print('The number of valling images = %d' % dataset_val_size)

    # 创建验证模型
    model_val = create_model(opt_val)
    model_val.eval()

    model_val.opt.epoch = 143 # 修改为最好模型
    model_val.setup(model_val.opt)

    for i, data in enumerate(dataset_val):
        if i > 100:
            break
        name = data["path"][0].split("/")[-1].split(".")[0]
        model_val.set_input(data)
        model_val.forward()
        pre = model_val.pre
        gt = np.squeeze(data["label"].numpy(), axis=1) 
        image = np.squeeze(data["B_img"].numpy(), axis=0) 
        image = (np.transpose(image, (1, 2, 0)) + 1) / 2.0 * 255.0
        image = image.astype(np.uint8)
        if tag == "one_class":
            pre = pre.data.max(1)[1].cpu().numpy()  # [N, W, H]
            #print(pre.shape)
            if pre.shape[0] == 1:
                gt = np.tile(gt, (3, 1, 1))
                gt = np.transpose(gt, (1, 2, 0)) * 255.0
                gt = gt.astype(np.uint8)

                pre = np.tile(pre, (3, 1, 1))
                pre = np.transpose(pre, (1, 2, 0)) * 255.0
                pre = pre.astype(np.uint8)


                # save image
                gt_path = os.path.join(gt_dir, name + '.jpg')
                pre_path = os.path.join(pre_dir, name + '.jpg')

                util.save_image(gt, gt_path)
                util.save_image(pre, pre_path)
            else:
                print("error")
        else:
            pre = pre.data.max(1)[1].cpu().numpy()  # [N, W, H]
            pre = np.transpose(pre, (1, 2, 0))
            pre = pre.astype(np.uint8)
            gt = np.transpose(gt, (1, 2, 0))
            gt = gt.astype(np.uint8)
            # save image
            gt_path = os.path.join(gt_dir, name + '.jpg')
            pre_path = os.path.join(pre_dir, name + '.jpg')
            util.save_image(gt, gt_path)
            util.save_image(pre, pre_path)
        image_path = os.path.join(img_dir, name + '.jpg')
        util.save_image(image, image_path)


