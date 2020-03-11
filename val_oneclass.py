from options.train_options import TrainOptions
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

best_result = 0

if __name__ == '__main__':
    # 验证设置
    opt_val = TestOptions().parse()

    # 设置显示验证结果存储的设置
    web_dir = os.path.join(opt_val.checkpoints_dir, opt_val.name, 'val')
    image_dir = os.path.join(web_dir, 'images')
    util.mkdirs([web_dir, image_dir])

    # 设置验证数据集
    dataset_val = create_dataset(opt_val)
    dataset_val_size = len(dataset_val)
    print('The number of valling images = %d' % dataset_val_size)

    # 创建验证模型
    model_val = create_model(opt_val)
    model_val.eval()

    # 训练设置
    opt_train = TrainOptions().parse()

    # 设置显示训练结果的类
    visualizer = Visualizer(opt_train)
    for epoch in range(opt_train.epoch_count, opt_train.niter + opt_train.niter_decay + 1):
        epoch_iters = 0
        epoch_start_time = time.time()

        # 验证结果
        metrics = RunningScore(opt_val.num_classes)

        model_val.opt.epoch = epoch
        model_val.setup(model_val.opt)

        for i, data in enumerate(dataset_val):
            model_val.set_input(data)
            model_val.forward()
            gt = np.squeeze(data["label"].numpy(), axis=1)  # [N, W, H]
            pre = model_val.pre
            pre = pre.data.max(1)[1].cpu().numpy()  # [N, W, H]
            metrics.update(gt, pre)
            # 保存结果
            if i % opt_train.display_freq == 0:  # 逻辑有点问题
                save_segment_result(model_val.get_current_visuals(), epoch, opt_train.display_winsize, image_dir,
                                    web_dir, opt_train.name)
        val_class_iou, iu = metrics.get_scores()
        if best_result < val_class_iou[1]:
            best_result = val_class_iou[1]
            with open(web_dir + "best_result.txt", mode="w+") as f:
                f.write("epoch" + str(epoch) + ":" + str(val_class_iou) + " best_mean_iou:" + str(best_result))

        # 一个epoch 改变一次学习率
        print('End of epoch %d / %d \t Time Taken: %d sec' % (
            epoch, opt_train.niter + opt_train.niter_decay, time.time() - epoch_start_time))





