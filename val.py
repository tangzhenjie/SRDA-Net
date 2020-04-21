from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_segment_result
from util.metrics import RunningScore
import time
import numpy as np

if __name__ == '__main__':

    opt_val = TestOptions().parse()

    dataset_val = create_dataset(opt_val)
    dataset_val_size = len(dataset_val)
    print('The number of valling images = %d' % dataset_val_size)

    model_val = create_model(opt_val)
    model_val.eval()

    metrics = RunningScore(opt_val.num_classes)

    model_val.opt.epoch = 114
    model_val.setup(model_val.opt)

    for i, data in enumerate(dataset_val):
        model_val.set_input(data)
        model_val.forward()
        gt = np.squeeze(data["label"].numpy(), axis=1)  # [N, W, H]
        pre = model_val.pre
        pre = pre.data.max(1)[1].cpu().numpy()  # [N, W, H]
        metrics.update(gt, pre)

    val_class_iou, iu = metrics.get_scores()
    best_result = val_class_iou[1]
    print("IOU:" + str(val_class_iou))






