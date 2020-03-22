import torch
import itertools
from .base_model import BaseModel
from . import networks
from torch import nn


class Deeplabv2BaselineModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.
        """
        parser.add_argument('--num_classes', type=int, default=2, help='for fcn determining the class number')
        parser.add_argument('--is_restore_from_imagenet', type=bool, default=True,
                            help='for fcn determining whether or not to restore resnet50 from imagenet')
        parser.add_argument('--resnet_weight_path', type=str, default='./resnetweight/',
                            help='the path to renet_weight from imagenet')

        return parser
    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ["cross_entropy"]    # "loss_"
        self.visual_names = ["image", "label", "prediction"]  # ""

        self.model_names = ['backbone']  # "net"

        self.netbackbone = networks.deeplabv2(is_restore_from_imagenet=opt.is_restore_from_imagenet,
                                                             resnet_weight_path=opt.resnet_weight_path,
                                                             num_classes = opt.num_classes,
                                                             gpu_ids=self.gpu_ids)

        if self.isTrain:

            # 语义分割损失
            self.loss_function = nn.CrossEntropyLoss().to(self.device)

            # 优化器
            self.optimizer = torch.optim.Adam(self.netbackbone.module.get_parameters(opt),
                                                lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=0.0005)
            self.optimizers.append(self.optimizer)

    def set_input(self, input):
        self.image = input["img"].to(self.device)
        self.label = input["label"].to(self.device)

    def forward(self):
        # iamge 通过：特征提取器网络
        self.pre = self.netbackbone(self.image)

        self.prediction = self.pre.data.max(1)[1].unsqueeze(1)

    def backward(self):
        """计算损失：image_fakeB分割损失"""

        # image_fakeB分割损失
        self.loss_cross_entropy = self.loss_function(self.pre, self.label.long().squeeze(1))
        self.loss_cross_entropy.backward()

    def optimize_parameters(self):
        self.forward()
        # 更新参数
        self.optimizer.zero_grad()
        self.backward()   # 计算参数的梯度
        self.optimizer.step() # 更新参数