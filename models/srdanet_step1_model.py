import torch
import itertools
from .base_model import BaseModel
from . import networks
from torch import nn
from util.image_pool import ImagePool
import torch.nn.functional as F
import util.loss as loss

class SrdanetStep1Model(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.
        """
        parser.add_argument('--num_classes', type=int, default=2, help='for determining the class number')
        return parser
    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ["G", "D"]    # "loss_"
        self.visual_names = ["imageA", "fakeB", "imageA_up", "imageB", "pixelfakeB_out"]  # ""    , "fcreal_out"

        self.model_names = ['generator', 'pixel_discriminator']  # "net"

        # 特征生成器
        self.netgenerator = networks.srdanet_generator(num_cls=opt.num_classes, gpu_ids=self.gpu_ids)

        # 像素空间判别器
        self.netpixel_discriminator = networks.define_D(3, 64, 'basic', norm="instance", gpu_ids=self.gpu_ids)

        # 像素判别器损失
        self.mse_loss = networks.GANLoss("lsgan").to(self.device)

        # idt 损失
        self.generator_criterion = networks.GeneratorLoss().to(self.device)

        # 内容一致损失
        self.L1_loss = nn.L1Loss().to(self.device)

        # 优化器
        self.optimizer = torch.optim.Adam(self.netgenerator.parameters(),
                                          lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=0.0005)
        self.optimizer_D = torch.optim.Adam(self.netpixel_discriminator.parameters(),
                                            lr=opt.lr_D, betas=(opt.beta1, 0.999), weight_decay=0.0005)
        self.optimizers.append(self.optimizer)
        self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        self.imageA = input["A_img"].to(self.device)  #[-1, 1]
        self.imageA_up = input["A_img_up"].to(self.device) #[-1, 1]
        self.imageB = input["B_img"].to(self.device) #[-1, 1]

    def forward(self):
        # iamgeA 通过 generator
        self.feature_A, self.pre, self.fakeB = self.netgenerator(self.imageA)
        _, _, h, w = self.imageA_up.size()
        self.fakeB = nn.functional.interpolate(self.fakeB, mode="bilinear", size=(h, w), align_corners=True)
        self.fakeB = F.tanh(self.fakeB)
        _, _, h1, w1 = self.imageA.size()
        self.fakeB_down = nn.functional.interpolate(self.fakeB, size=(h1, w1))

        self.fakeB_cut = self.fakeB.detach() # 隔断反向传播
        self.fakeB_down_cut = self.fakeB_down.detach() # 隔断反向传播
        self.feature_fakeB_down_cut, self.pre_aux, _ = self.netgenerator(self.fakeB_down_cut) # 还没有使用

        # imagesrA 通过判别器
        self.pixelfakeB_out = self.netpixel_discriminator(self.fakeB)

    def backward(self):
        """计算两个损失"""

        # 像素级对齐损失
        self.loss_da = self.mse_loss(self.pixelfakeB_out, True)

        # A内容一致性损失
        self.loss_idtA = self.generator_criterion(self.fakeB, self.imageA_up, is_sr=False)

        # fix_pointA loss
        self.loss_fix_point = self.L1_loss(self.feature_A, self.feature_fakeB_down_cut)

        loss_DA = self.loss_da
        loss_ID = self.loss_idtA + self.loss_fix_point * 0.5

        # 求分割损失和超分辨损失的和
        self.loss_G = loss_DA * 2 + loss_ID * 10
        self.loss_G.backward(retain_graph=True)

    def backward_D(self):

        pixeltrueB_out = self.netpixel_discriminator(self.imageB)

        self.loss_D_da = self.mse_loss(self.netpixel_discriminator(self.fakeB_cut), False) \
                          + self.mse_loss(pixeltrueB_out, True)

        self.loss_D = self.loss_D_da * 1
        self.loss_D.backward()

    def optimize_parameters(self):
        self.forward()

        # 不求判别器的梯度
        self.set_requires_grad([self.netpixel_discriminator], False)

        # 更新生成器的参数
        self.optimizer.zero_grad()
        self.backward()  # 计算生成器的参数的梯度
        self.optimizer.step()  # 更新参数

        # 可以求判别器的梯度
        self.set_requires_grad([self.netpixel_discriminator], True)
        self.optimizer_D.zero_grad()
        self.backward_D()  # 计算判别器的梯度
        self.optimizer_D.step()  # update weights

