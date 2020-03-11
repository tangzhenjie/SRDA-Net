import torch
import itertools
from .base_model import BaseModel
from . import networks
from torch import nn
from util.image_pool import ImagePool
import torch.nn.functional as F
import util.loss as loss

class SrdanetStep2Model(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.
        """
        parser.add_argument('--num_classes', type=int, default=2, help='for determining the class number')
        return parser
    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ["G", "D"]    # "loss_"
        self.visual_names = ["imageB_down", "label", "prediction"]  # ""
        if self.isTrain:
            self.visual_names += ["imageA", "fakeB", "imageA_up", "imageB", "pixelfakeB_out", "outputrealB_out"]  # ""    , "fcreal_out"

        self.model_names = ['generator']
        if self.isTrain:
            self.model_names += ['pixel_discriminator', 'fc_discriminator']  # "net"

        # 特征生成器
        self.netgenerator = networks.srdanet_generator(num_cls=opt.num_classes, gpu_ids=self.gpu_ids)
        if self.isTrain:
            # 像素空间判别器
            self.netpixel_discriminator = networks.define_D(3, 64, 'basic', norm="instance", gpu_ids=self.gpu_ids)

            # 输出空间判别器
            self.netfc_discriminator = networks.srdanet_ds(num_classes=opt.num_classes, gpu_ids=self.gpu_ids)

            # 语义分割损失
            self.loss_function = nn.CrossEntropyLoss(ignore_index=255).to(self.device)

            # idt 损失
            self.generator_criterion = networks.GeneratorLoss().to(self.device)

            # 像素判别器损失
            self.mse_loss = networks.GANLoss("lsgan").to(self.device)

            # 输出空间判别器损失
            self.bce_loss = networks.GANLoss("vanilla").to(self.device)

            # 内容一致损失
            self.L1_loss = nn.L1Loss().to(self.device)

            # 优化器
            self.optimizer = torch.optim.Adam(self.netgenerator.parameters(),
                                              lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=0.0005)
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netpixel_discriminator.parameters(),
                                                                self.netfc_discriminator.parameters()),
                                                lr=opt.lr_D, betas=(opt.beta1, 0.999), weight_decay=0.0005)
            self.optimizers.append(self.optimizer)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        if self.isTrain:
            self.imageA = input["A_img"].to(self.device)  #[-1, 1]
            self.imageA_up = input["A_img_up"].to(self.device) #[-1, 1]
            self.label = input["A_label_up"].to(self.device)
            self.imageB = input["B_img"].to(self.device) #[-1, 1]
            self.imageB_down = input["B_img_down"].to(self.device)
        else:
            self.imageB_down = input["B_img_down"].to(self.device)
            self.label = input["label"].to(self.device)
            self.imageB = input["B_img"].to(self.device)
    def forward(self):
        if self.isTrain:
            # iamgeA 通过 generator
            self.feature_A, self.pre, self.fakeB = self.netgenerator(self.imageA)
            _, _, h, w = self.imageA_up.size()
            self.fakeB = nn.functional.interpolate(self.fakeB, mode="bilinear", size=(h, w), align_corners=True)
            self.fakeB = F.tanh(self.fakeB)
            _, _, h1, w1 = self.imageA.size()
            self.fakeB_down = nn.functional.interpolate(self.fakeB, size=(h1, w1))

            self.fakeB_cut = self.fakeB.detach() # 隔断反向传播
            self.fakeB_down_cut = self.fakeB_down.detach() # 隔断反向传播
            self.feature_fakeB_down_cut, self.pre_aux, _ = self.netgenerator(self.fakeB_down_cut)

            self.pre_A_cut = self.pre.detach()  # 隔断反向传播
            self.prediction = self.pre.data.max(1)[1].unsqueeze(1)
            # imageB_down 通过 generator
            _, self.pre_B, _ = self.netgenerator(self.imageB_down)
            self.pre_B_cut = self.pre_B.detach()  # 隔断反向传播

            # fakeB 通过判别器
            self.pixelfakeB_out = self.netpixel_discriminator(self.fakeB)
            self.outputrealB_out = self.netfc_discriminator(F.softmax(self.pre_B, dim=1))
        else:
            _, self.pre, _ = self.netgenerator(self.imageB_down)
            _, _, h, w = self.imageB.size()
            self.pre = nn.functional.interpolate(self.pre, mode="bilinear", size=(h, w), align_corners=True)
            self.prediction = self.pre.data.max(1)[1].unsqueeze(1)
    def backward(self):
        """计算两个损失"""
        # 分割损失
        _, _, h, w = self.imageA_up.size()
        self.pre_up_to_label = nn.functional.interpolate(self.pre, mode="bilinear", size=(h, w), align_corners=True)
        self.pre_aux_up_to_label = nn.functional.interpolate(self.pre_aux, mode="bilinear", size=(h, w),
                                                             align_corners=True)
        cross_entropy = self.loss_function(self.pre_up_to_label, self.label.long().squeeze(1))
        cross_entropy_aux = self.loss_function(self.pre_aux_up_to_label, self.label.long().squeeze(1))
        self.loss_cross_entropy = (cross_entropy + cross_entropy_aux) * 0.5

        # 像素级对齐损失
        self.loss_da1 = self.mse_loss(self.pixelfakeB_out, True)

        # 输出空间对齐
        self.loss_da2 = self.bce_loss(self.outputrealB_out, False)

        # A内容一致性损失
        self.loss_idtA = self.generator_criterion(self.fakeB, self.imageA_up, is_sr=False)

        # fix_pointA loss
        self.loss_fix_point = self.L1_loss(self.feature_A, self.feature_fakeB_down_cut)

        loss_DA = self.loss_da1 + self.loss_da2
        loss_ID = self.loss_idtA + self.loss_fix_point * 0.5

        # 求分割损失和超分辨损失的和
        self.loss_G = loss_DA * 2 + loss_ID * 10 + self.loss_cross_entropy * 10
        self.loss_G.backward(retain_graph=True)

    def backward_D(self):

        pixeltrueB_out = self.netpixel_discriminator(self.imageB)

        self.loss_D_da1 = self.mse_loss(self.netpixel_discriminator(self.fakeB_cut), False) \
                          + self.mse_loss(pixeltrueB_out, True)
        self.loss_D_da2 = self.bce_loss(self.netfc_discriminator(F.softmax(self.pre_A_cut, dim=1)), False) \
                          + self.bce_loss(self.netfc_discriminator(F.softmax(self.pre_B_cut, dim=1)), True)

        self.loss_D = (self.loss_D_da1 + self.loss_D_da2) * 1
        self.loss_D.backward()

    def optimize_parameters(self):
        self.forward()

        # 不求判别器的梯度
        self.set_requires_grad([self.netpixel_discriminator, self.netfc_discriminator], False)

        # 更新生成器的参数
        self.optimizer.zero_grad()
        self.backward()  # 计算生成器的参数的梯度
        self.optimizer.step()  # 更新参数

        # 可以求判别器的梯度
        self.set_requires_grad([self.netpixel_discriminator, self.netfc_discriminator], True)
        self.optimizer_D.zero_grad()
        self.backward_D()  # 计算判别器的梯度
        self.optimizer_D.step()  # update weights

