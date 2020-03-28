import torch
import itertools
from .base_model import BaseModel
from . import networks
from torch import nn
from util.image_pool import ImagePool
import torch.nn.functional as F
import util.loss as loss


class AdaptsegnetModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.
        """
        parser.add_argument('--num_classes', type=int, default=2, help='for determining the class number')
        parser.add_argument('--is_restore_from_imagenet', type=bool, default=True,
                            help='for fcn determining whether or not to restore resnet50 from imagenet')
        parser.add_argument('--resnet_weight_path', type=str, default='./resnetweight/',
                            help='the path to renet_weight from imagenet')
        if is_train:
            parser.add_argument('--gan_mode', type=str, default='vanilla',
                                help='the type of GAN objective.')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ["G", "D"]  # "loss_"
        self.visual_names = ["imageB", "label", "prediction"]  # ""
        if self.isTrain:
            self.visual_names += ["imageA", "D_out_target1", "D_out_target2"]  # ""    , "fcreal_out"

        self.model_names = ["AdaptSegnet"]  # "net"
        if self.isTrain:
            self.model_names += ["Model_D1", "Model_D2"]  # "net"

        # 特征生成器
        self.netAdaptSegnet = networks.AdaptSegnet(
            is_restore_from_imagenet=opt.is_restore_from_imagenet,
            resnet_weight_path=opt.resnet_weight_path,
            num_classes=opt.num_classes,
            gpu_ids=self.gpu_ids)

        if self.isTrain:
            # 输出空间判别器
            self.netModel_D1 = networks.srdanet_ds(num_classes=opt.num_classes, gpu_ids=self.gpu_ids)
            self.netModel_D2 = networks.srdanet_ds(num_classes=opt.num_classes, gpu_ids=self.gpu_ids)

            # 语义分割损失
            self.loss_function = nn.CrossEntropyLoss(ignore_index=255).to(self.device)

            # 输出空间判别器损失
            self.bce_loss = networks.GANLoss("vanilla").to(self.device)

            # 优化器
            self.optimizer = torch.optim.Adam(self.netAdaptSegnet.module.optim_parameters(opt),
                                              lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=0.0005)
            self.optimizer_D = torch.optim.Adam(
                itertools.chain(self.netModel_D1.parameters(), self.netModel_D2.parameters()),
                lr=opt.lr_D, betas=(opt.beta1, 0.999), weight_decay=0.0005)
            self.optimizers.append(self.optimizer)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        if self.isTrain:
            self.imageA = input["A_img"].to(self.device)
            self.label = input["A_label"].to(self.device)
            self.imageB = input["B_img"].to(self.device)
        else:
            self.imageB = input["img"].to(self.device)
            self.label = input["label"].to(self.device)

    def forward(self):
        if self.isTrain:
            # iamgeA 通过 generator
            self.pred1, self.pre = self.netAdaptSegnet(self.imageA)
            self.prediction = self.pre.data.max(1)[1].unsqueeze(1)
            self.pred1_cut, self.pred2_cut = self.pred1.detach(), self.pre.detach()

            self.pred_target1, self.pred_target2 = self.netAdaptSegnet(self.imageB)
            self.pred_target1_cut, self.pred_target2_cut = self.pred_target1.detach(), self.pred_target2.detach()

            self.D_out_target1 = self.netModel_D1(F.softmax(self.pred_target1, dim=1))
            self.D_out_target2 = self.netModel_D2(F.softmax(self.pred_target2, dim=1))

        else:
            _, self.pre = self.netAdaptSegnet(self.imageB)
            self.prediction = self.pre.data.max(1)[1].unsqueeze(1)

    def backward(self):
        """计算两个损失"""
        # 分割损失
        loss_seg1 = self.loss_function(self.pred1, self.label.long().squeeze(1))
        loss_seg2 = self.loss_function(self.pre, self.label.long().squeeze(1))

        self.loss_cross_entropy = 0.1 * loss_seg1 + loss_seg2

        # 输出空间对齐
        loss_adv_target1 = self.bce_loss(self.D_out_target1, False)
        loss_adv_target2 = self.bce_loss(self.D_out_target2, False)

        self.loss_adv = 0.0002 * loss_adv_target1 + 0.001 * loss_adv_target2

        # 求分割损失和超分辨损失的和
        self.loss_G = self.loss_cross_entropy + self.loss_adv
        self.loss_G.backward(retain_graph=True)

    def backward_D(self):
        self.loss_D1 = self.bce_loss(self.netModel_D1(F.softmax(self.pred1_cut, dim=1)), False) \
                       + self.bce_loss(self.netModel_D1(F.softmax(self.pred_target1_cut, dim=1)), True)

        self.loss_D2 = self.bce_loss(self.netModel_D2(F.softmax(self.pred2_cut, dim=1)), False) \
                       + self.bce_loss(self.netModel_D2(F.softmax(self.pred_target2_cut, dim=1)), True)

        self.loss_D = (self.loss_D1 + self.loss_D2) * 0.5
        self.loss_D.backward()

    def optimize_parameters(self):
        self.forward()

        # 不求判别器的梯度
        self.set_requires_grad([self.netModel_D1, self.netModel_D2], False)

        # 更新生成器的参数
        self.optimizer.zero_grad()
        self.backward()  # 计算生成器的参数的梯度
        self.optimizer.step()  # 更新参数

        # 可以求判别器的梯度
        self.set_requires_grad([self.netModel_D1, self.netModel_D2], True)
        self.optimizer_D.zero_grad()
        self.backward_D()  # 计算判别器的梯度
        self.optimizer_D.step()  # update weights

