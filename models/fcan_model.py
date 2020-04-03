import torch
import itertools
from .base_model import BaseModel
from . import networks
from torch import nn
from util.image_pool import ImagePool
import torch.nn.functional as F
import util.loss as loss

class FcanModel(BaseModel):
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
            parser.add_argument('--gan_mode', type=str, default='lsgan',
                                help='the type of GAN objective.')
        return parser
    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ["G", "D"]    # "D"
        self.visual_names = ["imageB", "label", "prediction"]  # ""
        if self.isTrain:
            self.visual_names += ["imageA", "D_out_A", "D_out_B"]  # "D_out_A", "D_out_B"

        self.model_names = ["FCAN_Backbone", "psp_classifier"]  # "net"
        if self.isTrain:
            self.model_names += ["aspp_discriminator"]  # "net"

        # 特征生成器
        self.netFCAN_Backbone = networks.FCAN_Backbone(is_restore_from_imagenet=opt.is_restore_from_imagenet,
                                                                    resnet_weight_path=opt.resnet_weight_path,
                                                                    gpu_ids=self.gpu_ids)
        self.netpsp_classifier = networks.psp_classifier(opt.num_classes, self.gpu_ids)
        if self.isTrain:

            self.netaspp_discriminator = networks.aspp_discriminator(gpu_ids=self.gpu_ids)


            # 语义分割损失
            self.loss_function = nn.CrossEntropyLoss(ignore_index=255).to(self.device)

            # 输出空间判别器损失
            self.MSE_loss = networks.GANLoss("lsgan").to(self.device)

            # 优化器
            self.optimizer = torch.optim.Adam(itertools.chain(self.netFCAN_Backbone.parameters(), self.netpsp_classifier.parameters()),
                                              lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=0.0005)
            self.optimizer_D = torch.optim.Adam(self.netaspp_discriminator.parameters(),
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
            self.feature_A = self.netFCAN_Backbone(self.imageA)
            self.feature_B = self.netFCAN_Backbone(self.imageB)
            self.D_out_A = self.netaspp_discriminator(self.feature_A)
            self.D_out_B = self.netaspp_discriminator(self.feature_B)
         

            self.pre = self.netpsp_classifier(self.feature_A)
            _, _, h, w = self.imageA.size()
            self.pre = nn.functional.interpolate(self.pre, size=(h, w), mode='bilinear')
            self.prediction = self.pre.data.max(1)[1].unsqueeze(1)

        else:
            self.feature_B = self.netFCAN_Backbone(self.imageB)
            self.pre = self.netpsp_classifier(self.feature_B)
            _, _, h, w = self.imageB.size()
            self.pre = nn.functional.interpolate(self.pre, size=(h, w), mode='bilinear')
            self.prediction = self.pre.data.max(1)[1].unsqueeze(1)
    def backward(self):
        """计算两个损失"""
        # 分割损失
        self.loss_seg = self.loss_function(self.pre, self.label.long().squeeze(1))
        
        # 输出空间对齐
        self.loss_adv = self.MSE_loss(self.D_out_A, True)

        # 求分割损失和超分辨损失的和
        self.loss_G = self.loss_seg * 5 + self.loss_adv
        self.loss_G.backward(retain_graph=True)

    def backward_D(self):
        self.loss_D1 = self.MSE_loss(self.D_out_A, False) \
                          + self.MSE_loss(self.D_out_B, True)

        self.loss_D = self.loss_D1 * 0.5
        self.loss_D.backward()

    def optimize_parameters(self):
        self.forward()

        # 不求判别器的梯度
        self.set_requires_grad([self.netaspp_discriminator], False)

        # 更新生成器的参数
        self.optimizer.zero_grad()
        self.backward()  # 计算生成器的参数的梯度
        self.optimizer.step()  # 更新参数

        # 可以求判别器的梯度
        self.set_requires_grad([self.netaspp_discriminator], True)
        self.optimizer_D.zero_grad()
        self.backward_D()  # 计算判别器的梯度
        self.optimizer_D.step()  # update weights

