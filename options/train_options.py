from .base_options import BaseOptions
class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # 是否恢复权重参数
        parser.add_argument('--continue_train', action='store_true', help='continue training: load --epoch model')
        parser.add_argument('--epoch', type=str, default='100',
                            help='which epoch to load? set to latest to use latest cached model')

        # 显示和保存设置参数
        parser.add_argument('--display_freq', type=int, default=100,
                            help='frequency of showing training results on screen')
        parser.add_argument('--print_freq', type=int, default=80,
                            help='frequency of showing training results on console')
        parser.add_argument('--save_epoch_freq', type=int, default=5,
                            help='frequency of saving checkpoints at the end of epochs')


        # 学习率参数和训练次数参数
        parser.add_argument('--niter', type=int, default=20, help='# of iter at starting learning rate')
        parser.add_argument('--niter_decay', type=int, default=30,
                            help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.00025, help='initial learning rate for adam')
        parser.add_argument('--lr_D', type=float, default=0.00025, help='initial learning rate for adam')
        parser.add_argument('--lr_policy', type=str, default='linear',
                            help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=10,
                            help='multiply by a gamma every lr_decay_iters iterations') # 换学习率时才用到
        parser.add_argument('--epoch_count', type=int, default=1,
                            help='the starting epoch count, for linear learning rate')
        self.isTrain = True
        return parser