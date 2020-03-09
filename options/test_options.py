from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--results_dir', type=str, default='./results', help='saves results here.')

        parser.add_argument('--epoch', type=str, default='50',
                            help='which epoch to load? set to latest to use latest cached model')
        self.isTrain = False
        return parser