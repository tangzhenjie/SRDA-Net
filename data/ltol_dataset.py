from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
import random
from PIL import Image as m
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np


def transform(image, mask, opt):
    if not opt.no_crop:
        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=(opt.A_crop_size, opt.A_crop_size))
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)

    # Random horizontal flipping
    if random.random() > 0.5:
        image = TF.hflip(image)
        mask = TF.hflip(mask)

    # Random vertical flipping
    if random.random() > 0.5:
        image = TF.vflip(image)
        mask = TF.vflip(mask)

    mask = np.array(mask).astype(np.long)
    nomal_fun_image = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    # Transform to tensor
    image = TF.to_tensor(image)
    image = nomal_fun_image(image)

    mask = TF.to_tensor(mask)

    return image, mask


def transformB(image, opt):
    # Random crop
    i, j, h, w = transforms.RandomCrop.get_params(
        image, output_size=(opt.B_crop_size, opt.B_crop_size))
    image = TF.crop(image, i, j, h, w)

    # 降采样成 A_crop_size
    image_down = image.resize((opt.A_crop_size, opt.A_crop_size), resample=m.BICUBIC)

    nomal_fun_image = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    image_down = TF.to_tensor(image_down)
    image_down = nomal_fun_image(image_down)

    return  image_down


class LtolDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets

    It requires two directories to host training images from domain A '/path/A/train/images
    and from domain B '/path/B/train/images
    You can train the model with flag '--dataroot /path/'
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        :param parser: -- original option parser
        :param is_train: -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
        :return: the modified parser.
        """
        parser.add_argument('--A_crop_size', type=int, default=280, help='crop to this size')  # 240
        parser.add_argument('--B_crop_size', type=int, default=933, help='crop to this size')
        parser.add_argument('--inter_method_image', type=str, default='bicubic', help='the image Interpolation method')
        parser.add_argument('--inter_method_label', type=str, default='nearest', help='the label Interpolation method')
        parser.add_argument('--no_crop', type=bool, default=False,
                            help='crop the A and B according to the special datasets params  [crop | none],')
        parser.add_argument('--no_flip', type=bool, default=False,
                            help='if specified, do not flip the images for data augmentation')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
                            help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc  for the directory name')
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A_images = opt.dataroot + "/" + opt.phase + 'A/images'  # create a path '/trainA/images/*.tif'
        self.dir_B_images = opt.dataroot + "/" + opt.phase + 'B/images'  # create a path '/trainB/images/*.tif'
        self.dir_A_labels = opt.dataroot + "/" + opt.phase + 'A/labels'  # create a path '/trainA/labels/*.tif'

        self.A_images_paths = sorted(
            make_dataset(self.dir_A_images, opt.max_dataset_size))  # load images from '/trainA/images/*.tif'
        self.B_images_paths = sorted(
            make_dataset(self.dir_B_images, opt.max_dataset_size))  # load images from '/trainA/images/*.tif'
        self.A_labels_paths = sorted(make_dataset(self.dir_A_labels, opt.max_dataset_size))
        self.A_size = len(self.A_images_paths)  # get the size of dataset A
        self.B_size = len(self.B_images_paths)  # get the size of dataset B
        # self.transform_B = get_transformB(self.opt)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_image_path = self.A_images_paths[index % self.A_size]  # make sure index is within then range
        A_label_path = self.A_labels_paths[index % self.A_size]  # A_path is same as C_path
        if self.opt.serial_batches:  # make sure index is within then range
            index_B = index % self.B_size
        else:  # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_image_path = self.B_images_paths[index_B]

        A_img = m.open(A_image_path).convert('RGB')  # 马萨诸塞数据
        A_label = m.open(A_label_path).convert('L')  # 马萨诸塞标签
        B_img = m.open(B_image_path).convert('RGB')  # inria数据

        A_img, A_label = transform(A_img, A_label, self.opt)
        B_img_down = transformB(B_img, self.opt)

        return {'A_img': A_img,  'A_label': A_label, 'B_img': B_img_down}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)