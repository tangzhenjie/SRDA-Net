from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
import torchvision.transforms as transforms
from PIL import Image as m
import torch
import numpy as np
import torchvision.transforms.functional as TF
import random

def transform(image, mask, opt):

    if not opt.no_crop:
        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=(opt.crop_size, opt.crop_size))
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)

    mask = np.array(mask).astype(np.long)
    nomal_fun_image = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    # Transform to tensor
    image = TF.to_tensor(image)
    image = nomal_fun_image(image)
    mask = TF.to_tensor(mask)

    return image, mask

class ConvertDataset(BaseDataset):
    """load train and val for segmentation network
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        :param parser: -- original option parser
        :param is_train: -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
        :return: the modified parser.
        """
        parser.add_argument('--no_crop',  type=bool, default=True,
                            help='crop the A and B according to the special datasets params  [crop | none],')
        parser.add_argument('--crop_size', type=int, default=300, help='crop to this size')
        parser.add_argument('--phase', type=str, default='trainA', help='train, val, test, etc  for the directory name')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
                            help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        return parser
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_images = opt.dataroot + "/" + opt.phase + '/images'  # create a path '/trainA/images/*.png'
        self.dir_labels = opt.dataroot + "/" + opt.phase + '/labels'  # labels path

        self.images_paths = sorted(
            make_dataset(self.dir_images, opt.max_dataset_size))  # load images from '/images/*.png'
        self.labels_paths = sorted(
            make_dataset(self.dir_labels, opt.max_dataset_size))  # load images from '/labels/*.png'
        self.size = len(self.images_paths)

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
        # read the image and corresponding to the label
        image_path = self.images_paths[index % self.size]
        label_path = self.labels_paths[index % self.size]

        img = m.open(image_path).convert('RGB')  
        label = m.open(label_path).convert('L')  
        img, label = transform(img, label, self.opt)

        return {'img': img, 'label': label, "img_path":image_path, "label_path": label_path}


    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return self.size