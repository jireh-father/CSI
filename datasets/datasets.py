import os
import cv2

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import albumentations as al
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
from torch.utils.data.dataset import Subset
from torchvision import datasets, transforms

from utils.utils import set_random_seed

DATA_PATH = '~/data/'
IMAGENET_PATH = '~/data/ImageNet'

CIFAR10_SUPERCLASS = list(range(10))  # one class
IMAGENET_SUPERCLASS = list(range(30))  # one class

CIFAR100_SUPERCLASS = [
    [4, 31, 55, 72, 95],
    [1, 33, 67, 73, 91],
    [54, 62, 70, 82, 92],
    [9, 10, 16, 29, 61],
    [0, 51, 53, 57, 83],
    [22, 25, 40, 86, 87],
    [5, 20, 26, 84, 94],
    [6, 7, 14, 18, 24],
    [3, 42, 43, 88, 97],
    [12, 17, 38, 68, 76],
    [23, 34, 49, 60, 71],
    [15, 19, 21, 32, 39],
    [35, 63, 64, 66, 75],
    [27, 45, 77, 79, 99],
    [2, 11, 36, 46, 98],
    [28, 30, 44, 78, 93],
    [37, 50, 65, 74, 80],
    [47, 52, 56, 59, 96],
    [8, 13, 48, 58, 90],
    [41, 69, 81, 85, 89],
]


class MultiDataTransformAlbu(object):
    def __init__(self, transform):
        self.transform1 = transform
        self.transform2 = transform

    def __call__(self, sample):
        x1 = self.transform1(image=np.array(sample))['image']
        x2 = self.transform1(image=np.array(sample))['image']
        return x1, x2


class MultiDataTransform(object):
    def __init__(self, transform):
        self.transform1 = transform
        self.transform2 = transform

    def __call__(self, sample):
        x1 = self.transform1(sample)
        x2 = self.transform2(sample)
        return x1, x2


class MultiDataTransformList(object):
    def __init__(self, transform, clean_trasform, sample_num):
        self.transform = transform
        self.clean_transform = clean_trasform
        self.sample_num = sample_num

    def __call__(self, sample):
        set_random_seed(0)

        sample_list = []
        for i in range(self.sample_num):
            sample_list.append(self.transform(sample))

        return sample_list, self.clean_transform(sample)


def get_transform(image_size=None):
    # Note: data augmentation is implemented in the layers
    # Hence, we only define the identity transformation here
    if image_size:  # use pre-specified image size
        train_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
        ])
    else:  # use default image size
        train_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        test_transform = transforms.ToTensor()

    return train_transform, test_transform


def get_subset_with_len(dataset, length, shuffle=False):
    set_random_seed(0)
    dataset_size = len(dataset)

    index = np.arange(dataset_size)
    if shuffle:
        np.random.shuffle(index)

    index = torch.from_numpy(index[0:length])
    subset = Subset(dataset, index)

    assert len(subset) == length

    return subset


def get_transform_imagenet(use_albu_aug):
    if use_albu_aug:
        train_transform = al.Compose([
            # al.Flip(p=0.5),
            al.Resize(256, 256, interpolation=2),
            al.RandomResizedCrop(224, 224, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=2),
            al.HorizontalFlip(),
            al.OneOf([
                al.OneOf([
                    al.ShiftScaleRotate(border_mode=cv2.BORDER_CONSTANT, rotate_limit=30),  # , p=0.05),
                    al.OpticalDistortion(border_mode=cv2.BORDER_CONSTANT, distort_limit=5.0, shift_limit=0.1),
                    # , p=0.05),
                    al.GridDistortion(border_mode=cv2.BORDER_CONSTANT),  # , p=0.05),
                    al.ElasticTransform(border_mode=cv2.BORDER_CONSTANT, alpha_affine=15),  # , p=0.05),
                ], p=0.1),

                al.OneOf([
                    al.RandomGamma(),  # p=0.05),
                    al.HueSaturationValue(),  # p=0.05),
                    al.RGBShift(),  # p=0.05),
                    al.CLAHE(),  # p=0.05),
                    al.ChannelShuffle(),  # p=0.05),
                    al.InvertImg(),  # p=0.05),
                ], p=0.1),
                al.OneOf([
                    al.RandomSnow(),  # p=0.05),
                    al.RandomRain(),  # p=0.05),
                    al.RandomFog(),  # p=0.05),
                    al.RandomSunFlare(num_flare_circles_lower=1, num_flare_circles_upper=2, src_radius=110),
                    # p=0.05, ),
                    al.RandomShadow(),  # p=0.05),
                ], p=0.1),
                al.RandomBrightnessContrast(p=0.1),

                al.OneOf([
                    al.GaussNoise(),  # p=0.05),
                    al.ISONoise(),  # p=0.05),
                    al.MultiplicativeNoise(),  # p=0.05),
                ], p=0.1),

                al.OneOf([
                    al.ToGray(),  # p=0.05),
                    al.ToSepia(),  # p=0.05),
                    al.Solarize(),  # p=0.05),
                    al.Equalize(),  # p=0.05),
                    al.Posterize(),  # p=0.05),
                    al.FancyPCA(),  # p=0.05),
                ], p=0.1),

                al.OneOf([
                    # al.MotionBlur(blur_limit=1),
                    al.Blur(blur_limit=[3, 5]),
                    al.MedianBlur(blur_limit=[3, 5]),
                    al.GaussianBlur(blur_limit=[3, 5]),
                ], p=0.1),
                al.OneOf([
                    al.CoarseDropout(),  # p=0.05),
                    al.Cutout(),  # p=0.05),
                    al.GridDropout(),  # p=0.05),
                    al.ChannelDropout(),  # p=0.05),
                    al.RandomGridShuffle(),  # p=0.05),
                ], p=0.1),
                al.OneOf([
                    al.Downscale(),  # p=0.1),
                    al.ImageCompression(quality_lower=60),  # , p=0.1),
                ], p=0.1),

            ], p=0.5),
            al.Normalize(),
            ToTensorV2()
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    if use_albu_aug:
        train_transform = MultiDataTransformAlbu(train_transform)
    else:
        train_transform = MultiDataTransform(train_transform)

    return train_transform, test_transform


def get_dataset(P, dataset, test_only=False, image_size=None, download=False, eval=False):
    if not P.use_cifar10 or dataset in ['imagenet', 'cub', 'stanford_dogs', 'flowers102',
                                        'places365', 'food_101', 'caltech_256', 'dtd', 'pets', 'skin',
                                        'ab'] or dataset.startswith("skin"):
        if eval:
            train_transform, test_transform = get_simclr_eval_transform_imagenet(P.ood_samples,
                                                                                 P.resize_factor, P.resize_fix)
        else:
            train_transform, test_transform = get_transform_imagenet(P.use_albu_aug)
    else:
        if dataset == 'skin_small':
            image_size = (32, 32, 3)
        train_transform, test_transform = get_transform(image_size=image_size)

    if dataset == 'cifar10':
        image_size = (32, 32, 3)
        n_classes = 10
        train_set = datasets.CIFAR10(DATA_PATH, train=True, download=download, transform=train_transform)
        test_set = datasets.CIFAR10(DATA_PATH, train=False, download=download, transform=test_transform)

    elif dataset == 'cifar100':
        image_size = (32, 32, 3)
        n_classes = 100
        train_set = datasets.CIFAR100(DATA_PATH, train=True, download=download, transform=train_transform)
        test_set = datasets.CIFAR100(DATA_PATH, train=False, download=download, transform=test_transform)

    elif dataset == 'svhn':
        assert test_only and image_size is not None
        test_set = datasets.SVHN(DATA_PATH, split='test', download=download, transform=test_transform)

    elif dataset == 'lsun_resize':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'LSUN_resize')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    elif dataset == 'lsun_fix':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'LSUN_fix')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    elif dataset == 'imagenet_resize':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'Imagenet_resize')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    elif dataset == 'imagenet_fix':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'Imagenet_fix')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    elif dataset == 'imagenet':
        image_size = (224, 224, 3)
        n_classes = 30
        train_dir = os.path.join(IMAGENET_PATH, 'one_class_train')
        test_dir = os.path.join(IMAGENET_PATH, 'one_class_test')
        train_set = datasets.ImageFolder(train_dir, transform=train_transform)
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    elif dataset == 'skin':
        image_size = (224, 224, 3)
        n_classes = 2
        train_dir = os.path.join(DATA_PATH, 'skin', 'train')
        test_dir = os.path.join(DATA_PATH, 'skin', 'test')
        train_set = datasets.ImageFolder(train_dir, transform=train_transform)
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
    elif dataset == 'skin_foot':
        image_size = (224, 224, 3)
        n_classes = 2
        train_dir = os.path.join(DATA_PATH, 'skin_foot', 'train')
        test_dir = os.path.join(DATA_PATH, 'skin_foot', 'test')
        train_set = datasets.ImageFolder(train_dir, transform=train_transform)
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    elif dataset == 'skin_ear':
        image_size = (224, 224, 3)
        n_classes = 2
        train_dir = os.path.join(DATA_PATH, 'skin_ear', 'train')
        test_dir = os.path.join(DATA_PATH, 'skin_ear', 'test')
        train_set = datasets.ImageFolder(train_dir, transform=train_transform)
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
    elif dataset == 'ab':
        image_size = (224, 224, 3)
        n_classes = 2
        train_dir = os.path.join(DATA_PATH, 'ab', 'train')
        test_dir = os.path.join(DATA_PATH, 'ab', 'test')
        train_set = datasets.ImageFolder(train_dir, transform=train_transform)
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    elif dataset == 'skin_small':
        image_size = (32, 32, 3)
        n_classes = 2
        train_dir = os.path.join(DATA_PATH, 'skin', 'train')
        test_dir = os.path.join(DATA_PATH, 'skin', 'test')
        train_set = datasets.ImageFolder(train_dir, transform=train_transform)
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    elif dataset == 'skin_total':
        image_size = (224, 224, 3)
        n_classes = 3
        train_dir = os.path.join(DATA_PATH, 'skin_total', 'train')
        test_dir = os.path.join(DATA_PATH, 'skin_total', 'test')
        train_set = datasets.ImageFolder(train_dir, transform=train_transform)
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
    elif dataset == 'erythema_all_pos_0_neg_123':
        image_size = (224, 224, 3)
        n_classes = 2
        train_dir = os.path.join(DATA_PATH, 'erythema_all_pos_0_neg_123', 'train')
        test_dir = os.path.join(DATA_PATH, 'erythema_all_pos_0_neg_123', 'test')
        train_set = datasets.ImageFolder(train_dir, transform=train_transform)
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
    elif dataset == 'erythema_all_pos_01_neg_23':
        image_size = (224, 224, 3)
        n_classes = 2
        train_dir = os.path.join(DATA_PATH, 'erythema_all_pos_01_neg_23', 'train')
        test_dir = os.path.join(DATA_PATH, 'erythema_all_pos_01_neg_23', 'test')
        train_set = datasets.ImageFolder(train_dir, transform=train_transform)
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
    elif dataset == 'noskin':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'noskin')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        # test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    elif dataset == 'stanford_dogs':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'stanford_dogs')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    elif dataset == 'cub':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'cub200')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    elif dataset == 'flowers102':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'flowers102')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    elif dataset == 'places365':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'places365')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    elif dataset == 'food_101':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'food-101', 'images')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    elif dataset == 'caltech_256':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'caltech-256')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    elif dataset == 'dtd':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'dtd', 'images')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    elif dataset == 'pets':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'pets')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    else:
        if P.use_cifar10:
            raise NotImplementedError()
        else:
            image_size = (224, 224, 3)
            n_classes = 2
            train_dir = os.path.join(DATA_PATH, dataset, 'train')
            test_dir = os.path.join(DATA_PATH, dataset, 'test')
            train_set = datasets.ImageFolder(train_dir, transform=train_transform)
            test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    if test_only:
        return test_set
    else:
        return train_set, test_set, image_size, n_classes


def get_superclass_list(dataset):
    if dataset == 'cifar10':
        return CIFAR10_SUPERCLASS
    elif dataset == 'cifar100':
        return CIFAR100_SUPERCLASS
    elif dataset == 'imagenet':
        return IMAGENET_SUPERCLASS
    elif dataset in ['skin', 'skin_small', 'ab']:
        return list(range(2))
    elif dataset in ['skin_total']:
        return list(range(3))
    elif dataset.startswith("skin"):
        return list(range(2))
    else:
        return list(range(2))
        # raise NotImplementedError()


def get_subclass_dataset(dataset, classes):
    if not isinstance(classes, list):
        classes = [classes]

    indices = []
    for idx, tgt in enumerate(dataset.targets):
        if tgt in classes:
            indices.append(idx)

    dataset = Subset(dataset, indices)
    return dataset


def get_simclr_eval_transform_imagenet(sample_num, resize_factor, resize_fix):
    resize_scale = (resize_factor, 1.0)  # resize scaling factor
    if resize_fix:  # if resize_fix is True, use same scale
        resize_scale = (resize_factor, resize_factor)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=resize_scale),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    clean_trasform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    transform = MultiDataTransformList(transform, clean_trasform, sample_num)

    return transform, transform
