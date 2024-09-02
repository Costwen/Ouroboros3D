

import os
import math
import torch
import numbers
from einops import rearrange
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torchvision.datasets import CIFAR10, CIFAR100, SVHN
# from dataset.ffhq import FFHQ
# from dataset.diffae import CelebAttrDataset, CelebHQAttrDataset
import numpy as np


class Crop(object):
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def __call__(self, img):
        return F.crop(img, self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)

    def __repr__(self):
        return self.__class__.__name__ + "(x1={}, x2={}, y1={}, y2={})".format(
            self.x1, self.x2, self.y1, self.y2
        )


class Identity(object):
    def __init__(self):
        pass

    def __call__(self, img):
        return img


class Split(object):  # todo ��ͼ����������
    def __init__(self, split_type, split_block, split_smooth=False):
        self.type = split_type
        self.smooth = split_smooth
        # self.block = split_block
        self.ratio = int(math.log2(split_block)) if split_type == 'corner' else int(math.sqrt(split_block))

    @ torch.no_grad()
    def __call__(self, img):
        split_tpye = self.type
        if split_tpye == 'corner':
            split_ratio = self.ratio

            def split_fn(img):
                c, h, w = img.shape

                img_split = img.view(c, 2, h // 2, 2, w // 2)  # todo ��Ϊrearrange�汾
                img_split = img_split.permute(1, 3, 0, 2, 4).reshape(-1, c, h // 2, w // 2)
                img_split = torch.index_select(img_split, 0, torch.randint(0, 4, (1,)))

                # img_small = transforms.Resize((h // 2, w // 2))(img)
                img_small = F.resize(img, [h // 2, w // 2])
                return img_split, img_small

            img_split, img_small = None, None
            for _ in range(split_ratio // 2):
                img_split, img_small = split_fn(img)
                img = img_split
        elif split_tpye == 'center':
            split_ratio = self.ratio
            c, h, w = img.shape
            assert h == w
            h_new, w_new = h // split_ratio, w // split_ratio

            x, y = torch.randint(0, split_ratio, (2,))
            img = F.pad(img, [h_new // 2] * 4)
            # img = img[..., x * h_new:(x + 2) * h_new, y * h_new:(y + 2) * h_new]
            img = F.crop(img, x * h_new, y * h_new, 2 * h_new, 2 * h_new)

            # img_split = img[..., h_new // 2:-h_new // 2, h_new // 2:-h_new // 2]
            img_split = F.crop(img, h_new // 2, h_new // 2, h_new, h_new)
            # img_small = transforms.Resize((h_new, h_new))(img)
            img_small = F.resize(img, [h_new, h_new])
        elif split_tpye == 'resize':
            split_ratio = self.ratio
            c, h, w = img.shape
            assert h == w
            h_new, w_new = h // split_ratio, w // split_ratio

            if self.smooth:
                x, y = torch.randint(0, h - h_new, (2, ))
                img_split = F.crop(img, x, y, h_new, h_new)
            else:
                x, y = torch.randint(0, split_ratio, (2,))
                # img = F.pad(img, [4] * 4, fill=0)
                img_split = F.crop(img, x * h_new, y * h_new, h_new, h_new)
            # img_small = F.resize(F.resize(img_split, [(h_new + 8) // 2]*2), [h_new + 8]*2)
            return img_split
        else:
            img_split, img_small = None, None

        img_output = torch.cat([img_split, img_small], dim=0)

        return img_output


def get_dataset(config):
    if config['random_flip'] is False:
        tran_transform = test_transform = transforms.Compose(
            [
                transforms.Resize(config['image_size']),
                transforms.ToTensor(),
            ]
        )
    else:
        tran_transform = transforms.Compose(
            [
                transforms.Resize(config['image_size']),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.Resize(config['image_size']),
                transforms.ToTensor(),
            ]
        )

    if config['name'] == "CIFAR10":
        dataset = CIFAR10(
            os.path.join(os.getcwd(), "temp/dataset", "cifar10"),
            train=True,
            download=True,
            transform=tran_transform,
        )
        test_dataset = CIFAR10(
            os.path.join(os.getcwd(), "temp/dataset", "cifar10"),
            train=False,
            download=True,
            transform=test_transform,
        )

    elif config['name'] == "CIFAR100":
        dataset = CIFAR100(
            os.path.join(os.getcwd(), "temp/dataset", "cifar100"),
            train=True,
            download=True,
            transform=tran_transform,
        )
        test_dataset = CIFAR100(
            os.path.join(os.getcwd(), "temp/dataset", "cifar100"),
            train=False,
            download=True,
            transform=test_transform,
        )

    elif config['name'] == "SVHN":
        dataset = SVHN(
            os.path.join(os.getcwd(), "temp/dataset", "svhn"),
            split="train",
            download=True,
            transform=tran_transform,
        )
        test_dataset = SVHN(
            os.path.join(os.getcwd(), "temp/dataset", "svhn"),
            split="test",
            download=True,
            transform=test_transform,
        )

    else:
        dataset, test_dataset = None, None

    return dataset, test_dataset


def logit_transform(image, lam=1e-6):
    image = lam + (1 - 2 * lam) * image
    return torch.log(image) - torch.log1p(-image)


def data_transform(config, X):
    if config['uniform_dequantization']:
        X = X / 256.0 * 255.0 + torch.rand_like(X) / 256.0
    if config['gaussian_dequantization']:
        X = X + torch.randn_like(X) * 0.01

    if config['rescaled']:
        X = 2 * X - 1.0
    elif config['logit_transform']:
        X = logit_transform(X)

    # if hasattr(config, "image_mean"):
    #     return X - config.image_mean.to(X.device)[None, ...]

    return X


def inverse_data_transform(config, X):
    # if hasattr(config, "image_mean"):
    #     X = X + config.image_mean.to(X.device)[None, ...]

    if config['logit_transform']:
        X = torch.sigmoid(X)
    elif config['rescaled']:
        X = (X + 1.0) / 2.0

    return torch.clamp(X, 0.0, 1.0)