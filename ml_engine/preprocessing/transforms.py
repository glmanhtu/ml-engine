import random

import albumentations as A
import numpy as np
import torchvision
import torchvision.transforms.functional as F
from PIL import Image


class ACompose:
    def __init__(self, a_transforms):
        self.transform = A.Compose(a_transforms)

    def __call__(self, image):
        np_img = np.asarray(image)
        np_img = self.transform(image=np_img)['image']
        return Image.fromarray(np_img)


class PadCenterCrop(object):
    def __init__(self, size, pad_if_needed=False, fill=0, padding_mode='constant'):
        if isinstance(size, (int, float)):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.pad_if_needed = pad_if_needed
        self.padding_mode = padding_mode
        self.fill = fill

    def __call__(self, img):

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(img, (self.size[1] - img.size[0], 0), self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(img, (0, self.size[0] - img.size[1]), self.fill, self.padding_mode)

        return F.center_crop(img, self.size)


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


def split_patches(im: Image, n_cols, n_rows):
    width = im.width // n_cols
    height = im.height // n_rows
    patches = []
    for i in range(n_rows):
        for j in range(n_cols):
            box = (j*width, i*height, (j+1)*width, (i+1)*height)
            patches.append(im.crop(box))
    return patches


class RandomResize:
    def __init__(self, img_size, ratio=(0.6, 1.0)):
        self.ratio = ratio
        self.img_size = img_size

    def __call__(self, img):
        ratio = random.uniform(self.ratio[0], self.ratio[1])
        w, h = int(img.width * ratio), int(img.height * ratio)
        cropper = torchvision.transforms.Resize((h, w))
        return cropper(img)
