import random

import pandas as pd
import torchvision.transforms
from PIL import ImageOps
from torchvision.utils import make_grid


def annotate_img(img, is_correct, img_size=512, border_size=10):
    cropper = torchvision.transforms.CenterCrop(img_size - border_size * 2 - 4)
    img = cropper(img)
    if border_size > 0:
        border_color = 'green' if is_correct else 'red'
        img = ImageOps.expand(img, border=(border_size, border_size, border_size, border_size), fill=border_color)
    img = ImageOps.expand(img, border=(2, 2, 2, 2), fill='white')

    return torchvision.transforms.ToTensor()(img)


def visualize_similarity_matrix(similarity_matrix: pd.DataFrame, n_col: int, n_items: int,
                                read_img_fn, is_correct_fn, img_size=512, border_size=10):

    column_idxs = random.sample(range(1, len(similarity_matrix.columns)), k=n_col)

    images = []
    for col in column_idxs:
        col_images = []
        col_name = similarity_matrix.columns[col]
        records = similarity_matrix[col_name].nlargest(n_items)
        img = read_img_fn(col_name)
        col_images.append(annotate_img(img, is_correct=True, img_size=img_size, border_size=0))
        for key, value in records.items():
            img = read_img_fn(key)
            is_correct = is_correct_fn(col_name, key, value)
            col_images.append(annotate_img(img, is_correct, img_size=img_size, border_size=border_size))
        images.append(col_images)

    grid_images = []
    for i in range(len(images[0])):
        if i == 1:
            continue
        for j in range(len(images)):
            grid_images.append(images[j][i])

    # make grid from the input images
    grid = make_grid(grid_images, nrow=n_col)

    # display result
    img = torchvision.transforms.ToPILImage()(grid)
    return img

