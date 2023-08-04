#!/usr/bin/python3

# Random position selection methods.

import random


def get_random_row_col(img_rgba, width_margin, height_margin):
    """Gets random row and column within the image margins.

    Args:
      img_rgba: Source image RGBa-array.
      width_margin: Left and right margin.
      height_margin: Top and bottom margin.

    Returns:
      Random row and column or image centroid pos, if double margin is greater than
      appropriate image dimension.
    """
    img_width = img_rgba.shape[1]
    img_height = img_rgba.shape[0]

    if 2 * width_margin < img_width - 1:
        random_row = random.randint(width_margin, img_width - width_margin)
    else:
        random_row = img_width // 2

    if 2 * height_margin < img_height - 1:
        random_col = random.randint(height_margin, img_height - height_margin)
    else:
        random_col = img_height // 2

    return random_row, random_col
