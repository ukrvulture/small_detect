#!/usr/bin/python3

# Random tile cropping methods.

import random


def get_random_tile_row_col(img_rgba, tile_width, tile_height):
    """Gets row and column of the randomly cropped tile.

    Args:
      img_rgba: Source image RGBa-array.
      tile_width: Width of cropped tile.
      tile_height: Height of cropped tile.

    Returns:
      Row and column offsets of the randomly cropped tile to fit the source image.
    """
    img_width = img_rgba.shape[1]
    img_height = img_rgba.shape[0]
    if img_width <= tile_width and img_height <= tile_height:
        return img_rgba

    tile_top_left_row = 0
    if tile_width < img_width:
        tile_top_left_row = random.randrange(img_width - tile_width)

    tile_top_left_col = 0
    if tile_height < img_height:
        tile_top_left_col = random.randrange(img_height - tile_height)

    return tile_top_left_row, tile_top_left_col
