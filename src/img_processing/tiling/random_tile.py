#!/usr/bin/python3

# Random tile cropping methods.

import random


def get_random_tile_row_col(target_shape, tile_width, tile_height):
    """Gets row and column of the randomly cropped tile.

    Args:
      target_shape: Target image shape.
      tile_width: Width of cropped tile.
      tile_height: Height of cropped tile.

    Returns:
      Row and column offsets of the randomly cropped tile to fit the source image.
    """
    img_width = target_shape[1]
    img_height = target_shape[0]
    if img_width <= tile_width and img_height <= tile_height:
        return 0, 0

    tile_top_left_row = 0
    if tile_width < img_width:
        tile_top_left_row = random.randrange(img_height - tile_height)

    tile_top_left_col = 0
    if tile_height < img_height:
        tile_top_left_col = random.randrange(img_width - tile_width)

    return tile_top_left_row, tile_top_left_col
