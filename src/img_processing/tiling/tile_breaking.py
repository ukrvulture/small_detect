#!/usr/bin/python3

# Random tile cropping methods.

from src.img_processing.base import image_parts

import random


def any_tile_fit(tile_shape, target_shape, target_paddings=image_parts.Paddings.zeros()):
    """Checks that a tile can fit the target image with given paddings.

    Args:
      tile_shape: Cropped tile {height, width}.
      target_shape: Target image shape.
      target_paddings: Left, right, top and bottom paddings in the target image.

    Returns:
      True if the tile of given size can be cropped from the specified image.
    """
    tile_width, tile_height = tile_shape[1], tile_shape[0]
    img_width, img_height = target_shape[1], target_shape[0]

    return (
        target_paddings.left <= img_width - max(tile_width, target_paddings.right) and
        target_paddings.top <= img_height - max(tile_height, target_paddings.bottom))


def get_random_tile_row_col(
        tile_shape, target_shape, target_paddings=image_parts.Paddings.zeros()):
    """Gets row and column of the randomly cropped tile.

    Args:
      c: Cropped tile {height, width}.
      target_shape: Target image {height, width}.
      target_paddings: Left, right, top and bottom paddings in the target image.

    Returns:
      Row and column offsets of the randomly cropped tile to fit the source image.
    """
    tile_width, tile_height = tile_shape[1], tile_shape[0]
    img_width, img_height = target_shape[1], target_shape[0]
    if not any_tile_fit(tile_shape, target_shape, target_paddings):
        return 0, 0

    random_tile_row = random.randrange(
        target_paddings.top, img_height - max(tile_height, target_paddings.bottom))
    random_tile_col = random.randrange(
        target_paddings.left, img_width - max(tile_width, target_paddings.right))
    return random_tile_row, random_tile_col
