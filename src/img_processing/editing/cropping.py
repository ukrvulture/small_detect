#!/usr/bin/python3

# Methods for tile cropping.


def crop_rgba(img_rgba, top_left_row, top_left_column, tile_width, tile_height):
    """Crops the tile in RGBa-image.

    Args:
      img_rgba: Source image RGBa-array.
      top_left_row: Row of top left corner of the cropped image in the source image.
      top_left_column: Row of top left corner of the cropped image in the source image.
      tile_width: Width of cropped tile.
      tile_height: Height of cropped tile.

    Returns:
      Cropped RGBa-array.
    """
    img_width = img_rgba.shape[1]
    img_height = img_rgba.shape[0]

    row, column = max(0, top_left_row), max(0, top_left_column)
    width, height = min(tile_width, img_width), min(tile_height, img_height)

    return img_rgba[row:row + height, column:column + width, :]
