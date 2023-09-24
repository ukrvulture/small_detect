#!/usr/bin/python3

from src.img_processing.editing import cropping
from src.img_processing.tiling import tile_breaking

import argparse
import os
import pathlib
import sys

import skimage.io


def parse_args(argv):
    # Parse input arguments.
    parser = argparse.ArgumentParser(description='Random Tile Cropping')
    parser.add_argument('-i', '--input_img', dest='input_img_path',
        help='Path to the input image.', required=True)
    parser.add_argument('-o', '--output_img', dest='output_img_path',
        help='Path to the image tile.', required=True)

    parser.add_argument('-w', '--width', dest='tile_width',
        help='Width of the cropped tile.', required=True, type=int)
    parser.add_argument('-t', '--height', dest='tile_height',
        help='Height of the cropped tile.', required=True, type=int)

    return parser.parse_args(argv[1:])


def main(argv):
    parses_args = parse_args(argv)
    input_img_path = pathlib.Path(parses_args.input_img_path)
    output_img_path = pathlib.Path(parses_args.output_img_path)

    tile_width = parses_args.tile_width
    tile_height = parses_args.tile_height

    img_rgba = skimage.io.imread(str(input_img_path))
    tile_top_left_row, tile_top_left_col = (
        tile_breaking.get_random_tile_row_col((tile_height, tile_width), img_rgba.shape))
    tile_rgba = cropping.crop_rgba(
        img_rgba, tile_top_left_row, tile_top_left_col, tile_width, tile_height)
    skimage.io.imsave(str(output_img_path), tile_rgba)

    return os.EX_OK


if __name__ == '__main__':
    sys.exit(main(sys.argv))
