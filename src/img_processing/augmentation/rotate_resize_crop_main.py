#!/usr/bin/python3

# The script to generate all possible rotations of a template image.
#
# Usage:
#   python rotate_resize_crop_main.py \
#     --degrees 30 --width 40 --alpha_threshold 215 \
#     --input_img <input_img_path> --output_dir <input_dir_path>

from src.img_processing.augmentation import rotate_resize_crop

import argparse
import logging
import os
import pathlib
import sys

import numpy as np
import skimage.color
import skimage.io
import skimage.transform
import skimage.util


def parse_args(argv):
    # Parse input arguments.
    parser = argparse.ArgumentParser(description='Template Image Rotation')
    parser.add_argument('-a', '--degrees', dest='angle_in_degrees',
        help='Angle in degrees.', required=True, type=int)
    parser.add_argument('-w', '--width', dest='width_in_pixels',
        help='Width in pixels.', required=True, type=int)
    parser.add_argument('-t', '--alpha_threshold', dest='alpha_channel_threshold',
        help='Transparency threshold [0..255].', required=False, type=int)

    parser.add_argument('-i', '--input_img', dest='input_img_path',
        help='Path to the input image.', required=True)
    parser.add_argument('-o', '--output_dir', dest='output_dir_path',
        help='Path to the output dir.', required=True)
    return parser.parse_args(argv[1:])


def main(argv):
    parses_args = parse_args(argv)
    angle_in_degrees = parses_args.angle_in_degrees
    scaled_width_in_pixels = parses_args.width_in_pixels
    alpha_channel_threshold = (
        parses_args.alpha_channel_threshold
        if parses_args.alpha_channel_threshold is not None else 180)

    input_img_path = pathlib.Path(parses_args.input_img_path)
    output_dir_path = pathlib.Path(parses_args.output_dir_path)
    if not input_img_path.is_file() or not output_dir_path.is_dir():
        logging.error(f'{input_img_path} or {output_dir_path} is wrong.')
        return os.EX_NOINPUT

    img_rgba = skimage.io.imread(str(input_img_path))
    if img_rgba.shape[-1] == 3:
        img_rgba = np.insert(img_rgba, 3, 255, axis=2)

    output_rgba = rotate_resize_crop.rotate_resize_crop_rgba_img(
        img_rgba, angle_in_degrees, scaled_width_in_pixels,
        alpha_channel_threshold)

    output_img_suffix = f'.a{angle_in_degrees}.w{scaled_width_in_pixels}'

    output_img_path = output_dir_path.joinpath(
        input_img_path.stem + output_img_suffix + input_img_path.suffix)
    skimage.io.imsave(str(output_img_path), output_rgba)

    return os.EX_OK


if __name__ == '__main__':
    sys.exit(main(sys.argv))
