#!/usr/bin/python3

# The script to fit one image into another.
#
# Usage:
#   python adjust_overlay.py \+

from src.img_processing.augmentation import overlay

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
    parser = argparse.ArgumentParser(description='Image Overlay')
    parser.add_argument('-f', '--img_to_fit', dest='img_path_to_fit_into',
        help='Image to fit into.', required=True)
    parser.add_argument('-i', '--insert_img', dest='inserted_img_path',
        help='Image to overlay.', required=True)
    parser.add_argument('-o', '--output_img', dest='output_img_path',
        help='Path to the overlaid image.', required=True)

    parser.add_argument('-r', '--center_row', dest='center_row_of_overlay',
        help='Row of center where to overlay.', required=True, type=int)
    parser.add_argument('-c', '--center_column', dest='center_column_of_overlay',
        help='Column of center where to overl.', required=True, type=int)
    return parser.parse_args(argv[1:])


def main(argv):
    parses_args = parse_args(argv)
    img_path_to_fit_into = pathlib.Path(parses_args.img_path_to_fit_into)
    inserted_img_path = pathlib.Path(parses_args.inserted_img_path)
    output_img_path = pathlib.Path(parses_args.output_img_path)

    # Read image arrays.
    img_rgba_to_fit_into = skimage.io.imread(str(img_path_to_fit_into))
    width_of_img_to_fit = img_rgba_to_fit_into.shape[1]
    height_of_img_to_fit = img_rgba_to_fit_into.shape[0]
    if img_rgba_to_fit_into.shape[-1] == 3:
        img_rgba_to_fit_into = np.insert(img_rgba_to_fit_into, 3, 255, axis=2)
    inserted_img_rgba = skimage.io.imread(str(inserted_img_path))
    if inserted_img_rgba.shape[-1] == 3:
        inserted_img_rgba = np.insert(inserted_img_rgba, 3, 255, axis=2)

    center_row_of_overlay = parses_args.center_row_of_overlay
    center_column_of_overlay = parses_args.center_column_of_overlay
    if center_row_of_overlay < 0 or center_row_of_overlay >= width_of_img_to_fit:
        logging.error(f'Wrong {center_row_of_overlay} row offset.')
        return os.EX_NOINPUT
    if center_column_of_overlay < 0 or center_column_of_overlay >= height_of_img_to_fit:
        logging.error(f'Wrong {height_of_img_to_fit} column offset.')
        return os.EX_NOINPUT

    overlaid_img_rgba = overlay.fit_into_largest(
        img_rgba_to_fit_into, inserted_img_rgba,
        center_row_of_overlay, center_column_of_overlay)
    skimage.io.imsave(str(output_img_path), overlaid_img_rgba)

    return os.EX_OK

if __name__ == '__main__':
    sys.exit(main(sys.argv))
