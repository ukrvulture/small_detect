#!/usr/bin/python3

# The script to generate dataset by fitting rotated/scaled src objects into targets.
#
# Usage:
#   python src_rotate_resize_to_targets.py \

from src.img_processing.io import imgread
from src.img_processing.base import image

import argparse
import logging
import os
import pathlib
import sys


def parse_args(argv):
    # Parse input arguments.
    parser = argparse.ArgumentParser(description='Fit Sources into Targets')
    parser.add_argument('-s', '--src_dir', dest='src_dir_path',
        help='Folder with source object images.', required=True)
    parser.add_argument('-t', '--targets_dir', dest='targets_dir_path',
        help='Folder with target images.', required=True)
    parser.add_argument('-o', '--output_dir', dest='output_dir_path',
        help='Folder with target images.', required=True)

    return parser.parse_args(argv[1:])


def main(argv):
    parsed_args = parse_args(argv)
    src_dir_path = pathlib.Path(parsed_args.src_dir_path)
    targets_dir_path = pathlib.Path(parsed_args.targets_dir_path)
    output_dir_path = pathlib.Path(parsed_args.output_dir_path)
    if not src_dir_path.is_dir() or not targets_dir_path.is_dir():
        logging.error(f'{src_dir_path} or {targets_dir_path} is not a dir.')
        return os.EX_NOINPUT
    if output_dir_path.is_file():
        logging.error(f'{output_dir_path} is a file.')
        return os.EX_NOINPUT

    for file_path, img_pixels in imgread.load_images_from_dir(targets_dir_path):
        img = image.RawImage(path=file_path, rgba=img_pixels)
        print(img)
        pass  # process here

    return os.EX_OK


if __name__ == '__main__':
    sys.exit(main(sys.argv))
