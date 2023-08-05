#!/usr/bin/python3

# The script to generate dataset by fitting rotated/scaled src objects into targets.
#
# Usage:
#   python src_rotate_resize_to_targets.py \
#     --num_outputs 5 --tiles_per_img 4 \
#     --tile_width 128 --tile_height 128
#     --low_obj_width 15 --upper_obj_width 60 \
#     --src_dir <path_of_augmented_img_dir> --targets_dir <path_of_target_img_dir> \
#     --output_dir <output_img_path>

from src.img_processing.augmentation import overlay
from src.img_processing.augmentation import rotation_resizing_cropping
from src.img_processing.editing import cropping
from src.img_processing.editing import random_selection
from src.img_processing.io import imgread
from src.img_processing.tiling import random_tiling

import argparse
import itertools
import logging
import os
import pathlib
import random
import shutil
import sys

import skimage
import skimage.io


def parse_args(argv):
    # Parse input arguments.
    parser = argparse.ArgumentParser(description='Fit Sources into Targets')
    parser.add_argument('-s', '--src_dir', dest='src_dir_path',
        help='Folder with source object images.', required=True)
    parser.add_argument('-d', '--targets_dir', dest='targets_dir_path',
        help='Folder with target images.', required=True)
    parser.add_argument('-o', '--output_dir', dest='output_dir_path',
        help='Folder with target images.', required=True)

    parser.add_argument('-n', '--num_outputs', dest='num_of_outputs',
        help='Total number of augmented images.', required=False,
        type=int, default=1)
    parser.add_argument('-i', '--tiles_per_img', dest='num_tiles_per_image',
        help='Number of cropped tiles per one target image.', required=False,
        type=int, default=4)

    parser.add_argument('-w', '--tile_width', dest='tile_width',
        help='Width of the cropped tile.', required=True, type=int)
    parser.add_argument('-t', '--tile_height', dest='tile_height',
        help='Height of the cropped tile.', required=True, type=int)

    parser.add_argument('-l', '--low_obj_width', dest='low_object_width',
        help='Low boundary of augmented object width.', required=True, type=int)
    parser.add_argument('-u', '--upper_obj_width', dest='upper_object_width',
        help='Height boundary of augmented object width.', required=True, type=int)

    parser.add_argument('-a', '--alpha_threshold', dest='alpha_channel_threshold',
        help='Transparency threshold for smoothing [0..255].', required=False,
        type=int, default=215)

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

    shutil.rmtree(output_dir_path)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    num_of_outputs = parsed_args.num_of_outputs
    num_tiles_per_image = parsed_args.num_tiles_per_image
    if num_tiles_per_image <= 0 and num_of_outputs <= 0:
        logging.error('Number of tiles per one image or outputs is 0 or negative.')
        return os.EX_NOINPUT

    tile_width = parsed_args.tile_width
    tile_height = parsed_args.tile_height
    if tile_width <= 0 or tile_height <= 0:
        logging.error('Tile width or height is negative.')
        return os.EX_NOINPUT

    lower_bound_of_object_width = parsed_args.low_object_width
    upper_bound_of_object_width = parsed_args.upper_object_width
    if not (0 < lower_bound_of_object_width < tile_width and
            0 < upper_bound_of_object_width < tile_width):
        logging.error('Tile width or height is negative.')
        return os.EX_NOINPUT
    if lower_bound_of_object_width > upper_bound_of_object_width:
        logging.error('Lower bound of augmented object width is greater then the upper one.')
        return os.EX_NOINPUT

    alpha_channel_threshold = parsed_args.alpha_channel_threshold
    if not (0 <= alpha_channel_threshold <= 255):
        logging.error('Incorrect source object transparency threshold.')
        return os.EX_NOINPUT

    target_image_iter = itertools.cycle(imgread.load_images_from_dir(targets_dir_path))
    src_obj_img_iter = itertools.cycle(imgread.load_images_from_dir(src_dir_path))

    output_idx = 0
    while output_idx < num_of_outputs:
        for img_tile_idx in range(num_tiles_per_image):
            target_image = next(target_image_iter)
            src_obj_img = next(src_obj_img_iter)

            # Prepare the target tile.
            tile_top_left_row, tile_top_left_col = (
                random_tiling.get_random_tile_row_col(target_image.rgba, tile_width, tile_height))
            tile_rgba = cropping.crop_rgba(
                target_image.rgba, tile_top_left_row, tile_top_left_col, tile_width, tile_height)

            # Take next augmented source object.
            angle_in_degrees = random.randint(0, 180)
            scaled_width_in_pixels = random.randint(
                lower_bound_of_object_width, upper_bound_of_object_width + 1)
            augmented_obj_rgba = rotation_resizing_cropping.rotate_resize_crop_rgba_img(
                src_obj_img.rgba, angle_in_degrees, scaled_width_in_pixels,
                alpha_channel_threshold)

            # Find random place in the tile.
            augmented_obj_center_row, augmented_obj_center_col = (
                random_selection.get_random_row_col(tile_rgba,
                    augmented_obj_rgba.shape[1] // 2, augmented_obj_rgba.shape[0] // 2))

            target_with_augmented_rgba, binary_mask = overlay.fit_into_largest(
                tile_rgba, augmented_obj_rgba,
                augmented_obj_center_row, augmented_obj_center_col,
                alpha_channel_threshold)

            output_img_suffix = f'.a{angle_in_degrees}.w{scaled_width_in_pixels}'

            target_with_augmented_img_path = output_dir_path.joinpath(
                    target_image.path.stem + output_img_suffix + target_image.path.suffix)
            binary_mask_path = output_dir_path.joinpath(
                    target_image.path.stem + output_img_suffix + 'mask.png')
            skimage.io.imsave(str(target_with_augmented_img_path), target_with_augmented_rgba)
            skimage.io.imsave(str(binary_mask_path), skimage.img_as_uint(binary_mask))

            output_idx += 1
            if output_idx % 50 == 0:
                logging.info(f"{output_idx} outputs generated.")
            if output_idx >= num_of_outputs:
                break

    if output_idx % 50 != 0:
        logging.info(f"{output_idx} outputs generated.")
    return os.EX_OK


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    sys.exit(main(sys.argv))
