#!/usr/bin/python3

# The script to generate dataset by fitting rotated/scaled src objects into targets.
#
# Usage:
#   python src_rotate_resize_to_target_main.py \
#     --num_outputs 5 --tiles_per_img 4 \
#     --tile_width 128 --tile_height 128 \
#     --low_obj_width 15 --upper_obj_width 60 \
#     --src_dir <path_of_augmented_img_dir> --targets_dir <path_of_target_img_dir> \
#     --output_dir <output_img_path>

import os, sys
SCRIPT_DIRS = os.path.dirname(os.path.abspath(__file__)).split(os.sep)
sys.path.append(os.path.join(os.sep, *SCRIPT_DIRS[:SCRIPT_DIRS.index('src')]))

from src.img_processing.augmentation import adjust_overlay
from src.img_processing.augmentation import rotate_resize_crop
from src.img_processing.base import image_parts
from src.img_processing.editing import cropping
from src.img_processing.editing import random_selection
from src.img_processing.io import imgread
from src.img_processing.tiling import tile_breaking

from src.machine_learning.datasets.augmentation.records.angle_and_size_augmentation import AngledResizedSrcInTargetFileDesc
from src.machine_learning.datasets.augmentation.records.sample_file_record import SampleFileRecord

import argparse
import collections
import logging
import os
import re
import pathlib
import random
import shutil

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
                        help='Output folder.', required=True)

    parser.add_argument('-n', '--num_outputs', dest='num_of_outputs',
                        help='Total number of augmented images.', required=False,
                        type=int, default=1)
    parser.add_argument('-i', '--tiles_per_img', dest='num_tiles_per_image',
                        help='Number of cropped tiles per one target image.', required=False,
                        type=int, default=4)

    parser.add_argument('-p', '--target_paddings', dest='target_paddings',
                        help='Comma-separated left, right, top, bottom target image paddings.',
                        required=False, default="0,0,0,0")

    parser.add_argument('-w', '--tile_width', dest='tile_width',
                        help='Width of the cropped tile.', required=True, type=int)
    parser.add_argument('-t', '--tile_height', dest='tile_height',
                        help='Height of the cropped tile.', required=True, type=int)

    parser.add_argument('-l', '--low_src_width', dest='low_src_obj_width',
                        help='Low boundary of augmented object width.', required=True, type=int)
    parser.add_argument('-u', '--upper_src_width', dest='upper_src_obj_width',
                        help='Height boundary of augmented object width.', required=True, type=int)

    parser.add_argument('-a', '--alpha_threshold', dest='alpha_channel_threshold',
                        help='Transparency threshold for smoothing [0..255].', required=False,
                        type=int, default=230)

    return parser.parse_args(argv[1:])


def main(argv):
    parsed_args = parse_args(argv)
    src_dir_path = pathlib.Path(parsed_args.src_dir_path)
    targets_dir_path = pathlib.Path(parsed_args.targets_dir_path)
    output_dir_path = pathlib.Path(parsed_args.output_dir_path)
    if not src_dir_path.is_dir() or not targets_dir_path.is_dir():
        logging.error('%s or %s is not a dir.', src_dir_path, targets_dir_path)
        return os.EX_NOINPUT
    if output_dir_path.is_file():
        logging.error('%s is a file.', output_dir_path)
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

    target_paddings = image_parts.Paddings.parse_from_csv(parsed_args.target_paddings)
    if not target_paddings:
        logging.error('Invalid %s paddings.', target_paddings)
        return os.EX_NOINPUT

    lower_bound_of_src_obj_width = parsed_args.low_src_obj_width
    upper_bound_of_src_obj_width = parsed_args.upper_src_obj_width
    if not (0 < lower_bound_of_src_obj_width < tile_width and
            0 < upper_bound_of_src_obj_width < tile_width):
        logging.error('Tile width or height is negative.')
        return os.EX_NOINPUT
    if lower_bound_of_src_obj_width > upper_bound_of_src_obj_width:
        logging.error('Lower bound of augmented object width is greater then the upper one.')
        return os.EX_NOINPUT

    alpha_channel_threshold = parsed_args.alpha_channel_threshold
    if not (0 <= alpha_channel_threshold <= 255):
        logging.error('Incorrect source object transparency threshold.')
        return os.EX_NOINPUT

    labeling_file_regexp = re.compile(r'(.*\.ini|.*\.xcf|.*\.psd)')
    target_image_files = collections.deque(imgread.list_image_file_from_dir(
        targets_dir_path, recursive=True,
        ignore_non_images=True, ignored_file_regexp=labeling_file_regexp))
    src_obj_img_files = list(imgread.list_image_file_from_dir(
        src_dir_path, recursive=True,
        ignore_non_images=True, ignored_file_regexp=labeling_file_regexp))

    output_idx = 0
    failed_output_idx = 0
    while output_idx < num_of_outputs and target_image_files and src_obj_img_files:
        target_image_file = target_image_files[0]
        target_image = target_image_file.load()
        target_image_files.rotate(-1)
        if not target_image:
            logging.error('Malformed target %s.', target_image_files[0])
            target_image_files.pop()  # malformed
            continue

        if not (target_paddings.within_width(target_image.shape[1]) and
                target_paddings.within_height(target_image.shape[0])):
            logging.error('%s padding mismatch %s.', target_paddings, target_image_file)
            target_image_files.pop()
            continue

        img_tile_idx = 0
        while img_tile_idx < num_tiles_per_image and src_obj_img_files:
            src_obj_img_file = random.choice(src_obj_img_files)
            src_obj_img = src_obj_img_file.load()
            if (not src_obj_img or not tile_breaking.any_tile_fit(
                    (tile_height, tile_width), target_image.shape, target_paddings)):
                logging.error('Malformed or too small source %s.', src_obj_img_file)
                src_obj_img_files.remove(src_obj_img_file)
                continue
            img_tile_idx += 1

            aug_sample_desc = AngledResizedSrcInTargetFileDesc.from_src_and_target_file(
                src_img_file=src_obj_img_file, target_image_file=target_image_file)

            aug_sample_desc.tile_top_left_row, aug_sample_desc.tile_top_left_col = (
                tile_breaking.get_random_tile_row_col(
                    (tile_height, tile_width), target_image.shape, target_paddings))
            aug_sample_desc.tile_width, aug_sample_desc.tile_height = tile_width, tile_height
            aug_sample_desc.angle_in_degrees = random.randint(0, 180)
            aug_sample_desc.scaled_width_in_pixels = random.randint(
                min(src_obj_img.shape[1], lower_bound_of_src_obj_width),
                min(src_obj_img.shape[1], upper_bound_of_src_obj_width) + 1)

            # noinspection PyBroadException
            try:
                augment_source_in_target(
                    target_image, src_obj_img, alpha_channel_threshold,
                    aug_sample_desc, output_dir_path)
            except Exception:
                logging.exception('Augmentation %s.', aug_sample_desc)
                failed_output_idx += 1

            output_idx += 1
            if output_idx % 25 == 0:
                logging.info(f"{output_idx} outputs processed.")
            if output_idx >= num_of_outputs:
                break

    if not target_image_files or not src_obj_img_files:
        logging.warning('No target or source images.')
    if output_idx % 25 != 0:
        logging.info('%d output sets generated%s.', output_idx - failed_output_idx,
            '({} failed)'.format(failed_output_idx) if failed_output_idx else '')
    return os.EX_OK


def augment_source_in_target(
        target_image, src_obj_img, alpha_channel_threshold,
        aug_sample_desc, output_dir_path):
    """Overlays source image over the target with the specified augmentations.

    Args:
      target_image: Target image.
      src_obj_img: Source image (transparent pixels to outline the contour).
      alpha_channel_threshold: Threshold to filter out transparent pixels [0..255].
      aug_sample_desc: Meta-data of augmented sample.
      output_dir_path: Output folder.
    """
    # Prepare the target tile.
    tile_rgba = cropping.crop_rgba(
        target_image.rgba,
        aug_sample_desc.tile_top_left_row, aug_sample_desc.tile_top_left_col,
        aug_sample_desc.tile_width, aug_sample_desc.tile_height)

    # Take next augmented source object.
    augmented_obj_rgba = rotate_resize_crop.rotate_resize_crop_rgba_img(
        src_obj_img.rgba,
        aug_sample_desc.angle_in_degrees, aug_sample_desc.scaled_width_in_pixels,
        alpha_channel_threshold)

    # Find random place in the tile.
    augmented_obj_center_row, augmented_obj_center_col = (
        random_selection.get_random_row_col(
            tile_rgba,
            augmented_obj_rgba.shape[1] // 2, augmented_obj_rgba.shape[0] // 2))

    (target_with_augmented_rgba,
     augmented_src_rgba, binary_mask) = adjust_overlay.saliency_blend_into_largest(
        tile_rgba, augmented_obj_rgba,
        augmented_obj_center_row, augmented_obj_center_col,
        alpha_channel_threshold, tiny_img_max_side_size=20)

    file_desc_to_rgba = {
        AngledResizedSrcInTargetFileDesc.AUGMENTED_TILE : target_with_augmented_rgba,
        AngledResizedSrcInTargetFileDesc.TARGET_TILE: tile_rgba,
        AngledResizedSrcInTargetFileDesc.AUGMENTED_SRC: augmented_src_rgba,
        AngledResizedSrcInTargetFileDesc.SRC_MASK: binary_mask,
    }
    sample_file_record = SampleFileRecord(aug_sample_desc)
    for file_desc, rgba in file_desc_to_rgba.items():
        skimage.io.imsave(
            str(sample_file_record.get_saved_file_path(output_dir_path, file_desc, "png")),
            rgba, check_contrast=False)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    sys.exit(main(sys.argv))
