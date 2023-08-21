#!/usr/bin/python3

# The script to detect shadows using SAM neural network.
#
# Usage:
#   python sam_shadow_detection.py \
#     --dir <input_dir> --sam_pth sam_vit_b_01ec64.pth --sam_type vit_b


from src.img_processing.detection.shadow import shadow_region_selection
from src.img_processing.io import imgread
from src.img_processing.regions import region_contour
from src.machine_learning.segmentation.sam import sam_multi_mask_gen

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

MASK_FILE_EXT = '.shadow.mask.png'

def parse_args(argv):
    # Parse input arguments.
    parser = argparse.ArgumentParser(description='Shadow detection with SAM')

    parser.add_argument('-d', '--dir', dest='img_dir_path',
        help='Folder with images to detect shadows.', required=True)

    parser.add_argument('-s', '--sam_pth', dest='sam_checkpoint_path',
        help='Path to SAM pth-checkpoint.', required=True)
    parser.add_argument('-m', '--sam_type', dest='sam_model_type',
        help='SAM model type (vit_h, vit_b, etc.).', required=True)

    parser.add_argument('-a', '--alpha_threshold', dest='alpha_channel_threshold',
        help='Transparency threshold for smoothing [0..255].', required=False,
        type=int, default=150)

    return parser.parse_args(argv[1:])


class SamShadowDetectionConfig:
    def __init__(self):
        self.sam_config = sam_multi_mask_gen.SamAutoMaskGeneratorConfig()
        self.sam_config.pred_iou_thresh = 0.8
        self.sam_config.stability_score_thresh = 0.7
        self.sam_config.box_nms_thresh = 0.7
        self.sam_config.min_mask_region_area = 30

        self.total_brightest_area_min_share = 0.5
        self.target_area_max_quantile = 0.7
        self.other_area_min_quantile = 0.25

        self.min_shadow_boundary_contour_share = 0.3

        self.shadow_mask_element_min_share = 0.001
        self.total_shadow_mask_min_share = 0.01


def main(argv):
    parsed_args = parse_args(argv)

    img_dir_path = pathlib.Path(parsed_args.img_dir_path)
    sam_checkpoint_path = pathlib.Path(parsed_args.sam_checkpoint_path)
    if not img_dir_path.is_dir() or not sam_checkpoint_path.is_file():
        logging.error(f'Wrong {img_dir_path} or {sam_checkpoint_path}.')
        return os.EX_NOINPUT

    alpha_channel_threshold = parsed_args.alpha_channel_threshold
    if not (0 <= alpha_channel_threshold <= 255):
        logging.error(f'Incorrect alpha channel threshold.')
        return os.EX_NOINPUT

    shadow_detect_config = SamShadowDetectionConfig()

    sam_mask_inference = sam_multi_mask_gen.SamMultiMaskInference(
        sam_auto_mask_generator_config=shadow_detect_config.sam_config,
        sam_checkpoint=sam_checkpoint_path, sam_model_type=parsed_args.sam_model_type)

    for file_name in sorted(os.listdir(str(img_dir_path))):
        if file_name.endswith(MASK_FILE_EXT):
            os.remove(img_dir_path / file_name)

    for img in imgread.load_images_from_dir(img_dir_path):
        logging.info(f'Processing {img}.')
        img.add_alpha_if_absent()
        img.clear_half_transparent_pixels(alpha_channel_threshold)

        img_size = img.rgba.shape[0] * img.rgba.shape[1]
        transparent_pixels = img.rgba[:, :, 3] < alpha_channel_threshold

        binary_masks = sam_mask_inference.generate_all_masks(img.rgba)

        # Add target image area not covered by masks.
        total_binary_mask = np.zeros(img.rgba.shape[:2], dtype=bool)
        for binary_mask in binary_masks:
            total_binary_mask = np.logical_or(total_binary_mask, binary_mask)
        non_covered_mask_area = np.logical_and(~total_binary_mask, ~transparent_pixels)
        binary_masks.append(non_covered_mask_area)

        shadow_maks = shadow_region_selection.get_darker_then_most_of_rest(
            img.rgba, binary_masks,
            total_brightest_area_min_share=shadow_detect_config.total_brightest_area_min_share,
            target_area_max_quantile=shadow_detect_config.target_area_max_quantile,
            other_area_min_quantile=shadow_detect_config.other_area_min_quantile)

        result_mask = np.zeros(img.rgba.shape[:2], dtype=bool) if shadow_maks else None
        for shadow_mask in shadow_maks:
            total_contour_len, boundary_contour_len = region_contour.get_mask_contour_len(
                img.rgba, shadow_mask, alpha_channel_threshold, mask_extension_kernel_size=5)
            if (total_contour_len == 0 or boundary_contour_len / total_contour_len
                    < shadow_detect_config.min_shadow_boundary_contour_share):
                continue
            if shadow_mask.sum() <= img_size * shadow_detect_config.shadow_mask_element_min_share:
                continue
            result_mask = np.logical_or(result_mask, shadow_mask)

        if (result_mask is not None and
                result_mask.sum() > img_size * shadow_detect_config.total_shadow_mask_min_share):
            shadow_binary_mask_filename = img.path.stem + MASK_FILE_EXT
            shadow_binary_mask_path = img.path.resolve().parent.joinpath(
                shadow_binary_mask_filename)
            skimage.io.imsave(str(shadow_binary_mask_path), skimage.img_as_uint(result_mask))
            logging.info(f'{shadow_binary_mask_filename} is saved')

    return os.EX_OK


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    sys.exit(main(sys.argv))

