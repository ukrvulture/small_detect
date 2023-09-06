#!/usr/bin/python3

# Methods for image overlaying.

from src.img_processing.base import image_pool

import logging

import numpy as np
import skimage.transform


class ImageMagicProps:
    SEAMLESS_SALIENCY_BLEND_AGS = (
        '1000x0.000002+100')  # Max-IterationsXDistortion+<Print-Iterations>


def saliency_blend_into_largest(
    larger_rgba, smaller_rgba, center_row_in_larger, center_col_in_larger,
    smaller_alpha_mask_thold=None, tiny_img_max_side_size=20):
    """Blends smaller image into the larger one using ImageMagick algorithms.

    Args:
      larger_rgba: Larger image to fit in.
      smaller_rgba: Smaller image to overlay.
      center_row_in_larger: Row-offset of the smaller image center in the largest one.
      center_col_in_larger: Column-offset of the smaller image center in the largest one.
      tiny_img_max_side_size: Threshold to distinguish too small images.
      smaller_alpha_mask_thold: Alpha channel threshold [0..255] to create a mask of
                                the smaller image outlining it in the bigger one (optional).

    Returns:
      Overlaid RGBa-array, smaller image positioned
      in the bigger one before overlay and its binary mask.
    """
    with image_pool.ImagePool() as img_pool:
        smaller_rgba_width = smaller_rgba.shape[1]
        smaller_rgba_height = smaller_rgba.shape[0]
        smaller_min_size = min(smaller_rgba_width, smaller_rgba_height)

        top_left_row_in_larger = center_row_in_larger - smaller_rgba_height // 2
        top_left_col_in_larger = center_col_in_larger - smaller_rgba_width // 2

        if (top_left_row_in_larger > larger_rgba.shape[1] or
                top_left_col_in_larger > larger_rgba.shape[1]):
            logging.warning('Image inserted outside the larger one.')
            return larger_rgba

        if top_left_row_in_larger < 0:
            smaller_rgba = smaller_rgba[-top_left_row_in_larger:, :, :]
            top_left_row_in_larger = 0
        if top_left_col_in_larger < 0:
            smaller_rgba = smaller_rgba[:, -top_left_col_in_larger:, :]
            top_left_col_in_larger = 0

        smaller_rgba = smaller_rgba[
            :min(smaller_rgba.shape[0], larger_rgba.shape[0] - top_left_row_in_larger),
            :min(smaller_rgba.shape[1], larger_rgba.shape[1] - top_left_col_in_larger), :]

        fit_smaller_rgba = np.zeros(larger_rgba.shape, dtype=np.uint8)
        fit_smaller_rgba[
            top_left_row_in_larger:top_left_row_in_larger + smaller_rgba.shape[0],
            top_left_col_in_larger:top_left_col_in_larger + smaller_rgba.shape[1],
            :] = smaller_rgba

        # blended_rgba = cv.addWeighted(larger_rgba, 1.0, fit_smaller_rgba, 0.7, 0)

        src_img = img_pool.imagick_from_rgba(fit_smaller_rgba)
        target_img = img_pool.imagick_from_rgba(larger_rgba)

        if smaller_min_size > tiny_img_max_side_size:
            disk_kernel = round(smaller_min_size / 10.0, 1)
            src_img.morphology(method='erode', kernel=f'disk:{disk_kernel}', channel='alpha')
            target_img.composite(
                src_img, operator='saliency_blend',
                arguments=ImageMagicProps.SEAMLESS_SALIENCY_BLEND_AGS)
        else:
            # Average color of source image and use this color to
            # substitute black transparent pixels.
            # http://www.github.com/ImageMagick/ImageMagick/discussions/6578#discussioncomment-6899781
            src_img_to_restore_alpha = img_pool.imagick_from_imagick(src_img)
            one_pixel_avg_color_rgba = skimage.transform.resize(
                smaller_rgba, (1, 1), order=1, mode='constant', anti_aliasing=False,
                preserve_range=True).astype(np.uint8)[0, 0]
            uniform_avg_color_rgba = np.tile(
                one_pixel_avg_color_rgba, (larger_rgba.shape[0], larger_rgba.shape[1], 1))
            transparent_background_to_gradient = img_pool.imagick_from_rgba(
                uniform_avg_color_rgba)
            src_img.composite(transparent_background_to_gradient, operator='dst_over')  # gradient

            src_img.composite(src_img_to_restore_alpha, operator='copy_alpha')
            target_img.composite(
                src_img, operator='saliency_blend',
                arguments=ImageMagicProps.SEAMLESS_SALIENCY_BLEND_AGS)

        binary_mask = None
        if smaller_alpha_mask_thold is not None:
            binary_mask = fit_smaller_rgba[:, :, 3] >= smaller_alpha_mask_thold
            binary_mask = np.array(binary_mask).astype(np.uint8) * 255

        return np.array(target_img), fit_smaller_rgba, binary_mask
