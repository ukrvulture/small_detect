#!/usr/bin/python3

import logging

# Methods for image overlaying.

import cv2 as cv
import numpy as np


def fit_into_largest(
    larger_rgba, smaller_rgba, center_row_in_larger, center_col_in_larger,
    smaller_alpha_mask_thold=None):
    """Fits smaller image into the larger one.

    Args:
      larger_rgba: Larger image to fit in.
      smaller_rgba: Smaller image to overlay.
      center_row_in_larger: Row-offset of the smaller image center in the largest one.
      center_col_in_larger: Column-offset of the smaller image center in the largest one.
      smaller_alpha_mask_thold: Alpha channel threshold [0..255] to create a mask of
                                the smaller image outlining it in the bigger one (optional).

    Returns:
      Overlaid RGBa-array.
    """
    smaller_rgba_width = smaller_rgba.shape[1]
    smaller_rgba_height = smaller_rgba.shape[0]

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

    # noinspection PyUnresolvedReferences
    blended_rgba = cv.addWeighted(larger_rgba, 1.0, fit_smaller_rgba, 0.7, 0)

    binary_mask = None
    if smaller_alpha_mask_thold is not None:
        binary_mask = fit_smaller_rgba[:, :, 3] >= smaller_alpha_mask_thold
        binary_mask = np.array(binary_mask).astype(np.uint8) * 255

    return blended_rgba, binary_mask
