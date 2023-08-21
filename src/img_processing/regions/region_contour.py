#!/usr/bin/python3

# Region contours.

import numpy as np
from scipy import ndimage


def get_mask_contour_len(
    img_rgba, binary_mask, alpha_channel_threshold, mask_extension_kernel_size=0):
    """Gets length of curve adjacent_to the region contour.

    Args:
      img_rgba: Source image RGBa-array.
      binary_mask: Target region binary mask.
      alpha_channel_threshold: Threshold to filter out transparent pixels [0..255].
      mask_extension_kernel_size: Size of mask boundary dilation kernel needed to
                                  filter out a possible image editing noise.

    Returns:
      Total length of a curve adjacent_to the region contour.
      Length of the curve adjacent_to the region contour and passing through
      the image boundary or transparent pixels.
    """
    non_transparent_pixels = img_rgba[:, :, 3] >= alpha_channel_threshold
    binary_mask = np.logical_and(binary_mask, non_transparent_pixels)

    if mask_extension_kernel_size and mask_extension_kernel_size > 1:
        binary_mask_extension_kernel = np.ones(
            (mask_extension_kernel_size, mask_extension_kernel_size), dtype=bool)
        binary_mask = ndimage.binary_dilation(binary_mask, binary_mask_extension_kernel)

    total_contour_len, boundary_contour_len = 0, 0
    test_boundary = np.zeros(binary_mask.shape, dtype=bool)

    for row in range(0, img_rgba.shape[0]):
        for col in range(0, img_rgba.shape[1]):
            # Consider pixels near the image boundary.
            img_boundary = (
                row == 0 or row == img_rgba.shape[0] - 1 or
                col == 0 or col == img_rgba.shape[1] - 1)
            if img_boundary and binary_mask[row][col]:
                total_contour_len += 1
                boundary_contour_len += 1
                test_boundary[row, col] = True
            if img_boundary:
                continue

            # Ignore all the pixels in the mask.
            if binary_mask[row, col]:
                continue

            # Check pixels near the mask.
            adjacent_to_mask = (
                binary_mask[row - 1, col - 1] or binary_mask[row - 1, col] or
                binary_mask[row - 1, col + 1] or
                binary_mask[row + 1, col - 1] or binary_mask[row + 1, col] or
                binary_mask[row + 1, col + 1] or
                binary_mask[row, col - 1] or binary_mask[row, col + 1])
            if adjacent_to_mask:
                total_contour_len += 1
            if adjacent_to_mask and not non_transparent_pixels[row][col]:
                boundary_contour_len += 1
                test_boundary[row, col] = True
    return total_contour_len, boundary_contour_len
