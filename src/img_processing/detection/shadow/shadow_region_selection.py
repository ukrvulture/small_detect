#!/usr/bin/python3

# Methods for shadow region selection.


import numpy as np
import skimage.color
import skimage.io


def get_darker_then_most_of_rest(
    img_rgba, binary_masks,
    total_brightest_area_min_share=0.5,
    target_area_max_quantile=0.8, other_area_min_quantile=0.2,
    min_non_transparent_area_mask_sum=10,
    alpha_channel_threshold=150):
    """Gets the regions which are darker than most of all the rest rfegions.

    Args:
      img_rgba: Source image RGBa-array.
      binary_masks: List of region binary masks.
      total_brightest_area_min_share: Min share of regions lighter than the darker ones
      target_area_max_quantile: Color percentile of dark regions to use at comparison.
      other_area_min_quantile: Color percentile of bright regions to use at comparison.
      min_non_transparent_area_mask_sum: Min size of a region to filter the smallest ones.
      alpha_channel_threshold: Threshold to filter out transparent pixels [0..255].

    Returns:
      List af the darkest region binary masks.
    """
    non_transparent_pixels = img_rgba[:, :, 3] >= alpha_channel_threshold
    img_gray = skimage.color.rgb2gray(skimage.color.rgba2rgb(img_rgba))

    total_area_sum = 0
    non_transparent_areas = []
    non_transparent_area_sums = []
    area_max_quantiles = []
    area_min_quantiles = []

    for area_mask in binary_masks:
        non_transparent_area_mask = np.logical_and(area_mask, non_transparent_pixels)
        non_transparent_area_mask_sum = non_transparent_area_mask.sum()
        if non_transparent_area_mask_sum < min_non_transparent_area_mask_sum:
            continue
        total_area_sum += non_transparent_area_mask_sum
        non_transparent_areas.append(non_transparent_area_mask)
        non_transparent_area_sums.append(non_transparent_area_mask_sum)
        area_max_quantiles.append(np.quantile(
            img_gray[non_transparent_area_mask], target_area_max_quantile))
        area_min_quantiles.append(np.quantile(
            img_gray[non_transparent_area_mask], other_area_min_quantile))

    darker_area_masks = []
    for target_area_idx, target_area_mask in enumerate(non_transparent_areas):
        other_area_sum = 0
        for other_area_idx, other_area_mask in enumerate(non_transparent_areas):
            if target_area_idx == other_area_idx:
                continue

            if area_max_quantiles[target_area_idx] <= area_min_quantiles[other_area_idx]:
                other_area_sum += non_transparent_area_sums[other_area_idx]
        if other_area_sum / total_area_sum > total_brightest_area_min_share:
            darker_area_masks.append(target_area_mask)
    return darker_area_masks
