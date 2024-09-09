#!/usr/bin/python3

# Methods for mask up-scaling.

import numpy as np

# from skimage.transform import resize


def array_2d_split(src_array, dest_width, dest_height):
    split_by_rows = np.array_split(src_array, dest_height)
    return [np.array_split(x, dest_width, axis=1) for x in split_by_rows]


def scale_binary_mask(
        src_mask, target_width, target_height,
        borderline_val=None, borderline_ratio_thold=0.1):
    """Changes width and height of passed binary mask.

    Args:
      src_mask: Source binary mask.
      target_width: Number of columns to re-scale.
      target_height: Number of rows to re-scale.
      borderline_val: Value to fill borderline cells (skipped, if None).
      borderline_ratio_thold: Threshold of zero to one or one to zero ratio
                              to consider the area as a borderline area.

    Returns:
      Scaled binary mask and another binary mask highlighting non-borderline cells in
      the scaled mask (having zeros at borderline).
    """
    src_mask_cells = array_2d_split(src_mask.astype(int), target_width, target_height)

    scaled_mask, mask_of_mask = [], []
    for src_mask_cell_row in src_mask_cells:
        scaled_mask.append([])
        mask_of_mask.append([])
        for cell in src_mask_cell_row:
            zero_cnt, non_zero_cnt = np.count_nonzero(cell == 0), np.count_nonzero(cell)
            zero_share = zero_cnt / (zero_cnt + non_zero_cnt)
            non_zero_share = non_zero_cnt / (zero_cnt + non_zero_cnt)
            if   zero_share > borderline_ratio_thold:
                scaled_mask[-1].append(0)
                mask_of_mask[-1].append(1)
            elif non_zero_share > borderline_ratio_thold:
                scaled_mask[-1].append(1)
                mask_of_mask[-1].append(1)
            else:
                scaled_mask[-1].append(
                    borderline_val
                    if borderline_val is not None else int(non_zero_share > zero_share))
                mask_of_mask[-1].append(0)
    return (np.array(scaled_mask) * 255).astype(np.uint8), (np.array(mask_of_mask) * 255).astype(np.uint8)
