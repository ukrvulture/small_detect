#!/usr/bin/python3

import numpy as np


class RawImage:
    """Image pixels and meta-data representation."""

    def __init__(self, path='', rgba=None):
        self.path = path
        self.rgba = rgba

    def __repr__(self):
        return str(self.path)

    def add_alpha_if_absent(self):
        if 2 < len(self.rgba.shape) and self.rgba.shape[-1] == 3:
            self.rgba = np.insert(self.rgba, 3, 255, axis=2)

    def clear_half_transparent_pixels(self, alpha_channel_threshold):
        transparent_pixels = self.rgba[:, :, 3] < alpha_channel_threshold
        self.rgba[transparent_pixels] = [0, 0, 0, 0]
