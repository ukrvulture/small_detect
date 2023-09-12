#!/usr/bin/python3

import numpy as np
import skimage.io


class ImageFile:
    """Image not loaded to the memory."""

    def __init__(self, path=''):
        self.path = path

    def __repr__(self):
        return str(self.path)

    def __eq__(self, other):
        return self.path == other.path

    def __hash__(self):
        return hash((self.path,))

    def load(self):
        try:
            img = RawImage(path=self.path, rgba=skimage.io.imread(str(self.path)))
            img.add_alpha_if_absent()
            return img
        except ValueError:
            return None


class RawImage:
    """Image pixels and meta-data representation."""

    def __init__(self, path='', rgba=None):
        self.path = path
        self.rgba = rgba

    def __repr__(self):
        return str(self.path)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self.rgba:
            self.rgba.resize(0, 0, refcheck=False)

    def add_alpha_if_absent(self):
        if 2 < len(self.rgba.shape) and self.rgba.shape[-1] == 3:
            self.rgba = np.insert(self.rgba, 3, 255, axis=2)

    def clear_half_transparent_pixels(self, alpha_channel_threshold):
        transparent_pixels = self.rgba[:, :, 3] < alpha_channel_threshold
        self.rgba[transparent_pixels] = [0, 0, 0, 0]
