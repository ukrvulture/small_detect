#!/usr/bin/python3

import numpy as np
import os
import pathlib
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

    @property
    def stem(self):
        return (self.path.stem if isinstance(self.path, pathlib.Path) else
                os.path.splitext(os.path.basename(self.path))[0])

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

    @property
    def shape(self):
        return self.rgba.shape if self.rgba is not None else (0, 0)

    def add_alpha_if_absent(self):
        if 2 < len(self.rgba.shape) and self.rgba.shape[-1] == 3:
            self.rgba = np.insert(self.rgba, 3, 255, axis=2)

    def clear_half_transparent_pixels(self, alpha_channel_threshold):
        transparent_pixels = self.rgba[:, :, 3] < alpha_channel_threshold
        self.rgba[transparent_pixels] = [0, 0, 0, 0]
