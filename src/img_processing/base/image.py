#!/usr/bin/python3


class RawImage:
    """Image pixels and meta-data representation."""

    def __init__(self, path='', rgba=None):
        self.path = path
        self.rgba = rgba

    def __repr__(self):
        return str(self.path)
