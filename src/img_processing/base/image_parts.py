#!/usr/bin/python3


class Paddings:
    """Image paddings."""

    def __init__(self, left=0, right=0, top=0, bottom=0):
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom

    @staticmethod
    def zeros():
        return Paddings()

    @staticmethod
    def parse_from_csv(left_right_top_bottom):
        try:
            parsed_left_right_top_bottom = [
                int(val) for val in left_right_top_bottom.split(',')]
        except ValueError:
            return None

        if len(parsed_left_right_top_bottom) != 4:
            return None

        return Paddings(*parsed_left_right_top_bottom)

    def __repr__(self):
        return f'{self.left},{self.right},{self.top},{self.bottom}'

    def within_width(self, width):
        return 0 <= self.left and 0 <= self.right and self.left + self.right <= width

    def within_height(self, height):
        return 0 <= self.top and 0 <= self.bottom and self.top + self.bottom <= height
