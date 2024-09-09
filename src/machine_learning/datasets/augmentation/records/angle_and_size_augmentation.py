#!/usr/bin/python3

import collections
import os
import re


RecordFileType = collections.namedtuple('RecordFileDesc', ['suffix'])


FILE_NAME_FORBIDDEN_SYMBOLS = ''.join([
    '~', r'\)', r'\(', r'\\', r'\!', r'\*', r'\<', r'\>',
    r'\:', r'\;' r'\,', r'\?', r'\"', r'\*', r'\|', r'\/'])
FILE_NAME_RE_PATTERN = '[^' + FILE_NAME_FORBIDDEN_SYMBOLS + ']+'


class AngledResizedSrcInTargetDesc:
    """Descriptor of resized and rotated object augmented in target image."""

    AUGMENTED_FULL = RecordFileType('full')
    SRC_MASK_FULL = RecordFileType('full.mask')

    TARGET_TILE = RecordFileType('target')
    AUGMENTED_TILE = RecordFileType(suffix=None)
    AUGMENTED_SRC = RecordFileType('augsrc')
    TARGET_MASK = RecordFileType('mask')
    TARGET_SCALED_MASK = RecordFileType('scaled_mask')
    TARGET_SCALED_MASK_OF_MASK = RecordFileType('scaled_mask_mask')

    ALL_RECORD_FILES = {
        TARGET_TILE, AUGMENTED_TILE, AUGMENTED_SRC, TARGET_MASK, TARGET_SCALED_MASK, TARGET_SCALED_MASK_OF_MASK}

    def __init__(self, combined_augmented_file_prefix=''):
        self.combined_augmented_file_prefix = combined_augmented_file_prefix

        self.tile_top_left_row = 0
        self.tile_top_left_col = 0
        self.angle_in_degrees = 0
        self.scaled_width_in_pixels = 0

        self.sample_files = {}

    def __eq__(self, other):
        return (
            self.angle_in_degrees == other.angle_in_degrees and
            self.scaled_width_in_pixels == other.scaled_width_in_pixels and
            self.tile_top_left_row == other.tile_top_left_row and
            self.tile_top_left_col == other.tile_top_left_col and
            self.combined_augmented_file_prefix == other.combined_augmented_file_prefix)

    def __hash__(self):
        return hash((
            self.angle_in_degrees, self.scaled_width_in_pixels,
            self.tile_top_left_row, self.tile_top_left_col,
            self.combined_augmented_file_prefix))

    def __repr__(self):
        return self.sample_id

    @property
    def sample_id(self):
        return "{}.a{}.w{}.t{}.l{}".format(self.combined_augmented_file_prefix,
            self.angle_in_degrees, self.scaled_width_in_pixels,
            self.tile_top_left_row, self.tile_top_left_col)

    @classmethod
    def from_src_and_target_file(cls, src_img_file, target_image_file):
        return cls(combined_augmented_file_prefix=cls.combine_augmented_file_names(
            target_image_file, src_img_file))

    @staticmethod
    def combine_augmented_file_names(src_img_file, target_image_file):
        return f'{src_img_file.stem}~{target_image_file.stem}'

    def combine_record_file_name(self, file_desc, ext):
        if file_desc.suffix:
            return "{}.a{}.w{}.t{}.l{}.{}.{}".format(self.combined_augmented_file_prefix,
                self.angle_in_degrees, self.scaled_width_in_pixels,
                self.tile_top_left_row, self.tile_top_left_col, file_desc.suffix, ext)
        else:
            return "{}.a{}.w{}.t{}.l{}.{}".format(self.combined_augmented_file_prefix,
                self.angle_in_degrees, self.scaled_width_in_pixels,
                self.tile_top_left_row, self.tile_top_left_col, ext)

    @classmethod
    def parse_sample_file_path(cls, file_path):
        file_name = os.path.basename(file_path) if isinstance(file_path, str) else file_path.name
        name_match = AngledResizedSrcInTargetFileDesc.FILE_NAME_RE.match(file_name)
        if name_match:
            parsed_sample_desc = AngledResizedSrcInTargetDesc(
                combined_augmented_file_prefix=name_match.group('augmented_file_prefix'))
            parsed_sample_desc.tile_top_left_row = int(name_match.group('top'))
            parsed_sample_desc.tile_top_left_col = int(name_match.group('left'))
            parsed_sample_desc.angle_in_degrees = int(name_match.group('angle'))
            parsed_sample_desc.scaled_width_in_pixels = int(name_match.group('width'))

            parsed_file_desc = AngledResizedSrcInTargetFileDesc(
                file_path, combined_augmented_file_prefix=name_match.group('augmented_file_prefix'),
                target_file_desc=RecordFileType(name_match.group('suffix')))

            return parsed_sample_desc, parsed_file_desc
        else:
            return None, None

    def get_all_file_types(self):
        return set([sample_file for sample_file, _ in self.sample_files.items()])

    def add_file_desc(self, sample_file_desc):
        self.sample_files[sample_file_desc.target_file_desc] = sample_file_desc

    def create_saved_file_path(self, dir_path, file_desc, ext):
        return dir_path.joinpath(self.combine_record_file_name(file_desc, ext))


class AngledResizedSrcInTargetFileDesc:
    """Descriptor of one file which set represents resized and rotated object augmented in target image."""

    FILE_NAME_RE = re.compile('^' + ''.join([
        r'(?P<augmented_file_prefix>' + FILE_NAME_RE_PATTERN + '~' + FILE_NAME_RE_PATTERN, ')',
        r'\.a(?P<angle>\d+)', r'\.w(?P<width>\d+)', r'\.t(?P<top>\d+)', r'\.l(?P<left>\d+)',
        r'(\.(?P<suffix>\w+))?',
        r'\.(?P<ext>\w+)']) + '$')

    def __init__(self, file_path, combined_augmented_file_prefix='', target_file_desc=None):
        self.file_path = file_path
        self.target_file_desc = target_file_desc
        self.combined_augmented_file_prefix = combined_augmented_file_prefix

    def __eq__(self, other):
        return (
            self.combined_augmented_file_prefix == other.combined_augmented_file_prefix and
            self.target_file_desc == other.target_file_desc)

    def __hash__(self):
        return hash((
            self.combined_augmented_file_prefix, self.target_file_desc))

    def __repr__(self):
        return "{}-{}".format(self.combined_augmented_file_prefix, self.target_file_desc)

