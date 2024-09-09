#!/usr/bin/python3

import os, sys
SCRIPT_DIRS = os.path.dirname(os.path.abspath(__file__)).split(os.sep)
sys.path.append(os.path.join(os.sep, *SCRIPT_DIRS[:SCRIPT_DIRS.index('src')]))

from src.machine_learning.datasets.augmentation.records.angle_and_size_augmentation import AngledResizedSrcInTargetDesc
from src.machine_learning.datasets.augmentation.records.angle_and_size_augmentation import AngledResizedSrcInTargetFileDesc

import argparse
import glob
import logging
import pathlib


def parse_args(argv):
    # Parse input arguments.
    parser = argparse.ArgumentParser(description='Fit Sources into Targets')
    parser.add_argument('-d', '--dataset_dir', dest='root_dataset_dir',
                        help='Root of all the dataset files.', required=True)

    return parser.parse_args(argv[1:])


def main(argv):
    parsed_args = parse_args(argv)

    root_dataset_dir = parsed_args.root_dataset_dir

    aug_samples = {}

    for globPath in glob.iglob(os.path.join(root_dataset_dir, '**/**'), recursive=True):
        checked_path = pathlib.Path(globPath)
        if checked_path.is_file():
            aug_sample_desc, aug_sample_file_desc = AngledResizedSrcInTargetDesc.parse_sample_file_path(checked_path)
            if aug_sample_desc and aug_sample_desc.sample_id not in aug_samples:
                aug_samples[aug_sample_desc.sample_id] = aug_sample_desc
            if aug_sample_desc and aug_sample_desc not in aug_samples:
                aug_samples[aug_sample_desc.sample_id].add_file_desc(aug_sample_file_desc)

    # TODO (odruzh): validate samples
    for sample_key, sample_desc in aug_samples.items():
        if set(sample_desc.get_all_file_types()) != set(AngledResizedSrcInTargetDesc.ALL_RECORD_FILES):
            logging.error("%s has missing sample files.", sample_key)

    logging.info("%d total samples.", len(aug_samples))


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    sys.exit(main(sys.argv))
