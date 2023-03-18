"""Configuration file."""
from os.path import realpath, dirname, expanduser, join

ROOT_PATH = dirname(dirname(realpath(__file__)))
WARNING_DENSE_SIZE = 1e9
DATASET_PATH = join(expanduser("~"), ".gli/datasets")
