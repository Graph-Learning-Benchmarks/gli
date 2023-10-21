"""Configuration file."""
from os.path import realpath, dirname, expanduser, join

ROOT_PATH = dirname(dirname(realpath(__file__)))
WARNING_DENSE_SIZE = 1e9
DATASET_PATH = join(expanduser("~"), ".gli/datasets")
GLOBAL_FILE_URL = "https://jiaqima.github.io/gli/global_urls.json"
SERVER_IP = "http://34.211.28.138"
