"""Load system environment variables.

Variables of list type should be set as a string of comma-separated values, such as:
    DEFAULT_TRAIN_RATIOS = "0.1, 0.2, 0.3, 0.4, 0.5"
"""
import os

# Default data storage directory
DEFAULT_DATA_DIR = os.environ.get("DEFAULT_DATA_DIR", "./data")
# Default experiment storage directory
DEFAULT_EXP_DIR = os.environ.get("DEFAULT_EXP_DIR", "./experiments")
# Default name of the info file under each data directory.
DEFAULT_METADATA_FILE = os.environ.get("DEFAULT_METADATA_FILE", "metadata.json")

## Default values for creating experiments from extxyz data.

# Default ratio of test set size to total dataset size (train + val + test).
DEFAULT_TEST_RATIO = os.environ.get("DEFAULT_TEST_RATIO", 0.2)
# Default ratios of training set size to train + val dataset size.
DEFAULT_TRAIN_RATIOS = [
    float(r) for r in os.environ.get(
    "DEFAULT_TRAIN_RATIOS", "0.1, 0.2, 0.4, 0.6, 0.8, 1.0"
).split(",")
]
# Default maximum number of training data points. If maximum training set exceeds this,
#  the training set ratios will be scaled down proportionally.
DEFAULT_MAX_N_TRAIN = os.environ.get("DEFAULT_MAX_N_TRAIN", 3000)
# Default maximum number of test data points. If maximum test set exceeds this,
#  the test set ratio will be scaled down proportionally.
#  Cross test sets will also not exceed this size.
DEFAULT_MAX_N_TEST = os.environ.get("DEFAULT_MAX_N_TEST", 1000)