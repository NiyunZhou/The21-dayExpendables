"""Fake boost v2:
    Merge submission csv files in a couple of minutes.
Usage:
    Prepare '0.csv' to 'N.csv' in the same directory as me,
    and modify GAP (with length of (N + 1) ) as wanted.
Output:
    '5. Done! Check out merge result in merge.csv' on the screen,
    and file 'merge.csv' in the same directory as me.
"""
import pandas as pd
import numpy as np
from scipy.sparse import csc_matrix
import sys
import os

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

###########################
# User-defined parameters #
###########################
GAP = np.array([0.79421, 0.78887, 0.78707, 0.78686, 0.78678])  # to be modified
SCORE = np.exp(GAP / (1 - GAP))
WEIGHT = SCORE / np.sum(SCORE)
FILE_NUM = len(SCORE)

HEADERS = ["VideoId", "LabelConfidencePairs"]
ROWS = 700640
COLS = 4810
TOP = 20

SLICE_ROWS = 43790
SLICE_LEN = SLICE_ROWS * 20
NUM_SLICES = ROWS / SLICE_ROWS

TYPE_INT = np.int16
TYPE_FLOAT = np.float16
TYPE_STR = '|S8'

###################################
# Pre-process and merge csv files #
###################################
print "\x1b[2K\r1. Pre-processing before reading csv files...",

label_merged = np.array([0] * TOP * ROWS, dtype=TYPE_INT)
confidence_merged = np.array([0] * TOP * ROWS, dtype=TYPE_FLOAT)
confidence_sparse = csc_matrix((ROWS, COLS), dtype=TYPE_FLOAT)

csv_file = None
rows = np.array([[row] * TOP for row in range(ROWS)]).flatten()
for i in range(FILE_NUM):
    print "\x1b[2K\r2. Reading {}.csv, {} of {} files...".format(i, i + 1, FILE_NUM),

    csv_file = pd.read_csv('{}.csv'.format(i))
    arr = np.array(csv_file.values[:, 1])
    mat = np.array([np.array(arr[row].split(" ")).reshape((TOP, 2)) for row in range(ROWS)]).reshape((ROWS * TOP, 2))
    label = mat[:, 0].astype(TYPE_INT)
    confidence = mat[:, 1].astype(TYPE_FLOAT) * WEIGHT[i]
    confidence_sparse += csc_matrix((confidence, (rows, label)), shape=(ROWS, COLS))

####################################
# Trim prediction to top-20 labels #
####################################
# Batch-process data in case of MemoryError. 700640 * 20 = 16 * 43790 * 20 = 16 * 875800
for i in range(NUM_SLICES):
    print "\x1b[2K\r3. Generating new prediction data: slice {} of {}...".format(i + 1, NUM_SLICES),

    label_merged[SLICE_LEN * i:SLICE_LEN * (i + 1)] = \
        np.flip(np.argsort(confidence_sparse[SLICE_ROWS * i:SLICE_ROWS * (i + 1)].toarray(), axis=-1), -1)[:, :TOP].flatten()
    confidence_merged[SLICE_LEN * i:SLICE_LEN * (i + 1)] = \
        np.flip(np.sort(confidence_sparse[SLICE_ROWS * i:SLICE_ROWS * (i + 1)].toarray(), axis=-1), -1)[:, :TOP].flatten()

##########################
# Save merged prediction #
##########################
print "\x1b[2K\r4. New prediction data is ready, generating csv...",

lc_list = np.vstack((label_merged.astype(TYPE_STR), confidence_merged.astype(TYPE_STR))).T.reshape(ROWS, TOP * 2)
video_id = list(csv_file.values[:, 0])
lc_pairs = [" ".join(lc_list[row]) for row in range(ROWS)]

saver = pd.DataFrame({HEADERS[0]: video_id, HEADERS[1]: lc_pairs}, columns=HEADERS)
saver.to_csv("merge.csv", index=False)

print "\x1b[2K\r5. Done! Check out merge result in merge.csv"
