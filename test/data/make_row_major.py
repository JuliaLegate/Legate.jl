#!/usr/bin/env python3
"""Regenerate test/data/row_major.h5 (row-major file for the Julia HDF5 tests).

Datasets hold arange(prod(shape)).reshape(shape); numpy C-order => row-major file.
Run: python3 test/data/make_row_major.py
"""
import os

import h5py
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "row_major.h5")

DATASETS = {
    "vec1d": (np.float64, (10,)),
    "mat2d": (np.float64, (4, 5)),
    "mat3d": (np.int64, (3, 4, 5)),
}

with h5py.File(OUT, "w") as f:
    for name, (dtype, shape) in DATASETS.items():
        arr = np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)
        assert arr.flags["C_CONTIGUOUS"]
        f.create_dataset(name, data=arr)

print("wrote", OUT)
