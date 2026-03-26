from __future__ import annotations

from typing import Any

import numpy as np


def project_array_to_2d(value: Any) -> np.ndarray | None:
    """Project an array-like value into a 2D matrix for heatmap display.

    Singleton axes are squeezed out first. For rank-3+ tensors, the last two
    axes become the interior cell shape and the leading axes are tiled as a
    grid of blocks. Examples:

    - [N] -> [1, N]
    - [H, W] -> [H, W]
    - [16, 16, 1] -> [16, 16]
    - [3, 3, 16, 16] -> [48, 48] as a 3x3 grid of 16x16 blocks
    """
    try:
        arr = np.asarray(value, dtype=np.float32)
    except Exception:
        return None

    squeezed_shape = tuple(dim for dim in arr.shape if dim != 1)

    if not squeezed_shape:
        return arr.reshape(1, 1)
    if squeezed_shape != arr.shape:
        arr = arr.reshape(squeezed_shape)

    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim == 2:
        return arr

    outer_shape = arr.shape[:-2]
    inner_rows = int(arr.shape[-2])
    inner_cols = int(arr.shape[-1])
    outer_split = max(1, (len(outer_shape) + 1) // 2)
    block_row_shape = outer_shape[:outer_split]
    block_col_shape = outer_shape[outer_split:]
    block_rows = int(np.prod(block_row_shape, dtype=np.int64)) if block_row_shape else 1
    block_cols = int(np.prod(block_col_shape, dtype=np.int64)) if block_col_shape else 1

    outer_ndim = len(outer_shape)
    perm = [
        *range(0, outer_split),
        outer_ndim,
        *range(outer_split, outer_ndim),
        outer_ndim + 1,
    ]
    arr = np.transpose(arr, axes=perm)
    return arr.reshape(block_rows * inner_rows, block_cols * inner_cols)


def downsample_matrix(arr: np.ndarray, max_size: int = 64) -> np.ndarray:
    rows, cols = arr.shape
    if rows > max_size:
        r_idx = np.linspace(0, rows - 1, max_size, dtype=int)
        arr = arr[r_idx]
    if cols > max_size:
        c_idx = np.linspace(0, cols - 1, max_size, dtype=int)
        arr = arr[:, c_idx]
    return arr.astype(np.float32, copy=False)


def to_heatmap_2d(value: Any, max_size: int = 64) -> list[list[float]] | None:
    arr = project_array_to_2d(value)
    if arr is None:
        return None
    return downsample_matrix(arr, max_size=max_size).tolist()


def flat_values_for_client(value: Any, max_elements: int = 65536) -> list[float] | None:
    try:
        arr = np.asarray(value, dtype=np.float32)
    except Exception:
        return None

    squeezed_shape = tuple(dim for dim in arr.shape if dim != 1)
    if squeezed_shape:
        arr = arr.reshape(squeezed_shape)
    else:
        arr = arr.reshape(1)

    if int(arr.size) > max_elements:
        return None
    return arr.reshape(-1).astype(np.float32, copy=False).tolist()