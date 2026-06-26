"""Nearest-neighbour descriptor->preset retrieval (Task 5 baseline).

Index = TRAIN-split presets keyed by their 7-D AudioCommons descriptor.
For each test target descriptor vector, return the params of the train preset
whose descriptors are nearest (Euclidean in z-scored descriptor space).

The "do you beat lookup?" floor; the CVAE has to clear this to be worth
publishing as a baseline.
"""

from __future__ import annotations

import numpy as np

from baselines.common.io import TIMBRAL_KEYS


class DescriptorNNRetrieval:
    def __init__(self, Y_train: np.ndarray, params_train: list[dict],
                 standardise: bool = True):
        self.params_train = params_train
        if standardise:
            self.mean = np.nanmean(Y_train, axis=0)
            self.std = np.nanstd(Y_train, axis=0)
            self.std = np.where(self.std > 1e-6, self.std, 1.0)
        else:
            self.mean = np.zeros(Y_train.shape[1])
            self.std = np.ones(Y_train.shape[1])
        Y = np.nan_to_num(Y_train, nan=0.0)
        self.Y_std = (Y - self.mean) / self.std

    def _standardise(self, Y: np.ndarray) -> np.ndarray:
        Y = np.nan_to_num(Y, nan=0.0)
        return (Y - self.mean) / self.std

    def query(self, Y_target: np.ndarray) -> list[dict]:
        """For each target descriptor vector, return the matched train preset's params."""
        Q = self._standardise(Y_target)
        # ||q - y||^2 for all pairs; argmin per row.
        sq_q = (Q ** 2).sum(axis=1, keepdims=True)
        sq_y = (self.Y_std ** 2).sum(axis=1)[None, :]
        cross = Q @ self.Y_std.T
        dists = sq_q + sq_y - 2.0 * cross
        nn_idx = np.argmin(dists, axis=1)
        return [self.params_train[int(i)] for i in nn_idx]
