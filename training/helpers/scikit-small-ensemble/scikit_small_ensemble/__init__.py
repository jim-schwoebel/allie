from __future__ import absolute_import
from scikit_small_ensemble.scikit_ensemble import CompressedEstimators, DiskEstimators


def compress(model, ratio=0.5):
    if isinstance(model.estimators_, CompressedEstimators):
        raise Exception("The model is already compressed.")
    model.estimators_ = CompressedEstimators(model, ratio)


def memory_map(model, ratio=1.0):
    if isinstance(model.estimators_, DiskEstimators):
        raise Exception("The model is already memory mapped.")
    model.estimators_ = DiskEstimators(model, ratio)
