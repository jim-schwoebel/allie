# -*- coding: utf-8 -*-

import os
import tempfile
from collections import defaultdict
from datetime import datetime

import numpy as np
from sklearn.preprocessing import LabelEncoder


def encode_score(scorer, expected, observed):
    if expected.dtype == 'object':
        le = LabelEncoder()
        expected = le.fit_transform(expected)
        observed = le.transform(observed)

    return scorer(expected, observed)


def ensure_dir(directory):
    """Create diretory if it does not exist yet."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def make_dumpable(params, datetimes=False):
    """Get nested dicts of params to allow json dumping.

    Also work around this: https://github.com/HDI-Project/BTB/issues/79
    And convert numpy types to primitive types.

    Optionally dump datetimes to ISO format.

    Args:
        params (dict):
            Params dictionary with tuples as keys.
        datetimes (bool):
            whether to convert datetimes to ISO strings or not.

    Returns:
        dict:
            Dumpable params as a tree of dicts and nested sub-dicts.
    """
    nested_params = defaultdict(dict)
    for (block, param), value in params.items():
        if isinstance(value, np.integer):
            value = int(value)

        elif isinstance(value, np.floating):
            value = float(value)

        elif isinstance(value, np.ndarray):
            value = value.tolist()

        elif isinstance(value, np.bool_):
            value = bool(value)

        elif value == 'None':
            value = None

        elif datetimes and isinstance(value, datetime):
            value = value.isoformat()

        nested_params[block][param] = value

    return nested_params


def _walk(document, transform):
    if not isinstance(document, dict):
        return document

    new_doc = dict()
    for key, value in document.items():
        if isinstance(value, dict):
            value = _walk(value, transform)
        elif isinstance(value, list):
            value = [_walk(v, transform) for v in value]

        new_key, new_value = transform(key, value)
        new_doc[new_key] = new_value

    return new_doc


def remove_dots(document):
    """Replace dots with dashes in all the keys from the dictionary."""
    return _walk(document, lambda key, value: (key.replace('.', '-'), value))


def restore_dots(document):
    """Replace dashes with dots in all the keys from the dictionary."""
    return _walk(document, lambda key, value: (key.replace('-', '.'), value))


def make_keras_picklable():
    """Make the keras models picklable."""

    import keras.models   # noqa: lazy import slow dependencies

    def __getstate__(self):
        model_str = ""
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            keras.models.save_model(self, fd.name, overwrite=True)
            model_str = fd.read()

        return {'model_str': model_str}

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            fd.write(state['model_str'])
            fd.flush()
            try:
                model = keras.models.load_model(fd.name)

            except ValueError:
                from keras.applications import mobilenet
                from keras.utils.generic_utils import CustomObjectScope
                scope = {
                    'relu6': mobilenet.relu6,
                    'DepthwiseConv2D': mobilenet.DepthwiseConv2D
                }
                with CustomObjectScope(scope):
                    model = keras.models.load_model(fd.name)

        self.__dict__ = model.__dict__

    cls = keras.models.Model
    cls.__getstate__ = __getstate__
    cls.__setstate__ = __setstate__
