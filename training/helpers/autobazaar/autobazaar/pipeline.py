# -*- coding: utf-8 -*-

"""AutoBazaar Pipeline Module."""

import json
import logging
import os
import random
import uuid
from collections import Counter

import cloudpickle
import numpy as np
import pandas as pd
from mit_d3m.loaders import get_loader
from mit_d3m.metrics import METRICS_DICT
from mlblocks import MLPipeline

from autobazaar.utils import encode_score

LOGGER = logging.getLogger(__name__)


class ABPipeline(object):
    """AutoBazaar Pipeline Class."""

    def _extract_hyperparameters(self, preprocessing_primitives):
        block_names_count = Counter()
        block_names = list()
        for primitive in preprocessing_primitives:
            block_names_count.update([primitive])
            block_count = block_names_count[primitive]
            block_names.append('{}#{}'.format(primitive, block_count))

        pre_params = dict()
        hyperparameters = self.pipeline_dict['hyperparameters'].copy()
        for block_name in block_names:
            block_params = hyperparameters.pop(block_name, None)
            if block_params:
                pre_params[block_name] = block_params

        return pre_params, hyperparameters

    def __init__(self, pipeline_dict, loader, metric, problem_doc):
        self.pipeline_dict = pipeline_dict
        self.name = pipeline_dict['name']
        self.template = pipeline_dict.get('template')
        self.loader = loader
        self.metric = metric
        self.problem_doc = problem_doc

        preprocessing_blocks = self.pipeline_dict.get('preprocessing_blocks')
        if preprocessing_blocks:
            preprocessing = pipeline_dict.copy()
            preprocessing_primitives = preprocessing['primitives'][:preprocessing_blocks]
            preprocessing['primitives'] = preprocessing_primitives
            self._preprocessing = preprocessing

            tunable = pipeline_dict.copy()
            tunable_primitives = tunable['primitives'][preprocessing_blocks:]
            tunable['primitives'] = tunable_primitives
            self._tunable = tunable

            pre_params, tun_params = self._extract_hyperparameters(preprocessing_primitives)
            self._preprocessing['hyperparameters'] = pre_params
            self._tunable['hyperparameters'] = tun_params

        else:
            self._preprocessing = None
            self._tunable = pipeline_dict

        self.id = str(uuid.uuid4())
        self.cv_scores = list()

        self.rank = None
        self.score = None
        self.dumped = False
        self.fitted = False

        self.pipeline = MLPipeline.from_dict(pipeline_dict)

    def fit(self, data_params):
        """Fit the pipeline on the given params."""
        X, y = data_params.X, data_params.y

        self.pipeline = MLPipeline.from_dict(self.pipeline_dict)
        self.pipeline.fit(X, y, **data_params.context)

        self.fitted = True

    def predict(self, d3mds):
        """Get predictions for the given D3MDS."""
        data_params = self.loader.load(d3mds)

        predictions = self.pipeline.predict(data_params.X, **data_params.context)

        out_df = pd.DataFrame()
        out_df['d3mIndex'] = data_params.y.index
        out_df[d3mds.target_column] = predictions

        return out_df

    def _get_split(self, X, y, indexes):
        if hasattr(X, 'iloc'):
            X = X.iloc[indexes]
        else:
            X = X[indexes]

        if y is not None:
            if hasattr(y, 'iloc'):
                y = y.iloc[indexes]
            else:
                y = y[indexes]

        return X, y

    def _get_score(self):
        score = np.mean(self.cv_scores)
        std = np.std(self.cv_scores)

        if 'Error' in self.metric:
            rank = score

        elif score <= 1:
            rank = 1 - score

        else:
            raise ValueError("Found a score > 1 in a maximization problem: {}".format(score))

        return score, std, rank

    def preprocess(self, X, y, context):
        """Execute the preprocessing steps of the pipeline."""
        if self._preprocessing:
            LOGGER.info("Executing preprocessing pipeline")
            pipeline = MLPipeline.from_dict(self._preprocessing)
            pipeline.fit(X, y, **context)
            return pipeline.predict(X, **context)
        else:
            LOGGER.info("No preprocessing steps found")
            return X

    def cv_score(self, X, y, context, metric=None, cv=None):
        """Cross Validate this pipeline."""

        scorer = METRICS_DICT[metric or self.metric]

        LOGGER.debug('CV Scoring pipeline %s', self)

        self.cv_scores = list()

        for fold, (train_index, test_index) in enumerate(cv.split(X, y)):

            LOGGER.debug('Scoring fold: %s', fold)

            X_train, y_train = self._get_split(X, y, train_index)
            X_test, y_test = self._get_split(X, y, test_index)

            pipeline = MLPipeline.from_dict(self._tunable)
            pipeline.fit(X_train, y_train, **context)

            pred = pipeline.predict(X_test, **context)

            score = encode_score(scorer, y_test, pred)
            self.cv_scores.append(score)

            LOGGER.debug('Fold %s score: %s', fold, score)

        score, std, rank = self._get_score()

        LOGGER.debug('CV score: %s +/- %s; rank: %s', score, std, rank)

        self.score = score
        self.std = std
        self.rank = rank + random.random() * 1.e-12   # to avoid collisions

    def to_dict(self, problem_doc=False):
        """Return the details of this pipeline in a dict."""
        pipeline_dict = self.pipeline.to_dict().copy()
        pipeline_dict.update({
            'id': self.id,
            'name': self.name,
            'template': self.template,
            'loader': self.loader.to_dict(),
            'score': self.score,
            'rank': self.rank,
            'metric': self.metric
        })
        if problem_doc:
            pipeline_dict['problem_doc'] = self.problem_doc

        return pipeline_dict

    def __repr__(self):
        return 'ABPipeline({})'.format(json.dumps(self.to_dict(), indent=4))

    @classmethod
    def from_dict(cls, pipeline_dict):
        """Load a pipeline from a dict."""
        pipeline_dict = pipeline_dict.copy()
        loader = get_loader(**pipeline_dict.pop('loader'))
        metric = pipeline_dict['metric']
        problem_doc = pipeline_dict.pop('problem_doc')
        return cls(pipeline_dict, loader, metric, problem_doc)

    def dump(self, output_dir, rank=None):
        """Dump this pipeline using pickle."""
        if rank is None:
            rank = self.rank

        LOGGER.info('Dumping pipeline with rank %s: %s', rank, self.id)

        self.dumped = True

        pickle_path = os.path.join(output_dir, '{}.pkl'.format(self.id))
        with open(pickle_path, "wb") as pickle_file:
            LOGGER.info("Outputting pipeline %s", pickle_file.name)
            cloudpickle.dump(self, pickle_file)

        json_path = os.path.join(output_dir, '{}.json'.format(self.id))
        self.pipeline.save(json_path)
