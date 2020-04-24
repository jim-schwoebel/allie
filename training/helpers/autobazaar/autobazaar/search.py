# -*- coding: utf-8 -*-

"""AutoBazaar Search Module.


This module contains the PipelineSearcher, which is the class that
contains the main logic of the Auto Machine Learning process.
"""

import gc
import itertools
import json
import logging
import os
import signal
import warnings
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from btb import HyperParameter
from btb.tuning import GP, GPEi, Uniform
from mit_d3m.loaders import get_loader
from mlblocks.mlpipeline import MLPipeline
from sklearn.model_selection import KFold, StratifiedKFold

from autobazaar.pipeline import ABPipeline
from autobazaar.utils import ensure_dir, make_dumpable, remove_dots, restore_dots

LOGGER = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=DeprecationWarning)

TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), 'templates')

TRIVIAL_PIPELINE_METHOD = {
    'classification': 'mode',
    'regression': 'median',
    'collaborativeFiltering': 'median',
    'graphMatching': 'mode',
}

TUNERS = {
    'gp': GP,
    'gpei': GPEi,
    'uniform': Uniform
}

PARAM_TYPES = {
    'str': 'string',
}


class StopSearch(KeyboardInterrupt):
    pass


class UnsupportedProblem(Exception):
    pass


def log_times(name, append=False):
    def decorator(wrapped):
        def wrapper(self, *args, **kwargs):
            start = datetime.utcnow()
            result = wrapped(self, *args, **kwargs)
            elapsed = (datetime.utcnow() - start).total_seconds()

            if append:
                attribute = getattr(self, name, None)
                if attribute is None:
                    attribute = list()
                    setattr(self, name, attribute)

                attribute.append(elapsed)

            else:
                setattr(self, name, elapsed)

            return result

        return wrapper

    return decorator


class PipelineSearcher(object):
    """PipelineSearcher class.

    This class is responsible for searching the best pipeline to solve a
    given dataset and problem.
    """

    def __init__(self, pipelines_dir, db=None, test_id=None, tuner_type='gp',
                 cv_splits=5, random_state=0):

        self._db = db

        self._pipelines_dir = pipelines_dir
        ensure_dir(self._pipelines_dir)

        self._cv_splits = cv_splits
        self._random_state = random_state

        self._tuner_type = tuner_type
        self._tuner_class = TUNERS[tuner_type]

        self._test_id = test_id

    def _dump_pipelines(self):
        LOGGER.info('Dumping best pipelines')
        dumped = list()
        gc.collect()
        for details in self._to_dump:
            pipeline = details['pipeline']
            if not pipeline.dumped:
                pipeline.fit(self.data_params)

                mlpipeline = pipeline.pipeline

                LOGGER.info("Dumping pipeline %s: %s", pipeline.id, pipeline.pipeline)
                LOGGER.info("Hyperparameters: %s", mlpipeline.get_hyperparameters())

                pipeline.dump(self._pipelines_dir)

                details['pipeline'] = pipeline.id
                dumped.append(details)
                gc.collect()

            else:
                LOGGER.info("Skipping already dumped pipeline %s", pipeline.id)

        return dumped

    def _set_for_dump(self, pipeline):
        self._to_dump.append({
            'elapsed': (datetime.utcnow() - self.start_time).total_seconds(),
            'iterations': len(self.pipelines) - 1,
            'cv_score': self.best_pipeline.score,
            'rank': self.best_pipeline.rank,
            'pipeline': pipeline,
            'load_time': self.load_time,
            'trivial_time': self.trivial_time,
            'cv_time': np.sum(self.cv_times),
        })

    def _save_pipeline(self, pipeline):
        pipeline_dict = pipeline.to_dict(True)
        pipeline_dict['_id'] = pipeline.id
        pipeline_dict['ts'] = datetime.utcnow()

        self.pipelines.append(pipeline_dict)

        if self._db:
            insertable = remove_dots(pipeline_dict)
            insertable.pop('problem_doc')
            insertable['dataset'] = self.dataset_id
            insertable['tuner_type'] = self._tuner_type
            insertable['test_id'] = self._test_id
            self._db.pipelines.insert_one(insertable)

    @log_times('trivial_time')
    def _build_trivial_pipeline(self):
        LOGGER.info("Building the Trivial pipeline")
        try:
            method = TRIVIAL_PIPELINE_METHOD.get(self.task_type)
            pipeline_dict = {
                'name': 'trivial.{}'.format(method),
                'primitives': ['mlprimitives.custom.trivial.TrivialPredictor'],
                'init_params': {
                    'mlprimitives.custom.trivial.TrivialPredictor': {
                        'method': method
                    }
                }
            }
            pipeline = ABPipeline(pipeline_dict, self.loader, self.metric, self.problem_doc)
            pipeline.cv_score(self.data_params.X, self.data_params.y,
                              self.data_params.context, cv=self.kf)

            self._save_pipeline(pipeline)

            return pipeline

        except Exception:
            # if the Trivial pipeline crashes we can do nothing,
            # so we just log the error and move on.
            LOGGER.exception("The Trivial pipeline crashed.")

    def _load_template_json(self, template_name):
        if template_name.endswith('.json'):
            template_filename = template_name
            name = template_name[:-5]
        else:
            name = template_name
            template_name = template_name.replace('/', '.') + '.json'
            template_filename = os.path.join(TEMPLATES_DIR, template_name)

        if os.path.exists(template_filename):
            with open(template_filename, 'r') as template_file:
                template_dict = json.load(template_file)
                template_dict['name'] = name

            return template_dict

    def _find_template(self, template_name):
        match = {
            'metadata.name': template_name
        }
        cursor = self._db.pipelines.find(match)
        templates = list(cursor.sort('metadata.insert_ts', -1).limit(1))
        if templates:
            template = templates[0]
            template['name'] = template.pop('metadata')['name']
            template['template'] = str(template.pop('_id'))
            return restore_dots(template)

    def _load_template(self, template_name):
        if self._db:
            template = self._find_template(template_name)
            if template:
                return template

        return self._load_template_json(template_name)

    def _get_template(self, template_name=None):
        if template_name:
            template = self._load_template(template_name)
            if not template:
                raise ValueError("Template {} not found".format(template_name))

            primitives = '\n'.join(template['primitives'])
            LOGGER.info('Using template %s:\n%s', template_name, primitives)
            return template

        else:
            problem_type = [
                self.data_modality,
                self.task_type,
                self.task_subtype
            ]

            for levels in reversed(range(1, 4)):
                # Try the following options:
                # modality/task/subtask/default
                # modality/task/default
                # modality/default
                template_name = '/'.join(problem_type[:levels] + ['default'])
                template = self._load_template(template_name)
                if template:
                    primitives = '\n'.join(template['primitives'])
                    LOGGER.info('Using template %s:\n%s', template_name, primitives)
                    return template

            # Nothing has been found for this modality/task/subtask combination
            problem_type = '/'.join(problem_type)
            LOGGER.error('Problem type not supported %s', problem_type)
            raise UnsupportedProblem(problem_type)

    @log_times('cv_times', append=True)
    def _cv_pipeline(self, params=None):
        pipeline_dict = self.template_dict.copy()
        if params:
            pipeline_dict['hyperparameters'] = params

        pipeline = ABPipeline(pipeline_dict, self.loader, self.metric, self.problem_doc)

        X = self.data_params.X
        y = self.data_params.y
        context = self.data_params.context

        try:
            pipeline.cv_score(X, y, context, cv=self.kf)
        except KeyboardInterrupt:
            raise
        except Exception:
            LOGGER.exception("Crash cross validating pipeline %s", pipeline.id)
            return None

        return pipeline

    def _create_tuner(self, pipeline):
        # Build an MLPipeline to get the tunables and the default params
        mlpipeline = MLPipeline.from_dict(self.template_dict)
        tunable_hyperparameters = mlpipeline.get_tunable_hyperparameters()

        tunables = []
        tunable_keys = []
        for block_name, params in tunable_hyperparameters.items():
            for param_name, param_details in params.items():
                key = (block_name, param_name)
                param_type = param_details['type']
                param_type = PARAM_TYPES.get(param_type, param_type)
                if param_type == 'bool':
                    param_range = [True, False]
                else:
                    param_range = param_details.get('range') or param_details.get('values')

                value = HyperParameter(param_type, param_range)
                tunables.append((key, value))
                tunable_keys.append(key)

        # Create the tuner
        LOGGER.info('Creating %s tuner', self._tuner_class.__name__)

        self.tuner = self._tuner_class(tunables)

        if pipeline:
            try:
                # Add the default params and the score obtained by them to the tuner.
                default_params = defaultdict(dict)
                for block_name, params in pipeline.pipeline.get_hyperparameters().items():
                    for param, value in params.items():
                        key = (block_name, param)
                        if key in tunable_keys:
                            if value is None:
                                raise ValueError('None value is not supported')

                            default_params[key] = value

                if pipeline.rank is not None:
                    self.tuner.add(default_params, 1 - pipeline.rank)

            except ValueError:
                pass

    def _set_checkpoint(self):
        next_checkpoint = self.checkpoints.pop(0)
        interval = next_checkpoint - self.current_checkpoint
        self._stop_time = datetime.utcnow() + timedelta(seconds=interval)
        LOGGER.info("Setting %s seconds checkpoint in %s seconds: %s",
                    next_checkpoint, interval, self._stop_time)
        signal.alarm(interval)
        self.current_checkpoint = next_checkpoint

    def _checkpoint(self, signum=None, frame=None, final=False):
        signal.alarm(0)

        checkpoint_name = 'Final' if final else str(self.current_checkpoint) + ' seconds'

        LOGGER.info("%s checkpoint reached", checkpoint_name)

        try:
            if self.best_pipeline:
                self._set_for_dump(self.best_pipeline)

        except KeyboardInterrupt:
            raise
        except Exception:
            LOGGER.exception("Checkpoint dump crashed")

        if final or not bool(self.checkpoints):
            self.current_checkpoint = None
            # LOGGER.warn("Stopping Search")
            # raise StopSearch()
        else:
            self._set_checkpoint()

    def _check_stop(self):
        if self._stop_time and self._stop_time < datetime.utcnow():
            LOGGER.warn("Stop Time already passed. Stopping Search!")
            raise StopSearch()

    def _setup_search(self, d3mds, budget, checkpoints, template_name):
        self.start_time = datetime.utcnow()
        self.cv_times = list()

        # Problem variables
        self.problem_id = d3mds.get_problem_id()

        self.task_type = d3mds.get_task_type()
        self.task_subtype = d3mds.problem.get_task_subtype()

        # TODO: put this in mit-d3m loaders
        if self.task_type == 'vertex_classification':
            self.task_type = 'vertex_nomination'

        self.problem_doc = d3mds.problem_doc

        # Dataset variables
        self.dataset_id = d3mds.dataset_id
        self.data_modality = d3mds.get_data_modality()

        # TODO: put this in mit-d3m loaders
        if self.data_modality == 'edgeList':
            self.data_modality = 'graph'

        self.metric = d3mds.get_metric()

        self.loader = get_loader(self.data_modality, self.task_type)

        self.best_pipeline = None

        self.pipelines = []
        self.checkpoints = sorted(checkpoints or [])
        self.current_checkpoint = 0

        self._to_dump = []

        if not self.checkpoints and budget is None:
            self.budget = 1
        else:
            self.budget = budget

        self.template_dict = self._get_template(template_name)

        LOGGER.info("Running TA2 Search")
        LOGGER.info("Problem Id: %s", self.problem_id)
        LOGGER.info("    Data Modality: %s", self.data_modality)
        LOGGER.info("    Task type: %s", self.task_type)
        LOGGER.info("    Task subtype: %s", self.task_subtype)
        LOGGER.info("    Metric: %s", self.metric)
        LOGGER.info("    Checkpoints: %s", self.checkpoints)
        LOGGER.info("    Budget: %s", self.budget)

    @log_times('load_time')
    def _load_data(self, d3mds):
        self.data_params = self.loader.load(d3mds)

    def _setup_cv(self):
        if isinstance(self.data_params.y, pd.Series):
            min_samples = self.data_params.y.value_counts().min()
        else:
            y = self.data_params.y
            min_samples = y.groupby(list(y.columns)).size().min()

        if self.task_type == 'classification' and min_samples >= self._cv_splits:
            self.kf = StratifiedKFold(
                n_splits=self._cv_splits,
                shuffle=True,
                random_state=self._random_state
            )
        else:
            self.kf = KFold(
                n_splits=self._cv_splits,
                shuffle=True,
                random_state=self._random_state
            )

    def search(self, d3mds, template_name=None, budget=None, checkpoints=None):
        try:
            self._setup_search(d3mds, budget, checkpoints, template_name)
            self._load_data(d3mds)
            self._setup_cv()

            # Build the trivial pipeline
            self.best_pipeline = self._build_trivial_pipeline()

            # Do not continue if there is no budget or no fit data
            if budget == 0 or not len(self.data_params.X):
                raise StopSearch()

            # Build the default pipeline
            default_pipeline = self._cv_pipeline()
            if default_pipeline:
                self.best_pipeline = default_pipeline
                self._save_pipeline(default_pipeline)

            if budget == 1:
                raise StopSearch()
            elif budget is not None:
                iterator = range(budget - 1)
            else:
                iterator = itertools.count()   # infinite range

            # Build the tuner
            self._create_tuner(default_pipeline)

            LOGGER.info("Starting the tuning loop")

            if self.checkpoints:
                signal.signal(signal.SIGALRM, self._checkpoint)
                self._set_checkpoint()
            else:
                self._stop_time = None

            for iteration in iterator:
                self._check_stop()
                proposed_params = self.tuner.propose()
                params = make_dumpable(proposed_params)

                LOGGER.info("Cross validating pipeline %s", iteration + 1)
                pipeline = self._cv_pipeline(params)

                if pipeline and (pipeline.rank is not None):
                    self.tuner.add(proposed_params, 1 - pipeline.rank)

                    LOGGER.info("Saving pipeline %s: %s", iteration + 1, pipeline.id)
                    self._save_pipeline(pipeline)

                    if not self.best_pipeline or (pipeline.rank < self.best_pipeline.rank):
                        self.best_pipeline = pipeline
                        LOGGER.info('Best pipeline so far: %s; rank: %s, score: %s',
                                    self.best_pipeline, self.best_pipeline.rank,
                                    self.best_pipeline.score)

                else:
                    self.tuner.add(proposed_params, -1000000)

        except KeyboardInterrupt:
            pass

        finally:
            signal.alarm(0)

        if self.current_checkpoint:
            self._checkpoint(final=True)
        elif self.best_pipeline and not checkpoints:
            self._set_for_dump(self.best_pipeline)

        if self.best_pipeline:
            LOGGER.info('Best pipeline for problem %s found: %s; rank: %s, score: %s',
                        self.problem_id, self.best_pipeline,
                        self.best_pipeline.rank, self.best_pipeline.score)

        else:
            LOGGER.info('No pipeline could be found for problem %s', self.problem_id)

        return self._dump_pipelines()
