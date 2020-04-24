#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""AutoBazaar Command Line Module."""

import argparse
import gc
import json
import os
import shutil
import socket
import sys
import traceback
import warnings
from datetime import datetime

import cloudpickle
import pandas as pd
from mit_d3m import metrics
from mit_d3m.dataset import D3MDS
from mit_d3m.db import get_db
from mit_d3m.stats import get_stats
from mit_d3m.utils import logging_setup, make_abs

import autobazaar
from autobazaar.search import TUNERS, PipelineSearcher
from autobazaar.utils import encode_score, make_keras_picklable

warnings.filterwarnings(action='ignore')


VERSION = autobazaar.get_version()


def _load_targets(datasets_dir, dataset, problem):
    score_phase = 'SCORE'
    if problem:
        score_phase += '_' + problem

    score_dir = os.path.join(datasets_dir, dataset, score_phase)
    csv_path = os.path.join(score_dir, 'targets.csv')
    if not os.path.exists(csv_path):
        csv_path = os.path.join(score_dir, 'dataset_SCORE', 'tables', 'learningData.csv')

    return pd.read_csv(csv_path, index_col='d3mIndex')


def _get_metric(problem_path):
    problem_schema = os.path.join(problem_path, 'problemDoc.json')

    with open(problem_schema, 'r') as f:
        problem_doc = json.load(f)

    problem_metrics = problem_doc['inputs']['performanceMetrics']
    if len(problem_metrics) > 1:
        raise Exception("Wrong number of metrics")

    return metrics.METRICS_DICT[problem_metrics[0]['metric']]


def _get_dataset_paths(datasets_dir, dataset, phase, problem):
    if problem:
        full_phase = phase + '_' + problem
    else:
        full_phase = phase

    root_dir = os.path.join(datasets_dir, dataset, full_phase)
    dataset_path = os.path.join(root_dir, 'dataset_' + phase)
    problem_path = os.path.join(root_dir, 'problem_' + phase)

    return dataset_path, problem_path


def _search_pipeline(dataset, problem, template, input_dir, output_dir,
                     budget, checkpoints, splits, db, tuner_type, test_id):

    dataset_path, problem_path = _get_dataset_paths(input_dir, dataset, 'TRAIN', problem)

    d3mds = D3MDS(dataset_path, problem_path)

    searcher = PipelineSearcher(
        output_dir,
        cv_splits=splits,
        db=db,
        tuner_type=tuner_type,
        test_id=test_id
    )

    return searcher.search(d3mds, template, budget=budget, checkpoints=checkpoints)


def _test_pipeline(dataset, problem, pipeline_id, input_dir, output_dir):

    dataset_path, problem_path = _get_dataset_paths(input_dir, dataset, 'TEST', problem)

    pipeline_path = os.path.join(output_dir, '{}.pkl'.format(pipeline_id))
    with open(pipeline_path, 'rb') as pipeline_pkl:
        pipeline = cloudpickle.load(pipeline_pkl)

    print('Executing best pipeline {}'.format(pipeline))

    d3mds = D3MDS(dataset_path, problem_path)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        predictions = pipeline.predict(d3mds)

    return predictions


def _score_predictions(dataset, problem, predictions, input_dir):

    dataset_path, problem_path = _get_dataset_paths(input_dir, dataset, 'TEST', problem)
    metric = _get_metric(problem_path)

    predictions = predictions.set_index('d3mIndex')
    targets = _load_targets(input_dir, dataset, problem)[predictions.columns]

    if len(targets.columns) > 1 or len(predictions.columns) > 1:
        raise Exception("I don't know how to handle these")

    if any(targets.index != predictions.index):
        raise Exception("Different indexes cannot be compared")

    targets = targets.iloc[:, 0]
    predictions = predictions.iloc[:, 0]
    score = encode_score(metric, targets, predictions)
    print("Score: {}".format(score))

    summary = {'predictions': predictions, 'targets': targets}
    print(pd.DataFrame(summary).describe())

    return score


def _format_exception(e):
    error = '{}'.format(e.__class__.__name__)
    str_e = str(e)
    if str_e:
        error += ' - ' + str_e

    return error


def _box_print(message):
    length = len(message) + 10
    print(length * '#')
    print('#### {} ####'.format(message))
    print(length * '#')


def _insert_test(args, dataset):
    insert_ts = datetime.utcnow()
    document = {
        'test_id': args.test_id,
        'dataset': dataset,
        'timeout': args.timeout,
        'checkpoints': args.checkpoints,
        'budget': args.budget,
        'template': args.template,
        'status': 'running',
        'insert_ts': insert_ts,
        'update_ts': insert_ts,
        'version': VERSION,
        'hostname': socket.gethostname(),
        'tuner_type': args.tuner_type,
        'splits': args.splits,
    }
    args.db.tests.insert_one(document)


def _update_test(args, dataset, error, step):
    query = {
        'test_id': args.test_id,
        'dataset': dataset
    }
    update = {
        '$set': {
            'status': 'error' if error else 'done',
            'error': error,
            'step': step,
            'update_ts': datetime.utcnow()
        }
    }
    args.db.tests.update_one(query, update)


def _insert_test_result(args, result):
    document = result.copy()
    document['test_id'] = args.test_id
    document['insert_ts'] = datetime.utcnow()
    args.db.test_results.insert_one(document)


def _score_dataset(dataset, args):
    start_ts = datetime.utcnow()
    if args.db:
        _insert_test(args, dataset)

    result_base = {
        'dataset': dataset,
        'score': None,
        'elapsed': None,
        'iterations': None,
        'error': None,
        'step': None,
        'load_time': None,
        'trivial_time': None,
        'cv_time': None,
        'cv_score': None,
        'rank': None
    }
    results = []
    step = None
    error = None
    try:
        step = 'SEARCH'
        _box_print('Searching {}'.format(dataset))

        # cleanup
        if not args.keep:
            shutil.rmtree(args.output, ignore_errors=True)

        search_results = _search_pipeline(
            dataset, args.problem, args.template, args.input, args.output, args.budget,
            args.checkpoints, args.splits, args.db, args.tuner_type, args.test_id
        )
        gc.collect()

        for search_result in search_results or []:
            result = result_base.copy()
            result.update(search_result)
            results.append(result)

            pipeline = result['pipeline']
            try:
                step = 'TEST'
                _box_print('Executing {}'.format(dataset))
                predictions = _test_pipeline(dataset, args.problem, pipeline,
                                             args.input, args.output)

                step = 'SCORE'
                _box_print('Scoring {}'.format(dataset))
                score = _score_predictions(dataset, args.problem,
                                           predictions, args.input)
                result['score'] = score
                gc.collect()

            except Exception as e:
                error = _format_exception(e)
                print("Scoring pipeline {} for dataset {} failed on step {} with error {}"
                      .format(pipeline, dataset, step, error))
                traceback.print_exc()
                result['error'] = error
                result['step'] = step

            if args.db:
                _insert_test_result(args, result)

    except Exception as e:
        error = _format_exception(e)
        print("Dataset {} failed on step {} with error {}".format(dataset, step, error))
        traceback.print_exc()

        result_base['step'] = step
        result_base['error'] = error
        result_base['elapsed'] = (datetime.utcnow() - start_ts).total_seconds()
        results.append(result_base)

    if args.db:
        _update_test(args, dataset, error, step)

    return results


def _prepare_search(args):
    make_keras_picklable()

    if not args.datasets and not args.all:
        print('Please provide at least one dataset name or add the --all option')
        sys.exit(1)

    args.datasets = _get_datasets(args)

    if args.db:
        args.db = get_db(
            config=args.db_config,
            database=args.db_name,
            host=args.db_host,
            port=args.db_port,
            user=args.db_user,
            password=args.db_password
        )

    if args.checkpoints:
        args.checkpoints = [int(c) for c in args.checkpoints.split(',')]
    elif args.timeout:
        args.checkpoints = [args.timeout]

    if args.test_id is None:
        args.test_id = datetime.utcnow().strftime('%Y%m%d%H%M%S%f')


def _score_datasets(args):
    if args.report and os.path.exists(args.report):
        report = pd.read_csv(args.report)

    else:
        report = pd.DataFrame(columns=['dataset'])

    for dataset, row in args.datasets.iterrows():
        dataset_score = report[report.dataset == dataset]
        if dataset_score.empty or dataset_score.score.isnull().values[0]:

            if not dataset_score.empty:
                # clean-up
                report = report[report.dataset != dataset].copy()

            scores = _score_dataset(dataset, args)
            if scores:
                scores = pd.DataFrame(scores)
                scores = scores.merge(pd.DataFrame([row]), left_on='dataset', right_index=True)
                report = report.append(scores, ignore_index=True, sort=False)
                report = report.reindex(REPORT_COLUMNS, axis=1)

            if args.report:
                report.to_csv(args.report, index=False)

    return report


def _search(args):
    _prepare_search(args)

    print("{} - Processing Datasets: {}".format(args.test_id, args.datasets.index.values))
    report = _score_datasets(args)

    report = report.reindex(REPORT_COLUMNS, axis=1)
    columns = REPORT_COLUMNS[1:]
    print(report.set_index('dataset').to_string(columns=columns))


def _get_datasets(args):
    if args.all:
        datasets = [
            d for d in os.listdir(args.input)
            if os.path.isdir(os.path.join(args.input, d))
        ]
    else:
        datasets = args.datasets

    exclude = getattr(args, 'exclude', None) or []
    datasets = [dataset for dataset in datasets if dataset not in exclude]

    try:
        summary = get_stats(datasets, args.input)
    except KeyError:
        print("No matching datasets found")
        sys.exit(1)

    summary = summary.set_index('dataset').reindex(datasets)
    summary = summary[~summary.data_modality.isnull()]

    for field in ['data_modality', 'task_type', 'task_subtype']:
        value = getattr(args, field)
        if value:
            summary = summary[summary[field] == value]

    if summary.empty:
        print("No matching datasets found")
        sys.exit(1)

    return summary


REPORT_COLUMNS = [
    'dataset',
    'pipeline',
    'score',
    'rank',
    'cv_score',
    'metric',
    'data_modality',
    'task_type',
    'task_subtype',
    'elapsed',
    'iterations',
    'load_time',
    'trivial_time',
    'cv_time',
    'error',
    'step'
]


def _list(args):
    args.all = True
    datasets = _get_datasets(args)
    datasets = datasets.reset_index().sort_values('dataset').set_index('dataset')
    columns = [
        'data_modality', 'task_type', 'task_subtype', 'metric', 'size_human', 'train_samples'
    ]
    datasets = datasets.reindex(columns, axis=1)
    if args.report:
        print("Storing datasets as {}".format(args.report))
        datasets[columns].to_csv(args.report, index=True)
    else:
        print(datasets.to_string(columns=columns, index=True))


class ArgumentParser(argparse.ArgumentParser):
    def convert_arg_line_to_args(self, arg_line):
        return arg_line.split()


def _path_type(string):
    try:
        return make_abs(string)
    except ValueError:
        error = "Not a valid path: '{0}'.".format(string)
        raise argparse.ArgumentTypeError(error)


def _get_parser():

    # Logging
    logging_args = ArgumentParser(add_help=False)
    logging_args.add_argument('-v', '--verbose', action='count', default=0)
    logging_args.add_argument('-l', '--logfile')

    # Report
    report_args = ArgumentParser(add_help=False)
    report_args.add_argument('-r', '--report', type=_path_type,
                             help='Store results in the given CVS file.')

    # Dataset Selection
    dataset_args = ArgumentParser(add_help=False)
    dataset_args.add_argument('-i', '--input', default='input', type=_path_type,
                              help='Input datasets folder. Defaults to `input`.')
    dataset_args.add_argument('-o', '--output', type=_path_type,
                              help='Output pipelines folder. Defaults to `output`.',
                              default='output')
    dataset_args.add_argument('-p', '--problem', default='',
                              help='Problem suffix. Only needed if the dataset has more than one.')
    dataset_args.add_argument('-M', '--data-modality', type=str,
                              help='Only process datasets of the given Data Modality.')
    dataset_args.add_argument('-T', '--task-type', type=str,
                              help='Only process datasets of the given Task type')
    dataset_args.add_argument('-S', '--task-subtype', type=str,
                              help='Only process datasets of the given Task Subtype')

    # Search Configuration
    search_args = ArgumentParser(add_help=False)
    search_args.add_argument('-b', '--budget', type=int,
                             help='If given, maximum number tuning iterations to perform.')
    search_args.add_argument('-s', '--splits', type=int, default=5,
                             help='Number of Cross Validation Folds. Defaults to 5')
    search_args.add_argument('-c', '--checkpoints',
                             help=('Comma separated list of time checkpoints in seconds where '
                                   'the best pipeline so far will be dumped and stored.'))
    search_args.add_argument('-t', '--timeout', type=int,
                             help='Timeout in seconds. Ignored if checkpoints are given.')
    search_args.add_argument('-u', '--tuner-type', default='gp', choices=TUNERS.keys(),
                             help='Type of tuner to use. Defaults to "gp"')
    search_args.add_argument('--template',
                             help='Template to use. If not given, use the most appropriate one.')
    search_args.add_argument('-e', '--exclude', nargs='+',
                             help='Exclude these datasets. Useful in combination with --all.')
    search_args.add_argument('-a', '--all', action='store_true',
                             help='Process all the datasets found in the input folder.')
    search_args.add_argument('-k', '--keep', action='store_true',
                             help='Keep previous results in the output folder.')
    search_args.add_argument('--test-id', help='test_id associated with this run.')

    search_args.add_argument('datasets', nargs='*',
                             help='Datasets to process. Ignored if --all use used.')

    # Backend configuration
    db_args = ArgumentParser(add_help=False)
    db_args.add_argument('--db', action='store_true',
                         help='Use a MongoDB backend to store the results.')
    db_args.add_argument('--db-config', help='MongoDB configuraiton JSON file.')
    db_args.add_argument('--db-host', default='localhost')
    db_args.add_argument('--db-port', default=27017, type=int)
    db_args.add_argument('--db-name', default='autobazaar')
    db_args.add_argument('--db-user')
    db_args.add_argument('--db-password')

    parser = ArgumentParser(
        prog='autobazaar',
        description='AutoBazaar Experiments Suite',
        fromfile_prefix_chars='@',
        parents=[logging_args]
    )

    parser.add_argument('--version', action='version',
                        version='%(prog)s {version}'.format(version=VERSION))

    subparsers = parser.add_subparsers(title='command', help='Command to execute')
    parser.set_defaults(command=None)

    list_ = subparsers.add_parser('list', parents=[logging_args, dataset_args, report_args],
                                  help='List the available datasets that match the conditions.')
    list_.set_defaults(command=_list)

    search_parents = [
        logging_args,
        dataset_args,
        search_args,
        report_args,
        db_args
    ]
    search_ = subparsers.add_parser('search', parents=search_parents,
                                    help='Search the best pipeline for the given datasets.')
    search_.set_defaults(command=_search)

    return parser


def main():
    parser = _get_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        parser.exit()

    logging_setup(args.verbose, args.logfile)
    gc.enable()

    args.command(args)


if __name__ == '__main__':
    main()
