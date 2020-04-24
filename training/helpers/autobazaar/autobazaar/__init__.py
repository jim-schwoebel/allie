# -*- coding: utf-8 -*-

"""
AutoBazaar top module.

AutoBazaar is an AutoML system created to execute the experiments associated with the
[The Machine Learning Bazaar Paper: Harnessing the ML Ecosystem for Effective System
Development](https://arxiv.org/pdf/1905.08942.pdf)
by the [Human-Data Interaction (HDI) Project](https://hdi-dai.lids.mit.edu/) at LIDS, MIT.

* Free software: MIT license
* Documentation: https://HDI-Project.github.io/AutoBazaar
"""
import os

import git

__author__ = 'MIT Data To AI Lab'
__copyright__ = 'Copyright (c) 2019, MIT Data To AI Lab'
__email__ = 'dailabmit@gmail.com'
__license__ = 'MIT'
__version__ = '0.2.1-dev'


def _get_commit():
    try:
        base_path = os.path.dirname(__file__)
        repo = git.Repo(base_path, search_parent_directories=True)
        commit = repo.commit().hexsha[0:7]
        if repo.is_dirty(untracked_files=False):
            commit += '*'

        return commit
    except git.InvalidGitRepositoryError:
        return None


def get_version():
    commit = _get_commit()
    if commit:
        return '{} - {}'.format(__version__, commit)

    return __version__
