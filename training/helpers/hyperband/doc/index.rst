.. project-template documentation master file, created by
   sphinx-quickstart on Mon Jan 18 14:44:12 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to scikit-hyperband's documentation!
===============================================

This project contains an implementation of the hyperband algorithm.

Hyperband is an algorithm that can be used to find a good hyperparameter configuration for (machine learning) algorithms. The idea is based on the assumption that if a configuration is destined to be the best after a large number of iterations, it is more likely than not to perform in the top half of configurations after a small number of iterations [1]_. Hyperband does not emphasize the absolute performance of an algorithm, but more so its relative performance compared with many alternatives trained for the same number of iterations.

The authors of hyperband acknowledge that there are exceptions to this assumption, and the algorithm accounts for this by hedging over varying degrees of aggressiveness balancing breadth versus depth based search (exploration versus exploitation). For a precise description of the algorithm, we refer you to this page [2]_ written by the authors of the algorithm.

    .. toctree::
       :maxdepth: 2
       
       api
       auto_examples/index
       ...

References
==========

.. [1] Li, L., Jamieson, K., DeSalvo, G., Rostamizadeh, A. and Talwalkar, A.,
       2017. Hyperband: A novel bandit-based approach to hyperparameter
       optimization. The Journal of Machine Learning Research, 18(1),
       pp.6765-6816.
.. [2] https://people.eecs.berkeley.edu/~kjamieson/hyperband.html

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

