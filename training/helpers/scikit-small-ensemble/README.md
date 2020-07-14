scikit-small-ensemble
=====================

scikit-small-ensemble is a library to make your ensemble models(Random Forest Classifier, etc) have a small memory footprint/usage.


Introduction
============

Ensemble models can be very memory-intensive sometimes, for example, depending on the number of estimators and its depths, if you think of a tree-based ensemble model. This library wraps each estimator and compress its contents in LZ4 unless it's used. It trades performance for reduced memory usage.


Installation
============

```
$ pip install scikit-small-ensemble
```


Usage
=====


```python

# random forest ensemble model
from scikit_small_ensemble import compress, memory_map

# WARNING: This changes the model object itself.
# ratio is [0.0, 1.0], where 1.0 is most compressed.
compress(model, ratio=0.2)

# Or, you can memory-map estimators from the disk on demand.
# memory_map(model, ratio=1.0)

# Use it like nothing happened.
# Memory usage becomes 10x lower with ratio=1.0
Y = model.predict_proba(X)
```
