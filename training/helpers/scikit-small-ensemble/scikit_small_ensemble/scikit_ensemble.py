import lz4 as zlib
import tempfile
import joblib
import os
try:
    import _pickle as pickle
except ImportError:
    try:
        import cPickle as pickle
    except ImportError:
        print('cPickle is not installed. Using the builtin pickle instead.')
        import pickle


class CompressedEstimators(object):

    def __init__(self, estimators, ratio):
        self.cutoff = int(len(estimators) * ratio)
        self.estimators = [
            zlib.compress(pickle.dumps(x)) if i < self.cutoff else x
            for i, x in enumerate(estimators)
        ]

    def __getitem__(self, index):
        estimator = self.estimators[index]
        if index < self.cutoff:
            return pickle.loads(zlib.decompress(estimator))
        else:
            return estimator

    def __len__(self):
        return len(self.estimators)


class DiskEstimators(object):

    def __init__(self, estimators, ratio):
        self.cutoff = int(len(estimators) * ratio)
        self.saved_dir = tempfile.mkdtemp()
        for i in range(self.cutoff):
            joblib.dump(estimators[i], os.path.join(self.saved_dir, str(i)), compress=0)
        self.estimators = [
            os.path.join(self.saved_dir, str(i)) if i < self.cutoff else x
            for i, x in enumerate(estimators)
        ]

    def __getitem__(self, index):
        estimator = self.estimators[index]
        if index < self.cutoff:
            return joblib.load(estimator, mmap_mode='r')
        else:
            return estimator

    def __len__(self):
        return len(self.estimators)
