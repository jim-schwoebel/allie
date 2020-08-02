import numpy as np 
import json, pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.tree import DecisionTreeClassifier
from tpot.builtins import StackingEstimator
from sklearn.preprocessing import FunctionTransformer
from copy import copy

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
g=json.load(open('test_tpot_classifier.json'))
tpot_data=np.array(g['labels'])
features=np.array(g['data'])

training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data, random_state=None)

# Average CV score on the training set was: 0.5428571428571429
exported_pipeline = make_pipeline(
    make_union(
        FunctionTransformer(copy),
        FunctionTransformer(copy)
    ),
    DecisionTreeClassifier(criterion="gini", max_depth=5, min_samples_leaf=5, min_samples_split=16)
)

exported_pipeline.fit(training_features, training_target)
print('saving classifier to disk')
f=open('test_tpot_classifier.pickle','wb')
pickle.dump(exported_pipeline,f)
f.close()
