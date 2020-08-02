import numpy as np 
import json, pickle
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from tpot.builtins import OneHotEncoder

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
g=json.load(open('test_tpot_classifier.json'))
tpot_data=np.array(g['labels'])
features=np.array(g['data'])

training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data, random_state=None)

# Average CV score on the training set was: 0.7757352941176471
exported_pipeline = make_pipeline(
    OneHotEncoder(minimum_fraction=0.2, sparse=False, threshold=10),
    ExtraTreesClassifier(bootstrap=False, criterion="entropy", max_features=0.25, min_samples_leaf=13, min_samples_split=9, n_estimators=100)
)

exported_pipeline.fit(training_features, training_target)
print('saving classifier to disk')
f=open('test_tpot_classifier.pickle','wb')
pickle.dump(exported_pipeline,f)
f.close()
