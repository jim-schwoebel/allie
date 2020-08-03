import numpy as np 
import json, pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.svm import LinearSVC

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
g=json.load(open('gender_tpot_classifier.json'))
tpot_data=np.array(g['labels'])
features=np.array(g['data'])

training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data, random_state=None)

# Average CV score on the training set was: 0.8276292335115866
exported_pipeline = make_pipeline(
    Normalizer(norm="max"),
    LinearSVC(C=20.0, dual=True, loss="hinge", penalty="l2", tol=0.0001)
)

exported_pipeline.fit(training_features, training_target)
print('saving classifier to disk')
f=open('gender_tpot_classifier.pickle','wb')
pickle.dump(exported_pipeline,f)
f.close()
