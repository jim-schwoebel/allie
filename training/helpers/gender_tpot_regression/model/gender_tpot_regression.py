import numpy as np 
import json, pickle
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
g=json.load(open('gender_tpot_regression.json'))
tpot_data=np.array(g['labels'])
features=np.array(g['data'])

training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data, random_state=None)

# Average CV score on the training set was: -0.13558964188025885
exported_pipeline = make_pipeline(
    SelectPercentile(score_func=f_regression, percentile=19),
    StackingEstimator(estimator=ElasticNetCV(l1_ratio=0.2, tol=0.0001)),
    StackingEstimator(estimator=ExtraTreesRegressor(bootstrap=True, max_features=0.7000000000000001, min_samples_leaf=14, min_samples_split=15, n_estimators=100)),
    ElasticNetCV(l1_ratio=0.9, tol=0.001)
)

exported_pipeline.fit(training_features, training_target)
print('saving classifier to disk')
f=open('gender_tpot_regression.pickle','wb')
pickle.dump(exported_pipeline,f)
f.close()
