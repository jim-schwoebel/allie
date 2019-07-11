import os 
import pickle 
import json 
import numpy as np 

# configure the models
cwd = os.path.dirname(os.path.abspath(__file__)) 
data_dir = os.path.abspath(cwd + '/data') 
model_data = open(data_dir + '/male_female_standard_features_tpotclassifier.pickle', 'rb') 
model = pickle.load(model_data) 
model_data.close()

# classify 
def classify(features):
  results = model.predict(features)
  return results[0]