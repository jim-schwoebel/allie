import os
from pymongo import MongoClient
import numpy as np
import classify

# configure mongo
MONGO_URL = os.environ['MONGO_URL']
MONGO_DB = os.environ['MONGO_DB']
client = MongoClient(MONGO_URL)
db = client[MONGO_DB]
v1Features = db.v1Features
samples = db.samples

# process the payload
def process(payload):
  sampleID = payload['id']
  print('Classifying for ' + sampleID)

  # get features and classify
  v1Feature = v1Features.find_one({ "sampleID": sampleID })
  # check that we have the feature array
  if 'standard_feature_array' in v1Feature['features']['audio']:
    features_data = v1Feature['features']['audio']['standard_feature_array']
    features = np.array(features_data).reshape(1, -1)
    result = classify.classify(features)
    # save to mongo
    print(f'Inserted into mongo with gender: {result}')
    v1Features.find_one_and_update({ "sampleID": sampleID },
                                   { "$set": { "model.gender": result } })
    samples.find_one_and_update({ "sampleId": sampleID },
                                       { "$set": { "model.gender": result } })
  # otherwise skip
  else:
    print('Skipping as could not find the proper features')
