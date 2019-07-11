import os
from pymongo import MongoClient
import featurize_training_data

cwd = os.path.dirname(os.path.abspath(__file__))
male_training_folder = os.path.join(cwd, '..', 'data/training/male')
female_training_folder = os.path.join(cwd, '..', 'data/training/female')
# configure mongo
MONGO_URL = os.environ.get('MONGO_URL', 'mongodb://mongo-service:27017')
MONGO_DB = os.environ.get('MONGO_DB', 'nlx-data')
client = MongoClient(MONGO_URL)
db = client[MONGO_DB]
v1Features = db.v1Features
results = featurize_training_data.process_file(os.path.join(male_training_folder, 'male.wav'))
print(results)