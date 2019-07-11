import os
import json
from .pyAudioLex import process_audio
cwd = os.path.dirname(os.path.abspath(__file__))
male_training_folder = os.path.join(cwd, '..', 'data/training/male 
')female_training_folder = os.path.join(cwd, '..', 'data/training/female 
')# process file
def process_file(filepath):
  print('Processing... ' + filepath)
  results = process_audio(filepath)
  print(json.dumps(results['standard_feature_array'].tolist()))
  return results

# process all the data in a folder
def process_dataset(folder):
  files = os.listdir(folder)
  for file in files:
    filepath = folder + '/' + file
    process_file(filepath)
    break
process_file(os.path.join(male_training_folder, 'male.wav')) 
