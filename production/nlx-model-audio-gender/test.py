import os
import unittest
import classify
import json
import numpy as np

cwd = os.path.dirname(os.path.abspath(__file__))
test_dir = os.path.abspath(cwd + '/test')

class ClassifyTest(unittest.TestCase):
  def test_male_classification(self):
    features_data = open(test_dir + '/male_features.json', 'r').read()
    features = np.array(json.loads(features_data)).reshape(1, -1)
    results = classify.classify(features)
    self.assertEqual(results, 0)

  def test_female_classification(self):
    features_data = open(test_dir + '/female_features.json', 'r').read()
    features = np.array(json.loads(features_data)).reshape(1, -1)
    results = classify.classify(features)
    self.assertEqual(results, 1)

if __name__ == '__main__':
  unittest.main()
