## Production directory

From the command line, generate relevant repository for a trained machine 
learning model.

This assumes a standard feature array (for now), but it is expected we can 
adapt this to the future schema presented in this repository.

## How to call from command line 

```python3 
python3 create_yaml.py [sampletype] [simple_model_name] [model_name] [jsonfilename] [class 1] [class 2] ... [class N]
```

Where:
- sampletype = 'audio' | 'video' | 'text' | 'image' | 'csv'
- simple_model name = any string for a common name for model (e.g. gender for male_female_sc_classification.pickle)
- model_name = male_female_sc_classification.pickle
- jsonfile_name = male_female_sc_classification.json (JSON file with model information regarding accuracy, etc.)
- classes = ['male', 'female']

## Quick example

Assuming you have trained a model (stressed_calm_sc_classification.json) in the model directory (audio_models) with stressed_features.json and calm_features.json test cases, you can run the code as follows.

```python3
python3 create_yaml.py audio stress stressed_calm_sc_classification.pickle stressed_calm_sc_classification.json stressed calm  
```

This will then create a repository nlx-model-stress that can be used for production purposes. Note that automated tests 
require testing data, and some of this data can be provided during model training.

Now you can test the docker container by going to the directory and running docker build . 

```
cd nlx-model-stress
docker bulid .
```

if all is good, the tests should indicate it (like below).
```

```

If you have a fail, it's likely due to outputs in the model not being correct. In this case, you may need to manually tune the outputs.
```
======================================================================
FAIL: test_calm_classification (__main__.ClassifyTest)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "test.py", line 21, in test_calm_classification
    self.assertEqual(results, 1)
AssertionError: 'calm' != 1

======================================================================
FAIL: test_stressed_classification (__main__.ClassifyTest)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "test.py", line 15, in test_stressed_classification
    self.assertEqual(results, 0)
AssertionError: 'stressed' != 0

----------------------------------------------------------------------
Ran 2 tests in 0.011s

FAILED (failures=2)
```
