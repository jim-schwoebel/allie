# Tests

To test, all you need to do is run:

```
cd Allie/tests
python3 test.py
```

This would run all necessary tests and make sures everything is running properly.

![](https://github.com/jim-schwoebel/Allie/blob/master/tests/helpers/tests.gif)

## Specific tests

Here is the list of automated tests that are included in the unit_test.py script:
```python3 
class test_dependencies(unittest.TestCase):
``` 
- tests for modules and brew installations (FFmpeg and SoX) 
```python3 
class test_cleaning(unittest.TestCase):
```
- ability to clean files via cleaning scripts (mostly de-duplication, will expand in future) 
```python3
class test_augmentation(unittest.TestCase):
```
- ability to augment files via augmentation scripts (in ./datasets/) directory 
```python3
class test_features(unittest.TestCase):
```
- ability to featurize files via default_featurizers
```python3
class test_transcription(unittest.TestCase):
```
- ability to transcribe files
```python3
class test_training(unittest.TestCase):
```
- ability to train machine learning models (classification and regression) with all settings
```python3
class test_preprocessing(unittest.TestCase):
```
- ability to create transformations with the transform.py script (for model training)
```python3
class test_loading(unittest.TestCase):
```
- ability to load model files and make predictions via model directory (via test files / load_dir / models trained) 
```python3
class test_visualization(unittest.TestCase):
```
- ability to visualize classification problems through the visualize.py script

No testing suite is 100% perfect, but all tests were designed to be independent from each other. If you think additional things need to be added in, please write us some suggestions in the [GitHub issues forum](https://github.com/jim-schwoebel/allie/issues). 

## Seed test files 

You can seed files with the seed_files.py script. The an example is below:
```
python3 seed_files.py audio /Users/jimschwoebel/allie/train_dir/one
```

Where the 'audio' is the type of file you want to automatically generate and the /Users/jimschwoebel/allie/train_dir/one is the directory you'd like to put the files.

By default, this script generates 20 files at a time and does things like record ambient sound for audio. You can automatically generate csv, audio, text, image, and video files this way :-) 
