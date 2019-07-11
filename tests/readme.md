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

- ```python3 class test_dependencies(unittest.TestCase)``` - tests for modules and brew installations (FFmpeg and SoX) 
- ability to clean files via cleaning scripts (mostly de-duplication, will expand in future) 
- ability to augment files via augmentation scripts (in ./datasets/) directory 
- ability to featurize files via default_featurizers
- ability to properly execute model training scripts (via test files and model.py script)
- ability to transcribe files
- ability to compress machine learning models and make production-ready repositories (in ./production directory)
- ability to load model files and make predictions via model directory (via test files / load_dir / models trained) 

No testing suite is 100% perfect, but all tests were designed to be independent from each other. If you think additional things need to be added in, please write us some suggestions in the [GitHub issues forum](https://github.com/jim-schwoebel/allie/issues). 

## Seed test files 

You can seed files with the seed_files.py script. The an example is below:
```
python3 seed_files.py audio /Users/jimschwoebel/allie/train_dir/one
```

Where the 'audio' is the type of file you want to automatically generate and the /Users/jimschwoebel/allie/train_dir/one is the directory you'd like to put the files.

By default, this script generates 20 files at a time and does things like record ambient sound for audio. You can automatically generate csv, audio, text, image, and video files this way :-) 

## Work-in-progress (to add later)

Here are some tests we are working to add in:
- tests that the data schema is proper (as defined in various sections of the document) for all featurizers. Perhaps refactor code to inherit this from one location so that it does not have typos into the future. 
- test ability to spin up YAML files for trained models (if create_YAML==True in settings.json) 
- multi-class (3+) modeling outside of 2 things.

