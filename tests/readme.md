# Tests

To test, all you need to do is run:

```
cd Allie/tests
python3 test.py
```

This would run all necessary tests and make sures everything is running properly.

![](https://github.com/jim-schwoebel/Allie/blob/master/tests/helpers/tests.gif)

## Specific tests

Here is the list of automated tests that are in this repository:
- tests for modules and brew installations (FFmpeg and SoX)
- training tests for audio, text, image, video, and .CSV files (via test files and model.py script)
- featurization for audio, text, image, video, and .CSV files via default_featurizers.
- ability to load model files and make predictions via model directory (via test files / load_dir / models trained) 

## Seed test files 

You can seed files with the seed_files.py script. The an example is below:
```
python3 seed_files.py audio /Users/jimschwoebel/allie/train_dir/one
```

Where the 'audio' is the type of file you want to automatically generate and the /Users/jimschwoebel/allie/train_dir/one is the directory you'd like to put the files.

By default, this script generates 20 files at a time and does things like record ambient sound for audio. You can automatically generate csv, audio, text, image, and video files this way :-) 

## Work-in-progress (to add later)

Here are some tests we are working to add in:
- test ability to spin up YAML files for trained models (if create_YAML==True in settings.json) 
- test cleaning abilities (for audio, text, image, video, csv files)
- test augmentation capabilities (for audio, text, image, video, csv files)
- multi-class (3+) modeling outside of 2 things.

