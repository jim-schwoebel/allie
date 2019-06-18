# Tests

To test, all you need to do is run:

```
cd Allie/tests
python3 test.py
```

This would run all necessary tests and make sures everything is running properly.

## Specific tests

Here are the list of automated tests that are in this repository:
- tests for modules and brew installations (FFmpeg and SoX)
- training tests for audio, text, image, video, and .CSV files (via test files and model.py script)
- featurization for audio, text, image, video, and .CSV files via default_featurizers.
- ability to load model files and make predictions via model directory (via test files / load_dir / models trained) 

## Work-in-progress (to add later)

Here are some tests we are working to add in:
- ability to spin up YAML files for trained models (if create_YAML==True in settings.json) 
