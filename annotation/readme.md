## How to annotate

```python3
python3 annotate.py --directory {dirname} --sampletype {audio}
```

Converting to csv

```
python3 annotate2csv --directory {dirname}
```

## Annotation 
- Audio - sound_event_detection (can make this)
- Text - https://github.com/doccano/doccano
- Image - https://github.com/tzutalin/labelImg/tree/c1c1dbef315df52daad9b22a418c2e832b60dae5
- Video - 

## Can now use for classification and/or regression

### classiifcation
Sort files into folders {A} or {B} or {C} in train_dir

### regression
Create .CSV spreadsheet useful for regression problems in train_dir (copy here)

### things to do
- classification and/or regression problem (if regression, float, if classification, integer value)
- add classnames (to Optparser)
- add ability to split up each file into more files in the folder {audio file / video file --> audio file 1, 2, n...} 
