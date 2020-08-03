## How to use audio feature API

```
cd allie/features/csv_features
python3 featurize.py [folder]
```

### CSV 

.CSV can include numerical data, categorical data, audio files (./audio.wav), image files (.png), video files (./video.mp4), text files ('.txt' or text column), or other .CSV files. This scope of a table feature is inspired by [D3M schema design proposed by the MIT data lab](https://github.com/mitll/d3m-schema/blob/master/documentation/datasetSchema.md).

* [csv_features](https://github.com/jim-schwoebel/allie/blob/master/features/csv_features/csv_features.py) - standard CSV feature array
