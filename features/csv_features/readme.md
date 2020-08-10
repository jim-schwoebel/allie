## How to use audio feature API

```
cd allie/features/csv_features
python3 featurize_csv_regression.py -i /Users/jim/desktop/allie/train_dir/age_data.csv -o test.csv -t age
```
where
```
Options:
  -h, --help            show this help message and exit
  -i INPUT, --input=INPUT
                        the .CSV filename input to process
  -o OUTPUT, --output=OUTPUT
                        the .CSV filename output to process
  -t TARGET, --target=TARGET
                        the target class (e.g. age) - will not rename this
                        column.
```

### CSV 

.CSV can include numerical data, categorical data, audio files (./audio.wav), image files (.png), video files (./video.mp4), text files ('.txt' or text column), or other .CSV files. This scope of a table feature is inspired by [D3M schema design proposed by the MIT data lab](https://github.com/mitll/d3m-schema/blob/master/documentation/datasetSchema.md).

* [featurize_csv_regression](https://github.com/jim-schwoebel/allie/blob/master/features/csv_features/featurize_csv_regression.py) - standard CSV feature array that can accomodate audio, image, video, text, and numerical data formats.

### Settings

| setting | description | default setting | all options | 
|------|------|------|------| 
| [default_csv_features](https://github.com/jim-schwoebel/allie/tree/master/features/csv_features) | the default featurization technique(s) used as a part of model training for .CSV files. | ["csv_features_regression"] | ["csv_features_regression"]  | 
| default_csv_transcriber | the default transcription technique for .CSV file spreadsheets. | ["raw text"] | ["raw text"] | 
| transcribe_csv | a setting to define whether or not to transcribe csv files during featurization and model training via the default_csv_transcriber | True | True, False | 
