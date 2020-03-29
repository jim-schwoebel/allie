## Transformation scripts

This is a folder for manipulating and pre-processing features from audio, text, image, video, or .CSV files. 

This is done via a convention for transformers, which are in the proper folders (e.g. audio files --> audio_transformers). In this way, we can appropriately create transformers for various sample data types. 

## How to transform folders of featurized files

To transform an entire folder of a featurized files, you can run:

```
cd ~ 
cd allie/preprocessing/audio_transformers
python3 transform.py /Users/jimschwoebel/allie/train_dir/classA
```

The code above will transform all the featurized audio files (.JSON) in the folderpath via the default_transformer specified in the settings.json file (e.g. 'standard_transformer'). 

If you'd like to use a different transformer you can specify it optionally:

```
cd ~ 
cd allie/features/audio_transformer
python3 featurize.py /Users/jimschwoebel/allie/load_dir standard_scalar
```

Note you can extend this to any of the feature types. The table below overviews how you could call each as a featurizer. In the code below, you must be in the proper folder (e.g. ./allie/features/audio_features for audio files, ./allie/features/image_features for image files, etc.) for the scripts to work properly.

| Data type | Supported formats | Call to featurizer a folder | Current directory must be | 
| --------- |  --------- |  --------- | --------- | 
| audio files | .MP3 / .WAV | ```python3 transform.py [folderpath]``` | ./allie/features/audio_transformers | 
| text files | .TXT | ```python3 transform.py [folderpath]``` | ./allie/features/text_transformers | 
| image files | .PNG | ```python3 transform.py [folderpath]``` | ./allie/features/image_transformers | 
| video files | .MP4 | ```python3 transform.py [folderpath]``` |./allie/features/video_transformers | 
| csv files | .CSV | ```python3 transform.py [folderpath]``` | ./allie/features/csv_transformers | 

## Not Implemented / Work in progress

### All data
* [scikit-learn](https://scikit-learn.org/stable/modules/preprocessing.html)

### Audio
* [Wavelet transforms](http://ataspinar.com/2018/12/21/a-guide-for-using-the-wavelet-transform-in-machine-learning/) - could be useful for dataset augmentation techniques.

### Text
* coming soon

### Images 
* coming soon

### Videos 
* coming soon

### CSV 
* coming soon
