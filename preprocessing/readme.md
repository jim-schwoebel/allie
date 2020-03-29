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

## Standard feature dictionary (.JSON)

This is the stanard feature array to accomodate all types of samples (audio, text, image, video, or CSV samples):

```python3 
def make_features(sampletype):

	# only add labels when we have actual labels.
	features={'audio':dict(),
		  'text': dict(),
		  'image':dict(),
		  'video':dict(),
		  'csv': dict()}

	transcripts={'audio': dict(),
		     'text': dict(),
		     'image': dict(),
		     'video': dict(),
		      'csv': dict()}

	models={'audio': dict(),
		 'text': dict(),
		 'image': dict(),
		 'video': dict(),
		 'csv': dict()}

	data={'sampletype': sampletype,
		  'transcripts': transcripts,
		  'features': features,
		  'models': models,
		  'labels': []}

	return data
```

Note that there can be audio transcripts, image transcripts, and video transcripts. The image and video transcripts use OCR to characterize text in the image, whereas audio transcripts are transcipts done by traditional speech-to-text systems (e.g. Pocketsphinx). The schema above allows for a flexible definition for transcripts that can accomodate all forms. 

Quick note about the variables and what values they can take.
- Sampletype = 'audio', 'text', 'image', 'video', 'csv'
- Labels = ['classname_1', 'classname_2', 'classname_N...'] - classification problems.
- Labels = [{classname1: 'value'}, {classname2: 'value'}, ... {classnameN: 'valueN'}] where values are between [0,1] - regression problems. 

Note that only .CSV files may have audio, text, image, video features all-together (as the .CSV can contain files in a current directory that need to be featurized together). Otherwise, audio files likely will have audio features, text files will have text features, image files will have image features, and video files will have video features. 

## Not Implemented / Work in progress

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
