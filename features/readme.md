## Featurization scripts

![](https://github.com/jim-schwoebel/allie/blob/master/annotation/helpers/assets/featurize.png)

This is a folder for extracting features from audio, text, image, video, or .CSV files. This is done via a convention for featurizers, which are in the proper folders (e.g. audio files --> audio_features). In this way, we can appropriately create featurizers for various sample data types according to the default featurizers as specified in the [settings.json](https://github.com/jim-schwoebel/allie/blob/master/settings.json).

Note that as part of the featurization process, files can also be transcribed by default transcription types as specified in [settings.json](https://github.com/jim-schwoebel/allie/blob/master/settings.json).

## How to featurize folders of files 

To featurize an entire folder of a certain file type (e.g. audio files of .WAV format), you can run:

```
cd ~ 
cd allie/features/audio_features
python3 featurize.py /Users/jimschwoebel/allie/load_dir
```

The code above will featurize all the audio files in the folderpath via the default_featurizer specified in the settings.json file (e.g. 'standard_features'). 

If you'd like to use a different featurizer you can specify it optionally:

```
cd ~ 
cd allie/features/audio_features
python3 featurize.py /Users/jimschwoebel/allie/load_dir librosa_features
```

Note you can extend this to any of the feature types. The table below overviews how you could call each as a featurizer. In the code below, you must be in the proper folder (e.g. ./allie/features/audio_features for audio files, ./allie/features/image_features for image files, etc.) for the scripts to work properly.

| Data type | Supported formats | Call to featurizer a folder | Current directory must be | 
| --------- |  --------- |  --------- | --------- | 
| audio files | .MP3 / .WAV | ```python3 featurize.py [folderpath]``` | ./allie/features/audio_features | 
| text files | .TXT | ```python3 featurize.py [folderpath]``` | ./allie/features/text_features| 
| image files | .PNG | ```python3 featurize.py [folderpath]``` | ./allie/features/image_features | 
| video files | .MP4 | ```python3 featurize.py [folderpath]``` |./allie/features/video_features | 
| csv files | .CSV | ```python3 featurize.py [folderpath]``` | ./allie/features/csv_features | 

## [Standard feature dictionary (.JSON)](https://github.com/jim-schwoebel/allie/blob/master/features/standard_array.py)

This is the standard feature array to accomodate all types of samples (audio, text, image, video, or CSV samples):

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
	
	# getting settings can be useful to see if settings are the same in every
	# featurization, as some featurizations can rely on certain settings to be consistent
	prevdir=prev_dir(os.getcwd())
	try:
		settings=json.load(open(prevdir+'/settings.json'))
	except:
		# this is for folders that may be 2 layers deep in train_dir
		settings=json.load(open(prev_dir(prevdir)+'/settings.json'))
	
	data={'sampletype': sampletype,
		  'transcripts': transcripts,
		  'features': features,
		  'models': models,
		  'labels': [],
		  'errors': [],
		  'settings': settings,
		 }
	
	return data
```
There are many advantages for having this schema including:
- **sampletype definition flexibility** - flexible to 'audio' (.WAV / .MP3), 'text' (.TXT / .PPT / .DOCX), 'image' (.PNG / .JPG), 'video' (.MP4), and 'csv' (.CSV). This format can also can adapt into the future to new sample types, which can also tie to new featurization scripts. By defining a sample type, it can help guide how data flows through model training and prediction scripts. 
- **transcript definition flexibility** - transcripts can be audio, text, image, video, and csv transcripts. The image and video transcripts use OCR to characterize text in the image, whereas audio transcripts are transcipts done by traditional speech-to-text systems (e.g. Pocketsphinx). You can also add multiple transcripts (e.g. Google and PocketSphinx) for the same sample type.
- **featurization flexibility** - many types of features can be put into this array of the same data type. For example, an audio file can be featurized with 'standard_features' and 'praat_features' without really affecting anything. This eliminates the need to re-featurize and reduces time to sort through multiple types of featurizations during the data cleaning process.
- **label annotation flexibility** - can take the form of ['classname_1', 'classname_2', 'classname_N...'] - classification problems and [{classname1: 'value'}, {classname2: 'value'}, ... {classnameN: 'valueN'}] where values are between [0,1] for regression problems. 
- **model predictions** - one survey schema can be used for making model predictions and updating the schema with these predictions. Note that any model that is used for training can be used to make predictions in the load_dir. 
- **visualization flexibility** - can easily visualize features of any sample type through Allie's [visualization script](https://github.com/jim-schwoebel/allie/tree/master/visualize) (e.g. tSNE plots, correlation matrices, and more).
- **error tracing** - easily trace errors associated with featurization and/or modeling to review what is happening during a session.

This schema is inspired by [D3M-schema](https://github.com/mitll/d3m-schema/blob/master/documentation/datasetSchema.md) by the MIT media lab.

## Implemented 

Note that all scripts implemented have features and their corresponding labels. It is important to provide labels to understand what the features correspond to. It's also to keep in mind the relative speeds of featurization to optimize server costs (they are provided here for reference).

### Audio
* [audioset_features](https://github.com/jim-schwoebel/allie/blob/master/features/audio_features/audioset_features.py)
* [audiotext_features](https://github.com/jim-schwoebel/allie/blob/master/features/audio_features/audiotext_features.py)
* [librosa_features](https://github.com/jim-schwoebel/allie/blob/master/features/audio_features/librosa_features.py)
* [loudness_features](https://github.com/jim-schwoebel/allie/blob/master/features/audio_features/loudness_features.py)
* [meta_features](https://github.com/jim-schwoebel/allie/blob/master/features/audio_features/meta_features.py)
* [mixed_features](https://github.com/jim-schwoebel/allie/blob/master/features/audio_features/mixed_features.py) - random combinations of audio and text features (via ratios)
* [opensmile_features](https://github.com/jim-schwoebel/allie/blob/master/features/audio_features/opensmile_features.py) - 14 embeddings with OpenSMILE possible here; defaults to GeMAPSv01a.conf.
* [praat_features](https://github.com/jim-schwoebel/allie/blob/master/features/audio_features/praat_features.py)
* [prosody_features](https://github.com/jim-schwoebel/allie/blob/master/features/audio_features/prosody_features.py) - prosody using Google's VAD
* [pspeech_features](https://github.com/jim-schwoebel/allie/blob/master/features/audio_features/pspeech_features.py) 
* [pyaudio_features](https://github.com/jim-schwoebel/allie/blob/master/features/audio_features/pyaudio_features.py)
* [pyaudiolex_features]() - time series features for audio
* [pyworld_features](https://github.com/jim-schwoebel/allie/blob/master/features/audio_features/pyworld_features.py) - f0 and and spectrogram features
* [sa_features](https://github.com/jim-schwoebel/allie/blob/master/features/audio_features/sa_features.py)
* [sox_features](https://github.com/jim-schwoebel/allie/blob/master/features/audio_features/sox_features.py)
* [speechmetrics_features](https://github.com/aliutkus/speechmetrics) - estimating speech quality.
* [specimage_features](https://github.com/jim-schwoebel/allie/blob/master/features/audio_features/specimage_features.py)
* [specimage2_features](https://github.com/jim-schwoebel/allie/blob/master/features/audio_features/specimage2_features.py)
* [spectrogram_features](https://github.com/jim-schwoebel/allie/blob/master/features/audio_features/spectrogram_features.py) 
* [standard_features](https://github.com/jim-schwoebel/allie/blob/master/features/audio_features/standard_features.py) - standard audio feature array (default)

### Text
* [bert features](https://github.com/UKPLab/sentence-transformers) - extract BERT-related features from sentences (note shorter sentences run faster here, and long text can lead to long featurization times).
* [fast_features](https://github.com/jim-schwoebel/allie/blob/master/features/text_features/fast_features.py)
* [glove_features](https://github.com/jim-schwoebel/allie/blob/master/features/text_features/glove_features.py)
* [grammar_features](https://github.com/jim-schwoebel/allie/blob/master/features/text_features/grammar_features.py) - 85k+ grammar features (memory intensive)
* [nltk_features](https://github.com/jim-schwoebel/allie/blob/master/features/text_features/nltk_features.py) - standard text feature array (default)
* [spacy_features](https://github.com/jim-schwoebel/allie/blob/master/features/text_features/spacy_features.py)
* [textacy_features](https://github.com/jim-schwoebel/allie/blob/master/features/text_features/textacy_features.py) - a variety of document classification and topic modeling features
* [text_features](https://github.com/jim-schwoebel/allie/blob/master/features/text_features/text_features.py) - many different types of features like emotional word counts, total word counts, Honore's statistic and others.
* [w2v_features](https://github.com/jim-schwoebel/allie/blob/master/features/text_features/w2vec_features.py) - note this is the largest model from Google and may crash your computer if you don't have enough memory. I'd recommend fast_features if you're looking for a pre-trained embedding.

### Images 
* [image_features](https://github.com/jim-schwoebel/allie/blob/master/features/image_features/image_features.py) - standard image feature array (default)
* [inception_features](https://github.com/jim-schwoebel/allie/blob/master/features/image_features/inception_features.py)
* [resnet_features](https://github.com/jim-schwoebel/allie/blob/master/features/image_features/resnet_features.py)
* [squeezenet_features](https://github.com/rcmalli/keras-squeezenet) - efficient memory footprint
* [tesseract_features](https://github.com/jim-schwoebel/allie/blob/master/features/image_features/tesseract_features.py)
* [vgg16_features](https://github.com/jim-schwoebel/allie/blob/master/features/image_features/vgg16_features.py) 
* [vgg19_features](https://github.com/jim-schwoebel/allie/blob/master/features/image_features/vgg19_features.py) 
* [xception_features](https://github.com/jim-schwoebel/allie/blob/master/features/image_features/xception_features.py) 

### Videos 
* [video_features](https://github.com/jim-schwoebel/allie/blob/master/features/video_features/video_features.py) - standard video feature array (default)
* [y8m_features](https://github.com/jim-schwoebel/allie/blob/master/features/video_features/y8m_features.py) 

### CSV 

.CSV can include numerical data, categorical data, audio files (./audio.wav), image files (.png), video files (./video.mp4), text files ('.txt' or text column), or other .CSV files. 

* [csv_features](https://github.com/jim-schwoebel/allie/blob/master/features/csv_features/csv_features.py) - standard CSV feature array

## [Settings](https://github.com/jim-schwoebel/allie/blob/master/settings.json)

Here are some features settings that can be customized with Allie's API. Settings can be modified in the [settings.json](https://github.com/jim-schwoebel/allie/blob/master/settings.json) file. 

Here are some settings that you can modify in this settings.json file and the various options for these settings:

| setting | description | default setting | all options | 
|------|------|------|------| 
| version | version of Allie release | 1.0 | 1.0 |
| [default_audio_features](https://github.com/jim-schwoebel/voice_modeling/tree/master/features/audio_features) | default set of audio features used for featurization (list). | ["standard_features"] | ["audioset_features", "audiotext_features", "librosa_features", "meta_features", "mixed_features", "opensmile_features", "praat_features", "prosody_features", "pspeech_features", "pyaudio_features", "pyaudiolex_features", "sa_features", "sox_features", "specimage_features", "specimage2_features", "spectrogram_features", "speechmetrics_features", "standard_features"] | 
| default_audio_transcriber | the default transcription model used during audio featurization if trainscribe_audio == True | ["deepspeech_dict"] | ["pocketsphinx", "deepspeech_nodict", "deepspeech_dict", "google", "wit", "azure", "bing", "houndify", "ibm"] | 
| [default_csv_features](https://github.com/jim-schwoebel/allie/tree/master/features/csv_features) | the default featurization technique(s) used as a part of model training for .CSV files. | ["csv_features_regression"] | ["csv_features_regression"]  | 
| default_csv_transcriber | the default transcription technique for .CSV file spreadsheets. | ["raw text"] | ["raw text"] | 
| [default_image_features](https://github.com/jim-schwoebel/voice_modeling/tree/master/features/image_features) | default set of image features used for featurization (list). | ["image_features"] | ["image_features", "inception_features", "resnet_features", "squeezenet_features", "tesseract_features", "vgg16_features", "vgg19_features", "xception_features"] | 
| default_image_transcriber | the default transcription technique used for images (e.g. image --> text transcript) | ["tesseract"] | ["tesseract"] |
| [default_text_features](https://github.com/jim-schwoebel/voice_modeling/tree/master/features/csv_features) | default set of text features used for featurization (list). | ["nltk_features"] | ["bert_features", "fast_features", "glove_features", "grammar_features", "nltk_features", "spacy_features", "text_features", "w2v_features"] | 
| default_text_transcriber | the default transcription techniques used to parse raw .TXT files during model training| ["raw_text"] | ["raw_text"]  | 
| [default_video_features](https://github.com/jim-schwoebel/voice_modeling/tree/master/features/video_features) | default set of video features used for featurization (list). | ["video_features"] | ["video_features", "y8m_features"] | 
| default_video_transcriber | the default transcription technique used for videos (.mp4 --> text from the video) | ["tesseract (averaged over frames)"] | ["tesseract (averaged over frames)"] |
| test_size | a setting that specifies the size of the testing dataset for defining model performance after model training. | 0.10 | Any number 0.10-0.50 | 
| transcribe_audio | a setting to define whether or not to transcribe audio files during featurization and model training via the default_audio_transcriber | True | True, False | 
| transcribe_csv | a setting to define whether or not to transcribe csv files during featurization and model training via the default_csv_transcriber | True | True, False | 
| transcribe_image | a setting to define whether or not to transcribe image files during featurization and model training via the default_image_transcriber | True | True, False | 
| transcribe_text | a setting to define whether or not to transcribe text files during featurization and model training via the default_image_transcriber | True | True, False | 
| transcribe_video | a setting to define whether or not to transcribe video files during featurization and model training via the default_video_transcriber | True | True, False | 
