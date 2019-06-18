## Featurization scripts

This is a folder for extracting features from audio, text, image, video, or .CSV files. 

## How to use as an individual featurizer

To featurize an entire folder of a certain file type, you can run:

```
cd features/audio_features
python3 featurize.py [folderpath]
```

The code above will featurize all the audio files in the folderpath via the default_featurizer specified in the settings.json file. 

Note you can extend this to any of the feature types. The table below overviews how you could call each as a featurizer. In the code below, you must be in the proper folder (e.g. ./allie/features/audio_features for audio files, ./allie/features/image_features for image files, etc.) for the scripts to work properly.

| Data type | Supported formats | Call to featurizer a folder | Current directory must be | 
| --------- |  --------- |  --------- | --------- | 
| audio files | .MP3 / .WAV | ```python3 featurize.py [folderpath]``` | ./allie/features/audio_features | 
| text files | .MP3 / .WAV | ```python3 featurize.py [folderpath]``` | ./allie/features/text_features | 
| image files | .MP3 / .WAV | ```python3 featurize.py [folderpath]``` | ./allie/features/image_features | 
| video files | .MP3 / .WAV | ```python3 featurize.py [folderpath]``` | ./allie/features/video_features | 
| csv files | .MP3 / .WAV | ```python3 featurize.py [folderpath]``` | ./allie/features/csv_features | 

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

## Implemented 

Note that all scripts implemented have features and their corresponding labels. It is important to provide labels to understand what the features correspond to. It's also to keep in mind the relative speeds of featurization to optimize server costs (they are provided here for reference).

### Audio
* [audioset_features](https://github.com/jim-schwoebel/allie/blob/master/features/audio_features/audioset_features.py)
* [audiotext_features](https://github.com/jim-schwoebel/allie/blob/master/features/audio_features/audiotext_features.py)
* [librosa_features](https://github.com/jim-schwoebel/allie/blob/master/features/audio_features/librosa_features.py)
* [meta_features](https://github.com/jim-schwoebel/allie/blob/master/features/audio_features/meta_features.py)
* [mixed_features](https://github.com/jim-schwoebel/allie/blob/master/features/audio_features/mixed_features.py) - random combinations of audio and text features (via ratios)
* [myprosody_features](https://github.com/jim-schwoebel/allie/blob/master/features/audio_features/myprosody_features.py) - sometimes unstable 
* [praat_features](https://github.com/jim-schwoebel/allie/blob/master/features/audio_features/praat_features.py)
* [pspeech_features](https://github.com/jim-schwoebel/allie/blob/master/features/audio_features/pspeech_features.py) 
* [pyaudio_features](https://github.com/jim-schwoebel/allie/blob/master/features/audio_features/pyaudio_features.py)
* [sa_features](https://github.com/jim-schwoebel/allie/blob/master/features/audio_features/sa_features.py)
* [sox_features](https://github.com/jim-schwoebel/allie/blob/master/features/audio_features/sox_features.py)
* [specimage_features](https://github.com/jim-schwoebel/allie/blob/master/features/audio_features/specimage_features.py)
* [specimage2_features](https://github.com/jim-schwoebel/allie/blob/master/features/audio_features/specimage2_features.py)
* [spectrogram_features](https://github.com/jim-schwoebel/allie/blob/master/features/audio_features/spectrogram_features.py) 
* [standard_features](https://github.com/jim-schwoebel/allie/blob/master/features/audio_features/standard_features.py) - standard audio feature array (default)

### Text
* [fast_features](https://github.com/jim-schwoebel/allie/blob/master/features/text_features/fast_features.py)
* [glove_features](https://github.com/jim-schwoebel/allie/blob/master/features/text_features/glove_features.py) 
* [nltk_features](https://github.com/jim-schwoebel/allie/blob/master/features/text_features/nltk_features.py) - standard text feature array (default)
* [spacy_features](https://github.com/jim-schwoebel/allie/blob/master/features/text_features/spacy_features.py) 
* [w2vec_features](https://github.com/jim-schwoebel/allie/blob/master/features/text_features/w2vec_features.py) 

### Images 
* [image_features](https://github.com/jim-schwoebel/allie/blob/master/features/image_features/image_features.py) - standard image feature array (default)
* [inception_features](https://github.com/jim-schwoebel/allie/blob/master/features/image_features/inception_features.py)
* [resnet_features](https://github.com/jim-schwoebel/allie/blob/master/features/image_features/resnet_features.py)
* [tesseract_features](https://github.com/jim-schwoebel/allie/blob/master/features/image_features/tesseract_features.py)
* [vgg16_features](https://github.com/jim-schwoebel/allie/blob/master/features/image_features/vgg16_features.py) 
* [vgg19_features](https://github.com/jim-schwoebel/allie/blob/master/features/image_features/vgg19_features.py) 
* [xception_features](https://github.com/jim-schwoebel/allie/blob/master/features/image_features/xception_features.py) 

### Videos 
* [video_features](https://github.com/jim-schwoebel/allie/blob/master/features/video_features/video_features.py) - standard video feature array (default)
* [y8m_features](https://github.com/jim-schwoebel/allie/blob/master/features/video_features/y8m_features.py) 

### CSV 
* [csv_features](https://github.com/jim-schwoebel/allie/blob/master/features/csv_features/csv_features.py) - standard CSV feature array

## Not Implemented / Work in progress
- Reduce redundnacy; if already in schema, do not re-featurize (to reduce computational overhead again).
- Pass through featurization type as a command line argument 

### Audio
* allow Ludwig model type to dictate featurization - 'audio_ludwig_features'
* [Wavelet transforms](http://ataspinar.com/2018/12/21/a-guide-for-using-the-wavelet-transform-in-machine-learning/) - could be useful for dataset augmentation techniques.
* [fft python](https://stackoverflow.com/questions/23377665/python-scipy-fft-wav-files)
* [kaldi features](https://github.com/pykaldi/pykaldi)  - GMM and other such features. https://pykaldi.github.io/api/kaldi.feat.html#module-kaldi.feat.fbank
* [CountNet](https://github.com/faroit/CountNet) - number of speakers in a mixture (5 second interval). Combine with WebRTC VAD (https://github.com/wiseman/py-webrtcvad) to get featurization per segment like average lengths, etc. 

### Text
* allow Ludwig model type to dictate featurization (.JSON files --> .CSV images). - 'text_ludwig_features'
* follow up with Larry on semantic coherence vector / value 
* add in text transcription (as the default value) 
* input text files 
* BERT pre-trained model - https://github.com/huggingface/pytorch-pretrained-BERT

### Images 
* allow Ludwig model type to dictate featurization (.JSON files --> .CSV). - 'image_ludwig_features'
* Add in transcription to standard image array if settings.JSON image transcript == True.
* [Kornia](https://kornia.readthedocs.io/en/latest/color.html) - Harris feature detection - https://kornia.readthedocs.io/en/latest/color.html

### Videos 
* add in transcription to the standard video array {'transcript': video_transcript, 'type': video} if settings.JSON video transcript == True.
* [Age](https://github.com/deepinsight/insightface) - age/gender with video 
* [Near duplicate](https://github.com/Chinmay26/Near-Duplicate-Video-Detection)

### CSV 
* be able to determine file type and featurize accordingly on local path ./img.jpg ,./audio.wav, ./video.mp4, ./text.txt, etc.; these will then be featurized with default featurizers for images, audio, video, and text accordingly.
* accomodate all types on ludwig (https://uber.github.io/ludwig/getting_started/) - binary, numerical, category, set, bag, sequence, text, timeseries, image
* [seglearn](https://github.com/dmbee/seglearn) - time series pipeline.

