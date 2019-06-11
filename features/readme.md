## Featurization scripts

This is a folder for extracting features from audio, text, image, video, or .CSV files. 

## Standard feature dictionary (.JSON)

Show outline for this below. This makes it flexible to any featurization and transcript type.

```
def make_features(sampletype):

	# only add labels when we have actual labels.
	features={'audio':dict(),
		  'text': dict(),
		  'image':dict(),
		  'video':dict(),
		  'csv': dict(),
		  }

	transcripts={'audio': dict(),
			 'text': dict(),
			 'image': dict(),
			 'video': dict(),
			 'csv': dict()}

	data={'sampletype': sampletype,
		  'transcripts': transcripts,
		  'features': features,
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
* [audioset_features]()
* [librosa_features]()
* [meta_features]()
* [mixed_features]() - random combinations of audio and text features (via ratios)
* [myprosody_features]() - sometimes unstable 
* [nltk_features]() - text feature array 
* [praat_features]()
* [pspeech_features]() 
* [pyaudio_features]()
* [sa_features]()
* [sox_features]()
* [standard_features]() - standard audio feature array (default)
* [spectrogram_features]() 
* [specimage_features]()
* [specimage2_features]()

### Text
* [fast_features]()
* [glove_features]() 
* [nltk_features]() - standard text feature array (default)
* [spacy_features]() 
* [w2vec_features]() 

### Images 
* [image_features]() - standard image feature array (default)
* [inception_features]() 	
* [resnet_features]()
* [tesseract_features]()	
* [vgg16_features]() 
* [vgg19_features]() 
* [xception_features]() 

### Videos 
* [video_features]() - standard video feature array (default)
* [y8m_features]() 

### CSV 
* [csv_features]() - standard CSV feature array

## Not Implemented / Work in progress

- Reduce redundnacy; if already in schema, do not re-featurize (to reduce computational overhead again).

### Audio
* allow Ludwig model type to dictate featurization (.JSON files --> .CSV).
* [kaldi features](https://github.com/pykaldi/pykaldi)  - GMM and other such features. https://pykaldi.github.io/api/kaldi.feat.html#module-kaldi.feat.fbank
* [CountNet](https://github.com/faroit/CountNet) - number of speakers in a mixture (5 second interval). Combine with WebRTC VAD (https://github.com/wiseman/py-webrtcvad) to get featurization per segment like average lengths, etc. 

### Text
* allow Ludwig model type to dictate featurization (.JSON files --> .CSV).
* add in text transcription (as the default value) 
* input text files 
* BERT pre-trained model - https://github.com/huggingface/pytorch-pretrained-BERT

### Images 
* allow Ludwig model type to dictate featurization (.JSON files --> .CSV).
* Add in transcription to standard image array if settings.JSON image transcript == True.
* [Kornia](https://kornia.readthedocs.io/en/latest/color.html) - Harris feature detection - https://kornia.readthedocs.io/en/latest/color.html

### Videos 
* allow Ludwig model type to dictate featurization (.JSON files --> .CSV)
* add in transcription to the standard video array {'transcript': video_transcript, 'type': video} if settings.JSON video transcript == True.
* [Age](https://github.com/deepinsight/insightface) - age/gender with video 
* [Near duplicate](https://github.com/Chinmay26/Near-Duplicate-Video-Detection)

### CSV 
* be able to determine file type and featurize accordingly on local path ./img.jpg ,./audio.wav, ./video.mp4, ./text.txt, etc.; these will then be featurized with default featurizers for images, audio, video, and text accordingly.
* accomodate all types on ludwig (https://uber.github.io/ludwig/getting_started/) - binary, numerical, category, set, bag, sequence, text, timeseries, image
* [seglearn](https://github.com/dmbee/seglearn) - time series pipeline.

