## Featurization scripts

This is a folder for extracting features from audio, text, image, video, or .CSV files. This is done via a convention for featurizers, which are in the proper folders (e.g. audio files --> audio_features). In this way, we can appropriately create featurizers for various sample data types. 

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
* [blabla](https://github.com/novoic/blabla) - NLP library for clinical assessment
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

.CSV can include numerical data, categorical data, audio files (./audio.wav), image files (.png), video files (./video.mp4), text files ('.txt' or text column), or other .CSV files. This scope of a table feature is inspired by [D3M schema design proposed by the MIT data lab](https://github.com/mitll/d3m-schema/blob/master/documentation/datasetSchema.md).

* [csv_features](https://github.com/jim-schwoebel/allie/blob/master/features/csv_features/csv_features.py) - standard CSV feature array

## Not Implemented / Work in progress

### Audio
looking into actively
* [pysepm](https://github.com/schmiph2/pysepm) - speech quality measures
* [pystoi](https://github.com/mpariente/pystoi) - speech intelligibility measure 
* [pb_bss](https://github.com/fgnt/pb_bss/blob/master/examples/mixture_model_example.ipynb) - blind source separation (training models)
* [Kaldi](https://pykaldi.github.io/api/kaldi.feat.html#)
* [Pysptk](https://github.com/r9y9/pysptk) - Tokoyo based lab
* [speechpy](https://github.com/astorfi/speechpy)
* [sigfeat](https://github.com/SiggiGue/sigfeat)
* [kapre-kears](https://github.com/keunwoochoi/kapre)
* [torchaudio-contrib](https://github.com/keunwoochoi/torchaudio-contrib)
* [sonopy](https://github.com/MycroftAI/sonopy) - MFCCs fastest featurizer
* [spafe](https://github.com/SuperKogito/spafe) - many features
* [Essentia](https://github.com/kushagrasurana/Essentia-feature-extraction)
* [Wav2Letter](https://github.com/facebookresearch/wav2letter/wiki/Python-bindings) 
* [Formant extraction](https://github.com/danilobellini/audiolazy/blob/master/examples/formants.py)
* [speaker diarization speakers]() - assumes long-form audio files

tried but not user friendly
* [Gammatone](https://github.com/detly/gammatone) - spectrograms of fixed lengths
* [surfboard](https://github.com/novoic/surfboard) - GPL
* [Shennong](https://github.com/bootphon/shennong) - using kaldi / post-processing
* Ludwig audio features - add them in.
* [pyspk](https://nbviewer.jupyter.org/github/r9y9/pysptk/blob/master/examples/pysptk%20introduction.ipynb) - Fundamental frequency estimation using pyspk (bottom)
* [auDeep](https://github.com/auDeep/auDeep)
* fix myprosody_features.py feature script (in helpers for now). This is buggy and may change into the future as the library is more supported by more senior developers.
* allow Ludwig model type to dictate featurization - 'audio_ludwig_features'
* [Wavelet transforms](http://ataspinar.com/2018/12/21/a-guide-for-using-the-wavelet-transform-in-machine-learning/) - could be useful for dataset augmentation techniques.
* [fft python](https://stackoverflow.com/questions/23377665/python-scipy-fft-wav-files)
* [kaldi features](https://github.com/pykaldi/pykaldi)  - GMM and other such features. https://pykaldi.github.io/api/kaldi.feat.html#module-kaldi.feat.fbank
* [CountNet](https://github.com/faroit/CountNet) - number of speakers in a mixture (5 second interval). Combine with WebRTC VAD (https://github.com/wiseman/py-webrtcvad) to get featurization per segment like average lengths, etc. 
* [pyroomacoustics](https://github.com/LCAV/pyroomacoustics)
* [resin](https://github.com/kylerbrown/resin) - copyleft license

### Text
* [bigrams-trigrams](https://www.sttmedia.com/syllablefrequency-english#trigrams) - add in bigrams/trigrams
* [keras-bert](https://github.com/CyberZHG/keras-bert)
* [Flair](https://github.com/flairNLP/flair)
* [textacy](https://chartbeat-labs.github.io/textacy/build/html/index.html) - from references
* [jellyfish](https://github.com/jamesturk/jellyfish) - distance measurements 
* [text2image](https://github.com/mansimov/text2image) - create an image from a sentence
* [stylemetry](https://github.com/jpotts18/stylometry) - many features around lexical richness
* [Swivel](https://github.com/bigiceberg/models/tree/master/swivel) - Google team
* [summerize](https://github.com/junaidiiith/TextSummerizer/blob/master/summerizer.py) - misspelled but many text features
* BERT pre-trained model - add through Ludwig
* Allen NLP pre-trained models.
* [coreNLP](https://stanfordnlp.github.io/stanfordnlp/) - stanfordNLP parser and featurizer
* [pytorch-transformers](https://github.com/huggingface/pytorch-transformers) - a lot of pre-trained models that could be useful for featurizations (e.g. BERT and other model types here).
* allow Ludwig model type to dictate featurization (.JSON files --> .CSV images). - 'text_ludwig_features'
* follow up with Larry on semantic coherence vector / value 
* add in text transcription (as the default value) 
* input text files 

### Images 
* [kornia](https://github.com/kornia/kornia)
* SURF, ORB, BRIEF, AKAZE - corner detectors - SIFT is most accurate but also the slowest (patented), BRIEF is the fastest but least accurate
* allow Ludwig model type to dictate featurization (.JSON files --> .CSV). - 'image_ludwig_features'
* Yolo-9000 classes - https://github.com/philipperemy/yolo-9000
* Add in transcription to standard image array if settings.JSON image transcript == True.
* [Kornia](https://kornia.readthedocs.io/en/latest/color.html) - Harris feature detection - https://kornia.readthedocs.io/en/latest/color.html

### Videos 
* add in transcription to the standard video array {'transcript': video_transcript, 'type': video} if settings.JSON video transcript == True.
* [video_features_0](https://github.com/antoine77340/video_feature_extractor) - general-purpose video feature extractor
* [video_features_1](https://github.com/zo7/deep-features-video/blob/master/extract_features.py)
* [video_features_2](https://github.com/jonasrothfuss/videofeatures)
* [Age](https://github.com/deepinsight/insightface) - age/gender with video 
* [Near duplicate](https://github.com/Chinmay26/Near-Duplicate-Video-Detection)
* [pliers](https://github.com/tyarkoni/pliers) - multiple feature extractors / video
* [video2data](https://github.com/KristopherKubicki/video2data) - sophisticated video pipeline
* [video face extraction](https://github.com/ChangLabUcsf/face_extraction)

### CSV 

Perhaps rename CSV to tabular data type instead? 

* be able to determine file type and featurize accordingly on local path ./img.jpg ,./audio.wav, ./video.mp4, ./text.txt, etc.; these will then be featurized with default featurizers for images, audio, video, and text accordingly.
* accomodate all types on ludwig (https://uber.github.io/ludwig/getting_started/) - binary, numerical, category, set, bag, sequence, text, timeseries, image
* [seglearn](https://github.com/dmbee/seglearn) - time series pipeline.

### Time series 

TBA soon 

### Geospatial data 

TBA soon - how to work with cartography and maps.
