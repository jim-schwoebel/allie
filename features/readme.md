## Featurization scripts

This is a folder for extracting features from audio, text, image, video, or .CSV files. 

## Standard feature dictionary (.JSON)

Show outline for this below. This makes it flexible to any featurization and transcript type.

Note that there can be audio transcripts, image transcripts, and video transcripts. The image and video transcripts use OCR to characterize text in the image, whereas audio transcripts are transcipts done by traditional speech-to-text systems (e.g. Pocketsphinx). The schema above allows for a flexible definition for transcripts that can accomodate all forms. 

## Implemented 

Note that all scripts implemented have features and their corresponding labels. It is important to provide labels to understand what the features correspond to. It's also to keep in mind the relative speeds of featurization to optimize server costs (they are provided here for reference).

### Audio
* [audioset_features]()
* [librosa_features]()
* [meta_features]()
* [praat_features]()
* [pyaudio_features]()
* [sa_features]()
* [sox_features]()
* [standard_features]() - standard audio feature array (default)
* [spectrogram_features]() 

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
### Audio
* Add in transcription to standard audio array if settings.JSON audio transcript == True; customize transcription types.
* [Pause features](https://github.com/jim-schwoebel/pauses) - retrain a classifier on Common Voice Dataset (100 files).
* [Python-speech-features](https://github.com/jameslyons/python_speech_features) - another feature extraction library.
* [Parselmouth image features](https://github.com/YannickJadoul/Parselmouth) - fundamental frequency estimation.
* [DeepFormants](https://github.com/MLSpeech/DeepFormants) - formant frequency extraction.
* [Pyworld](https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder) - PyWorldVocoder - A Python wrapper for World Vocoder, fundamental frequency determination.
* [Auditok](https://github.com/amsehili/auditok) - for audio event detection.
* [Noise separation](https://github.com/seanwood/gcc-nmf) - noise separation technique.
* [Make noisy](https://github.com/Sato-Kunihiko/audio-SNR/) - noisy add-on.

### Text
* BERT pre-trained model 

### Images 
* Add in transcription to standard image array if settings.JSON image transcript == True.

### Videos 
* [Age](https://github.com/deepinsight/insightface) - age/gender with video 
* add in transcription to the standard video array {'transcript': video_transcript, 'type': video} if settings.JSON video transcript == True.


