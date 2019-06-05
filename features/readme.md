## Featurization scripts

This is a folder for extracting features from audio, text, image, video, or .CSV files. 

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
* [standard_features]() - standard audio feature array 
* [spectrogram_features]() - 

### Text
* [fast_features]()
* [glove_features]() 
* [nltk_features]() - standard text feature array 
* [spacy_features]() 
* [w2vec_features]() 

### Images 
* [image_features]() - standard image feature array
* [inception_features]() 	
* [resnet_features]()
* [tesseract_features]()	
* [vgg16_features]() 
* [vgg19_features]() 
* [xception_features]() 

### Videos 
* [video_features]() - standard video array 

## Not Implemented / Work in progress
### Audio
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
* ??

### Videos 
* [Age](https://github.com/deepinsight/insightface) - age/gender with video 

### transcripts
* change transcript array to have audio_transcript, image_transcript, and video_transcript
* all of these transcript types are different
* image transcript = pytesseract, video transcript = pytesseract + all images cut at interval, audio transcript = transcription engine.
