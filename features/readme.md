## Featurization scripts

This is a folder for extracting features from audio, text, image, video, or .CSV files. 

## Implemented 
### Audio
* [Librosa]()

### Text
* [NLTK]()
* [SpaCy]()

### Images 
* [OpenCV]()

### Videos 
* [Scikit-Video]()

## Not Implemented / Work in progress
### Audio
* [Pause features](https://github.com/jim-schwoebel/pauses) - retrain a classifier on Common Voice Dataset (100 files).
* [Python-speech-features](https://github.com/jameslyons/python_speech_features) - another feature extraction library.
* [Parselmouth image features](https://github.com/YannickJadoul/Parselmouth) - fundamental frequency estimation.
* [DeepFormants](https://github.com/MLSpeech/DeepFormants) - formant frequency extraction.
* [Pyworld](https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder) - PyWorldVocoder - A Python wrapper for World Vocoder, fundamental frequency determination.
* [Auditok](https://github.com/amsehili/auditok) - for audio event detection
* [Create-noisy](https://github.com/Sato-Kunihiko/audio-SNR/blob/master/create_noisy_minumum_code.py)
* [Noise separation](https://github.com/seanwood/gcc-nmf)

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
