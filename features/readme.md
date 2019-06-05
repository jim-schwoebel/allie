## Featurization scripts

This is a folder for extracting features from audio, text, image, video, or .CSV files. 

## Work in progress
### Audio
* [Pauses](https://github.com/jim-schwoebel/pauses)
* [DeepFormants](https://github.com/MLSpeech/DeepFormants) - formant frequency extraction.
* [Pyworld](https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder) - PyWorldVocoder - A Python wrapper for World Vocoder, fundamental frequency determination.
* [Parselmouth](https://parselmouth.readthedocs.io/en/latest/examples/psychopy_experiments.html) - Praat features 
* [Genderless](https://github.com/drfeinberg/genderless) - Praat features that are not affected by changing genders.
* [Noise separation](https://github.com/seanwood/gcc-nmf)
* [Python-audio-effects](https://github.com/carlthome/python-audio-effects)
* [Librosa]()
* [Auditok](https://github.com/amsehili/auditok) - for audio event detection
* [Age](https://github.com/deepinsight/insightface) - age/gender with video 
* [Create-noisy](https://github.com/Sato-Kunihiko/audio-SNR/blob/master/create_noisy_minumum_code.py)

### transcripts
* change transcript array to have audio_transcript, image_transcript, and video_transcript
* all of these transcript types are different
* image transcript = pytesseract, video transcript = pytesseract + all images cut at interval, audio transcript = transcription engine.
