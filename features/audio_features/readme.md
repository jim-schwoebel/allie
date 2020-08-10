## How to use audio feature API

```
cd allie/features/audio_features
python3 featurize.py [folder] [featuretype]
```

### Audio
* [audioset_features](https://github.com/jim-schwoebel/allie/blob/master/features/audio_features/audioset_features.py) - 
simple script to extract features using the VGGish model released by Google.
* [audiotext_features](https://github.com/jim-schwoebel/allie/blob/master/features/audio_features/audiotext_features.py) - Featurizes data with text feautures extracted from the transcript.
* [librosa_features](https://github.com/jim-schwoebel/allie/blob/master/features/audio_features/librosa_features.py) - 
extracts acoustic features using the [LibROSA library](https://librosa.org/).
* [loudness_features](https://github.com/jim-schwoebel/allie/blob/master/features/audio_features/loudness_features.py) - extracts loudness features using the [pyloudnorm](https://github.com/csteinmetz1/pyloudnorm) library.
* [meta_features](https://github.com/jim-schwoebel/allie/blob/master/features/audio_features/meta_features.py) - extracts meta features from models trained on the audioset dataset.
* [mixed_features](https://github.com/jim-schwoebel/allie/blob/master/features/audio_features/mixed_features.py) - random combinations of audio and text features (via ratios).
* [multispeaker_features](https://github.com/jim-schwoebel/allie/blob/master/features/audio_features/multispeaker_features.py) - detect number of speakers in audio files using the [CountNet](https://github.com/faroit/CountNet) deep learning model.
* [opensmile_features](https://github.com/jim-schwoebel/allie/blob/master/features/audio_features/opensmile_features.py) - 14 embeddings with [OpenSMILE](https://www.audeering.com/opensmile/) possible here; defaults to GeMAPSv01a.conf.
* [praat_features](https://github.com/jim-schwoebel/allie/blob/master/features/audio_features/praat_features.py) - extracts features from the [parselmouth.praat library](https://pypi.org/project/praat-parselmouth/).
* [prosody_features](https://github.com/jim-schwoebel/allie/blob/master/features/audio_features/prosody_features.py) - prosody using Google's VAD - including pause length, total number of pauses, and pause variability.
* [pspeech_features](https://github.com/jim-schwoebel/allie/blob/master/features/audio_features/pspeech_features.py) - extracts features with the [python_speech features library](https://github.com/jameslyons/python_speech_features).
* [pyaudio_features](https://github.com/jim-schwoebel/allie/blob/master/features/audio_features/pyaudio_features.py) - extract features withh the [pyaudioanalysis](https://github.com/tyiannak/pyAudioAnalysis) library.
* [pyaudiolex_features](https://github.com/tyiannak/pyAudioAnalysis) - time series features extracted with the [pyaudioanalysis](https://github.com/tyiannak/pyAudioAnalysis) library.
* [pyworld_features](https://github.com/jim-schwoebel/allie/blob/master/features/audio_features/pyworld_features.py) - f0 and and spectrogram features.
* [sa_features](https://github.com/jim-schwoebel/allie/blob/master/features/audio_features/sa_features.py) - some additional features extracted using the [SignalAnalysis](https://brookemosby.github.io/Signal_Analysis/Signal_Analysis.features.html#module-Signal_Analysis.features.signal) library.
* [sox_features](https://github.com/jim-schwoebel/allie/blob/master/features/audio_features/sox_features.py) - features extracted from the [sox](http://sox.sourceforge.net/sox.html) command line interface.
* [speechmetrics_features](https://github.com/aliutkus/speechmetrics) - extracts features that estimate speech quality without references using the [speechmetrics](https://github.com/aliutkus/speechmetrics) library.
* [specimage_features](https://github.com/jim-schwoebel/allie/blob/master/features/audio_features/specimage_features.py) - image-based features from spectrograms.
* [specimage2_features](https://github.com/jim-schwoebel/allie/blob/master/features/audio_features/specimage2_features.py) - image-based features from spectrograms (alternative).
* [spectrogram_features](https://github.com/jim-schwoebel/allie/blob/master/features/audio_features/spectrogram_features.py) - spectrogram-based features.
* [standard_features](https://github.com/jim-schwoebel/allie/blob/master/features/audio_features/standard_features.py) - standard audio feature array (default).

### Settings

These are some default settings and possible settings for Allie's audio featurization API:

| setting | description | default setting | all options | 
|------|------|------|------| 
| [default_audio_features](https://github.com/jim-schwoebel/voice_modeling/tree/master/features/audio_features) | default set of audio features used for featurization (list). | ["standard_features"] | ["audioset_features", "audiotext_features", "librosa_features", "meta_features", "mixed_features", "opensmile_features", "praat_features", "prosody_features", "pspeech_features", "pyaudio_features", "pyaudiolex_features", "sa_features", "sox_features", "specimage_features", "specimage2_features", "spectrogram_features", "speechmetrics_features", "standard_features"] | 
| default_audio_transcriber | the default transcription model used during audio featurization if trainscribe_audio == True | ["deepspeech_dict"] | ["pocketsphinx", "deepspeech_nodict", "deepspeech_dict", "google", "wit", "azure", "bing", "houndify", "ibm"] | 
| transcribe_audio | a setting to define whether or not to transcribe audio files during featurization and model training via the default_audio_transcriber | True | True, False | 
