## How to use audio feature API

```
cd allie/features/audio_features
python3 featurize.py [folder] [featuretype]
```

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

### Settings

These are some default settings and possible settings for Allie's audio featurization API:

| setting | description | default setting | all options | 
|------|------|------|------| 
| [default_audio_features](https://github.com/jim-schwoebel/voice_modeling/tree/master/features/audio_features) | default set of audio features used for featurization (list). | ["standard_features"] | ["audioset_features", "audiotext_features", "librosa_features", "meta_features", "mixed_features", "opensmile_features", "praat_features", "prosody_features", "pspeech_features", "pyaudio_features", "pyaudiolex_features", "sa_features", "sox_features", "specimage_features", "specimage2_features", "spectrogram_features", "speechmetrics_features", "standard_features"] | 
| default_audio_transcriber | the default transcription model used during audio featurization if trainscribe_audio == True | ["deepspeech_dict"] | ["pocketsphinx", "deepspeech_nodict", "deepspeech_dict", "google", "wit", "azure", "bing", "houndify", "ibm"] | 
| transcribe_audio | a setting to define whether or not to transcribe audio files during featurization and model training via the default_audio_transcriber | True | True, False | 
