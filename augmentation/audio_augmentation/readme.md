## Getting started

Here is a way to quickly augment a folder of audio files:
```
cd ~ 
cd allie/features/audio_augmentation
python3 augment.py /Users/jimschwoebel/allie/load_dir
```

## Implemented
* [augment_tsaug](https://github.com/jim-schwoebel/allie/blob/master/augmentation/audio_augmentation/augment_tsaug.py) - adds noise and various shifts to audio files, addes 2x more data; see tutorial [here](https://tsaug.readthedocs.io/en/stable/).
* [augment_addnoise](https://github.com/jim-schwoebel/allie/blob/master/augmentation/audio_augmentation/augment_addnoise.py) - adds noise to an audio file.
* [augment_noise](https://github.com/jim-schwoebel/allie/blob/master/augmentation/audio_augmentation/augment_noise.py) - removes noise from audio files randomly.
* [augment_pitch](https://github.com/jim-schwoebel/allie/blob/master/augmentation/audio_augmentation/augment_pitch.py) - shifts pitch up and down to correct for gender differences. 
* [augment_randomsplice](https://github.com/jim-schwoebel/allie/blob/master/augmentation/audio_augmentation/augment_randomsplice.py) - randomly splice an audio file to generate more data.
* [augment_silence](https://github.com/jim-schwoebel/allie/blob/master/augmentation/audio_augmentation/augment_silence.py) - add silence to an audio file to augment a dataset.
* [augment_time](https://github.com/jim-schwoebel/allie/blob/master/augmentation/audio_augmentation/augment_time.py) - change time duration for a variety of audio files through making new files.
* [augment_volume](https://github.com/jim-schwoebel/allie/blob/master/augmentation/audio_augmentation/augment_volume.py) - change volume randomly (helps to mitigate effects of microphohne distance on a model).

## Settings
| setting | description | default setting | all options | 
|------|------|------|------| 
| augment_data | whether or not to implement data augmentation policies during the model training process via default augmentation scripts. | True | True, False |
| [default_audio_augmenters](https://github.com/jim-schwoebel/allie/tree/master/augmentation/audio_augmentation) | the default augmentation strategies used during audio modeling if augment_data == True | ["augment_tsaug"] | ["augment_tsaug", "augment_addnoise", "augment_noise", "augment_pitch", "augment_randomsplice", "augment_silence", "augment_time", "augment_volume"] | 
