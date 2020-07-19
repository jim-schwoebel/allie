# audio_augmentation
Augment audio datasets to prepare for machine learning purposes. This library creates 10 addition files for each audio file (.WAV) in a folder to augment the dataset during machine learning training.

![](https://media.giphy.com/media/141f3JuM2hDdhm/giphy.gif)

Note that augmentation techniques may not work great for datasets >1000 in size. These may be better for smaller datasets that may not have enough data samples (across a range of speakers and microphone configurations). Or, datasets with high degrees of bias (e.g. 80% male / 20% female). 

## getting started

To get started, clone the repository (assume you have homebrew and are on mac):
```
git clone --recurse-submodules -j8 git@github.com:jim-schwoebel/audio_augmentation.git
cd audio_augmentation
python3 setup.py
```

Now all you need to do is run 
```
python3 audio_augment.py [folder]
python3 audio_augment.py /Users/jimschwoebel/audio_augmentation/test
```

This will add 10 augmented files to each file in the folder. This assumes the folders are filled with .WAV files, so you may need to do some pre-processsing to get files to .WAV format.

## What is augmented / benefits 

There are 10 manipulations per audio file, generating 10 additional audio files in the folder. Assuming we had 1 audio file (1.wav), there would be 10 new files created: 

* 1_trimmed.wav - trim silence from the file (corrects for long periods of silence).
* 1_decrease_0.33 - decreases volume by 33% (corrects for age or gender differences).
* 1_freq_1.wav - decrease frequencies by 1/2 octave (corrects for gender differences).
* 1_freq_2.wav - increases frequencies by 1/2 octave (corrects for gender differences)
* 1_increase_3.wav - increase volume by 3x (corrects for age differences)
* 1_noise_remove.wav - remove noise using sox noise floor to focus on signal (corrects if noise in background)
* 1_opus.wav - filter via .OPUS codec for voice range (corrects for noise)
* 1_peak_normalized.wav - normalize volume by peak power (corrects for low volume speakers and various microphones)
* 1_stretch_0.wav - make audio file 1.5x faster (corrects for changes in pace)
* 1_stretch_2.wav - make audio file 1.5x slower (corrects for changes in pace)
* 1_addednoise_1.wav - make audio file have noise (more robust in noisey environment)
* 1_addednoise_2.wav - make audio file have noise (more robust in noisy environment)

These augmentations correct for many things - including gender, ages, noise environments, and changes in speaking rates. So, you'd expect that in augmenting datasets they would make machine learning models more robust.

The con of doing this approach is that you'll likely need 10x the hard disk space / more CPU time to train machine learning models. Feel free to change the script here to delete some if you think they are superfluous.

## preparing for machine learning

Can make many folders here to test accuracies of the various techniques for augmentation. 

## building pipelines

Can call any of the functions here in order to create some random augmentations.

--> trimmed.decrease.freqshift.increase.remove_noise().

## tests

Using this methodology, we have seen increases in machine learning accuracies:
* classifying female_african vs. female african controls (non-african accents) - ~92.9% accuracy (+0.4% accuracy)

## some testing 
* age detection
* gender detection

## Additional things to add (in future)

This repository is in active development; we're always creating new ways to augment audio files. Here are some things we plan on adding into the future:

* [additive noise](https://github.com/keunwoochoi/kapre) - in keras.
* [scraper](https://scaper.readthedocs.io/en/latest/api.html) - soundscape synthesis (can generate SNR and other events to add to the mixture of audio files randomly)
* [Normal distribution on means](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.normal.html) - add normal distribution capability for volume change and noise level and then pick a random point on here for noise threshold (https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.normal.html). 
* [Noise separation](https://github.com/seanwood/gcc-nmf) - noise separation technique.
* [Make noisy](https://github.com/Sato-Kunihiko/audio-SNR/) - noisy add-on.
* [VAD](https://github.com/wiseman/py-webrtcvad) - VAD cleaning.
* [GAN - Cycla GAN](https://github.com/leimao/Voice_Converter_CycleGAN) - convert speech from one form to another. 
* [Relativistic_GAN](https://github.com/deepakbaby/se_relativisticgan) - adverserial speech enhancer.
* [DNN enhance](https://github.com/eesungkim/Speech_Enhancement_DNN_NMF) - Speech enhancement via DNN_NMF
* [SEGAN](https://github.com/leftthomas/SEGAN) - speech enhancement.
* add this in as enhancer (https://github.com/yongxuUSTC/sednn)
* speech enhancement via logMMSE (https://github.com/braindead/logmmse)
* adverserial model using noise (http://www.openslr.org/28/)
* [Python-audio-effects](https://github.com/carlthome/python-audio-effects) - for augmentation pipelines. https://github.com/carlthome/python-audio-effects
* augmentation methods using markov chains (probability of current and next state!!) 
* make improvements to pitch and speed (randomly select a number from 1 to 100 and scale accordingly) - this will make the data more robust - pitch changes. 
* add in noise from urban sound dataset and other datasets (generated adversarially) at various volumes (ffmpeg) - improve this!! - make volumes randomly up/down 
* make a random splice (e.g. 5 seconds) for additional data. 
* add in random pauses into the speech (silence files).
* changes in speaktype (e.g. reading from passage vs. extemporaneous speech) - perhaps could do this through pitch variability.

To have
* 1_spliced_1.wav - random splice (reduces effect of phoneme order)
* 1_spliced_2.wav - random splice (reduces effect of phoneme order)
* 1_paused_1.wav - random 100 ms pauses inserted into wav file (reduces effect of speaking types)
* 1_paused_2.wav - random 100 ms pauses inserted into wav file (reduces effect of speaking types)

Other stuff 

* text (for transcripts), image (video), and video capabilities for audio augmentation. This will allow for more robust learning on videos. 

If you have any other ideas, please reach out.

## additional libraries
* [sednn](https://github.com/yongxuUSTC/sednn)
* [ffmpeg](https://ffmpeg.org/)
* [sox](http://sox.sourceforge.net/)
* [ffmpeg-normalize](https://github.com/slhck/ffmpeg-normalize)
* [librosa](https://github.com/librosa/librosa)
* [opus-tools](https://opus-codec.org/downloads/)
* [Python-audio-effects](https://github.com/carlthome/python-audio-effects) - for augmentation pipelines. https://github.com/carlthome/python-audio-effects

