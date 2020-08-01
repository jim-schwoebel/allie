## Augmentation 

This part of Allie's skills relates to data augmentation.

Data augmentation is used to expand the training dataset in order to improve the performance and ability of a machine learning model to generalize. For example, you may want to shift, flip, brightness, and zoom on images to augment datasets to make models perform better in noisy environments indicative of real-world use. Data augmentation is especially useful when you don't have that much data, as it can greatly expand the amount of training data that you have for machine learning. 

Typical augmentation scheme is to take 50% of the data and augment it and leave the rest the same. This is what they did in Tacotron2 architecture. 

You can read more about data augmentation [here](https://towardsdatascience.com/1000x-faster-data-augmentation-b91bafee896c).

OLD SUBMODULES
```
[submodule "datasets/augmentation/eda_nlp"]
	path = datasets/augmentation/eda_nlp
	url = https://github.com/jasonwei20/eda_nlp
[submodule "datasets/augmentation/imgaug"]
	path = datasets/augmentation/imgaug
	url = https://github.com/aleju/imgaug
[submodule "datasets/augmentation/vidaug"]
	path = datasets/augmentation/vidaug
	url = https://github.com/okankop/vidaug
[submodule "datasets/PyDataset"]
	path = datasets/PyDataset
	url = https://github.com/iamaziz/PyDataset
[submodule "datasets/youtube_scrape"]
	path = datasets/youtube_scrape
	url = https://github.com/jim-schwoebel/youtube_scrape
[submodule "training/keras_compressor"]
	path = training/keras_compressor
	url = https://github.com/DwangoMediaVillage/keras_compressor
[submodule "training/scikit-small-ensemble"]
	path = training/scikit-small-ensemble
	url = https://github.com/stewartpark/scikit-small-ensemble
[submodule "datasets/labeling/sound_event_detection"]
	path = datasets/labeling/sound_event_detection
	url = https://github.com/jim-schwoebel/sound_event_detection
[submodule "datasets/labeling/labelImg"]
	path = datasets/labeling/labelImg
	url = https://github.com/tzutalin/labelImg
[submodule "datasets/labeling/labelme"]
	path = datasets/labeling/labelme
	url = https://github.com/wkentaro/labelme
[submodule "datasets/augmentation/audio_augmentation"]
	path = datasets/augmentation/audio_augmentation
	url = https://github.com/jim-schwoebel/audio_augmentation
```

## Types of augmentation

There are two main types of data augmentation:

* adding data to data (e.g. YouTube videos to iPhone videos) 
* generating new data (through manipulations) 

The vast majority of approaches focus on generating new data because it's often more convenient and useful for modeling purposes. It's much harder to make two datasets look like each other, as there exists variation in the datasets themselves which may confuse machine learning models (or lead to overfitting to one dataset over another). 

To read more about data augmentation stratgies, please review [this page](https://github.com/AgaMiko/data-augmentation-review).

## Augmentation scripts 

To get started, all you need to do is:
```
augment.py [foldername]
```

The augmentation then will auto-detect the file type of the folder and augment the data appropriately for machine learning purposes.

Note that it augment_data=True in settings.json, this script will automatically run in the model.py script.

## Implemented 
### Audio
* [augment_tsaug](https://tsaug.readthedocs.io/en/stable/) - see tutorial here.
* [augment_addnoise]()
* [augment_noise]()
* [augment_pitch]()
* [augment_randomsplice]()
* [augment_silence]()
* [augment_time]()
* [augment_volume]()

### Text
* [EDA_NLP]()

### Image
* [IMGAUG]()

### Video
* [VIDAUG]()

### CSV
* [TGAN](https://github.com/sdv-dev/TGAN) - generative adverserial examples - can be done on continuous audio data as well

## Work-in-progress (WIP)

Audio 
* [ESC-10 augmentation](https://github.com/JasonZhang156/Sound-Recognition-Tutorial/blob/master/data_augmentation.py) - script
* [adding noise with ffmpeg[(https://stackoverflow.com/questions/15792105/simulating-tv-noise)
* [Audio data augmentation](https://github.com/sid0710/audio_data_augmentation)
* [Keras-AudioDataGenerator](https://github.com/AhmedImtiazPrio/Keras-AudioDataGenerator)'
* [Audio degrader](https://github.com/emilio-molina/audio_degrader)
* [Spectral cluster](https://github.com/wq2012/SpectralCluster) - spectral cluster 
* [spec_augment](https://github.com/zcaceres/spec_augment) - from Google research / uses images 
* [audiomentations](https://github.com/iver56/audiomentations) - various transformations (can randomize noise)
* [audio dataset augmenter](https://github.com/kleydon/Audio-Dataset-Augmenter)
* [audio preprocessing](https://github.com/dedkoster/audio_preproccesing)
* [extract loudest section](https://github.com/petewarden/extract_loudest_section)

Text
* [Augmentation with Textacy](https://chartbeat-labs.github.io/textacy/build/html/api_reference/augmentation.html)

Image 
* [Augmentor](https://github.com/mdbloice/Augmentor)

Video
* TBA

CSV 
* [AutoAugment (Google)](https://github.com/tensorflow/models/tree/master/research/autoaugment) - paper [here](https://arxiv.org/abs/1805.09501) and implementation [here](https://github.com/DeepVoltaire/AutoAugment)
* [Population Based Augmentation (PBA)](https://github.com/arcelien/pba) - paper [here](https://arxiv.org/abs/1711.09846)


## References
* [Review of data augmentation strategies](https://towardsdatascience.com/1000x-faster-data-augmentation-b91bafee896c)
* [1000x Faster Data Augmentation](https://towardsdatascience.com/1000x-faster-data-augmentation-b91bafee896c)

