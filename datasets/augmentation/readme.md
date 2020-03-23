## Augmentation 

This part of Allie's skills relates to data augmentation.

Data augmentation is used to expand the training dataset in order to improve the performance and ability of a machine learning model to generalize. For example, you may want to shift, flip, brightness, and zoom on images to augment datasets to make models perform better in noisy environments indicative of real-world use. Data augmentation is especially useful when you don't have that much data, as it can greatly expand the amount of training data that you have for machine learning. 

Typical augmentation scheme is to take 50% of the data and augment it and leave the rest the same. This is what they did in Tacotron2 architecture. 

You can read more about data augmentation [here](https://towardsdatascience.com/1000x-faster-data-augmentation-b91bafee896c).

## Types of augmentation

There are two main types of data augmentation:

* adding data to data (e.g. YouTube videos to iPhone videos) 
* generating new data (through manipulations) 

The vast majority of approaches focus on generating new data because it's often more convenient and useful for modeling purposes. It's much harder to make two datasets look like each other, as there exists variation in the datasets themselves which may confuse machine learning models (or lead to overfitting to one dataset over another). 

## Augmentation scripts 

To get started, all you need to do is:
```
augment.py [foldername]
```

The augmentation then will auto-detect the file type of the folder and augment the data appropriately for machine learning purposes.

Note that it augment_data=True in settings.json, this script will automatically run in the model.py script.

## Folder structures 

Here is a brief description of what the folders in this section of the repository are for:

- [audio_augmentation](https://github.com/jim-schwoebel/audio_augmentation/tree/a1b7838063684f451fbbacfc23311bbf8ca38897) - for augmenting audio files
- [eda_nlp]() - for augmenting text files (not implemented yet)
- [imgaug]() - for augmenting images (not implemented yet)
- [vidaug]() - for augmenting video files (not implemented yet)

## Work-in-progress (WIP)
General 
* [AutoAugment (Google)](https://github.com/tensorflow/models/tree/master/research/autoaugment) - paper [here](https://arxiv.org/abs/1805.09501) and implementation [here](https://github.com/DeepVoltaire/AutoAugment)
* [Population Based Augmentation (PBA)](https://github.com/arcelien/pba) - paper [here](https://arxiv.org/abs/1711.09846)

Audio 
* [Audio data augmentation](https://github.com/sid0710/audio_data_augmentation)
* [Keras-AudioDataGenerator](https://github.com/AhmedImtiazPrio/Keras-AudioDataGenerator)
* [Spectral cluster](https://github.com/wq2012/SpectralCluster) - spectral cluster 
* [spec_augment](https://github.com/zcaceres/spec_augment) - from Google research / uses images 
* [audiomentations](https://github.com/iver56/audiomentations) - various transformations (can randomize noise)

Text
* [EDA_NLP]()

Image 
* [IMGAUG]()
* [Augmentor](https://github.com/mdbloice/Augmentor)

Video
* [VIDAUG]()

CSV 
* [TGAN](https://github.com/sdv-dev/TGAN)

## References
* [1000x Faster Data Augmentation](https://towardsdatascience.com/1000x-faster-data-augmentation-b91bafee896c)

