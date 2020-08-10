# Cleaning

![](https://github.com/jim-schwoebel/allie/blob/master/annotation/helpers/assets/clean.png)

This part of Allie's skills relates to data cleaning.

Data cleansing is the process of making clean datasets - like removing noise in audio files. It allows for data with a higher signal-to-noise ratio for modeling, increasing robustness of models.

## How to use cleaning scripts

To clean an entire folder of a certain file type (e.g. audio files of .WAV format), you can run:

```python3
cd /Users/jim/desktop/allie
cd cleaning/audio_cleaning
python3 clean.py /Users/jim/desktop/allie/train_dir/males/
python3 clean.py /Users/jim/desktop/allie/train_dir/females/
```

Click the .GIF below to follow along this example in a video format:

[![](https://github.com/jim-schwoebel/allie/blob/master/annotation/helpers/assets/clean.gif)](https://drive.google.com/file/d/1gqEHb_3WYFZNnBYdiwJZL--1Aw5KYLUR/view?usp=sharing)

The code above will featurize all the audio files in the folderpath via the default_cleaner specified in the settings.json file (e.g. 'clean_mono16hz'). 

## Extending to other file types

Note you can extend this to any of the file types. The table below overviews how you could call each as a augmenter. In the code below, you must be in the proper folder (e.g. ./allie/augmentation/audio_augmentations for audio files, ./allie/augmentation/image_augmentation for image files, etc.) for the scripts to work properly.

| Data type | Supported formats | Call to featurizer a folder | Current directory must be | 
| --------- |  --------- |  --------- | --------- | 
| audio files | .MP3 / .WAV | ```python3 clean.py [folderpath]``` | ./allie/cleaning/audio_cleaning | 
| text files | .TXT | ```python3 clean.py [folderpath]``` | ./allie/cleaning/text_cleaning| 
| image files | .PNG | ```python3 clean.py [folderpath]``` | ./allie/cleaning/image_cleaning | 
| video files | .MP4 | ```python3 clean.py [folderpath]``` |./allie/cleaning/video_cleaning| 
| csv files | .CSV | ```python3 clean.py [folderpath]``` | ./allie/cleaning/csv_cleaning | 

## Implemented

### Implemented for all file types 
* [delete_duplicates](https://github.com/jim-schwoebel/allie/blob/master/datasets/cleaning/delete_duplicates.py) - deletes duplicate files in the directory 
* [delete_json](https://github.com/jim-schwoebel/allie/blob/master/datasets/cleaning/delete_json.py) - deletes all .JSON files in the directory (this is to clean the featurizations) 

### [Audio](https://github.com/jim-schwoebel/allie/tree/master/cleaning/audio_cleaning)
* [clean_getfirst3secs]() - gets the first 3 seconds of the audio file
* [clean_keyword]() - keeps only keywords that are spoken based on a transcript (from the default_audio_transcriber)
* [clean_mono16hz]() - converts all audio to mono 16000 Hz for analysis (helps prepare for many preprocessing techniques)
* [clean_towav]() - converts all audio files to wav files
* [clean_multispeaker]() - deletes audio files from a dataset that have been identified as having multiple speakers from a deep learning model
* [clean_normalizevolume]() - normalizes the volume of all audio files using peak normalization methods from ffmpeg-normalize
* [clean_opus]() - converts an audio file to .OPUS audio file format then back to wav (a lossy conversion) - narrowing in more on voice signals over noise signals.
* [clean_random20secsplice]() - take a random splice (time specified in the script) from the audio file.
* [clean_removenoise]() - removes noise from the audio file using SoX program and noise floors.
* [clean_removesilence]() - removes silence from an audio file using voice activity detectors.
* [clean_rename]() - renames all the audio files in the current directory with a new UUID
* [clean_utterances]() - converts all audio files into unique utterances (1 .WAV file --> many .WAV file utterances) for futher analysis.

### [Text](https://github.com/jim-schwoebel/allie/tree/master/cleaning/text_cleaning)
* [clean_summary](https://github.com/jim-schwoebel/allie/blob/master/cleaning/text_cleaning/clean_summary.py) - extracts a 100 word summary of a long piece of text and deletes the original work (using [Text rank summarization](https://github.com/davidadamojr/TextRank))
* [clean_textacy](https://github.com/jim-schwoebel/allie/blob/master/cleaning/text_cleaning/clean_textacy.py) - removes punctuation and a variety of other operations to clean a text (uses [Textacy](https://chartbeat-labs.github.io/textacy/build/html/api_reference/text_processing.html))

### [Image](https://github.com/jim-schwoebel/allie/tree/master/cleaning/image_cleaning)
* [clean_extractfaces](https://github.com/jim-schwoebel/allie/blob/master/cleaning/image_cleaning/clean_extractfaces.py) - extract faces from an image
* [clean_greyscale](https://github.com/jim-schwoebel/allie/blob/master/cleaning/image_cleaning/clean_greyscale.py) - make all images greyscale 
* [clean_jpg2png](https://github.com/jim-schwoebel/allie/blob/master/cleaning/image_cleaning/clean_jpg2png.py) - make images from jpg to png to standardize image formats

### [Video](https://github.com/jim-schwoebel/allie/tree/master/cleaning/video_cleaning)
* [clean_alignfaces](https://github.com/jim-schwoebel/allie/blob/master/cleaning/video_cleaning/clean_alignfaces.py) - takes out faces from a video frame and keeps the video for an added label
* [clean_videostabilize](https://github.com/jim-schwoebel/allie/blob/master/cleaning/video_cleaning/clean_videostabilize.py) - stabilizes a video frame using [vidgear](https://github.com/abhiTronix/vidgear) (note this is a WIP)

### [CSV](https://github.com/jim-schwoebel/allie/tree/master/cleaning/csv_cleaning)
* [clean_csv](https://github.com/jim-schwoebel/allie/blob/master/cleaning/csv_cleaning/clean_csv.py) - uses [datacleaner](https://github.com/rhiever/datacleaner), a standard excel sheet cleaning script that imputes missing values and prepares CSV spreadsheets for machine learning

## [Settings](https://github.com/jim-schwoebel/allie/blob/master/settings.json)

Allie has multiple default settings for model training to help you start out with the framework. Here are some of the settings that relate to Allie's cleaning API. Settings can be modified in the [settings.json](https://github.com/jim-schwoebel/allie/blob/master/settings.json) file. 


| setting | description | default setting | all options | 
|------|------|------|------| 
| clean_data | whether or not to clean datasets during the model training process via default cleaning scripts. | False | True, False | 
| [default_audio_cleaners](https://github.com/jim-schwoebel/allie/tree/master/cleaning/audio_cleaning) | the default cleaning strategies used during audio modeling if clean_data == True | ["clean_mono16hz"] | ["clean_getfirst3secs", "clean_keyword", "clean_mono16hz", "clean_towav", "clean_multispeaker", "clean_normalizevolume", "clean_opus", "clean_randomsplice", "clean_removenoise", "clean_removesilence", "clean_rename", "clean_utterances"] |
| [default_csv_cleaners](https://github.com/jim-schwoebel/allie/tree/master/cleaning/csv_cleaning) | the default cleaning strategies used to clean .CSV file types as part of model training if clean_data==True | ["clean_csv"] | ["clean_csv"] | 
| [default_image_cleaners](https://github.com/jim-schwoebel/allie/tree/master/cleaning/image_cleaning) | the default cleaning techniques used for image data as a part of model training is clean_data == True| ["clean_greyscale"] |["clean_extractfaces", "clean_greyscale", "clean_jpg2png"] | 
| [default_text_cleaners](https://github.com/jim-schwoebel/allie/tree/master/cleaning/text_cleaning) | the default cleaning techniques used during model training on text data if clean_data == True| ["clean_textacy"] | ["clean_summary", "clean_textacy"]  | 
| [default_video_cleaners](https://github.com/jim-schwoebel/allie/tree/master/cleaning/video_cleaning) | the default cleaning strategies used for videos if clean_data == True | ["clean_alignfaces"] | ["clean_alignfaces", "clean_videostabilize"] | 
