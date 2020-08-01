# Cleaning

This part of Allie's skills relates to data cleaning.

Data cleansing is the process of making clean datasets - like removing noise in audio files. It allows for data with a higher signal-to-noise ratio for modeling, increasing robustness of models.

Common data cleaning things include
- class auto balance (this is done by default in the model.py script)
- removing file duplicates 
- normalizing volume in the audio files
- trimming silence out of voice files 

## How to use cleaning scripts

To clean an entire folder of a certain file type (e.g. audio files of .WAV format), you can run:

```
cd ~ 
cd allie/cleaning/audio_cleaning
python3 cleaning.py /Users/jimschwoebel/allie/load_dir
```

The code above will featurize all the audio files in the folderpath via the default_augmenter specified in the settings.json file (e.g. 'augment_tasug'). 

Note you can extend this to any of the augmentation types. The table below overviews how you could call each as a augmenter. In the code below, you must be in the proper folder (e.g. ./allie/augmentation/audio_augmentations for audio files, ./allie/augmentation/image_augmentation for image files, etc.) for the scripts to work properly.

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

### [Audio](https://github.com/jim-schwoebel/allie/tree/master/datasets/cleaning/audio)
* [delete_multi_speaker](https://github.com/jim-schwoebel/allie/blob/master/datasets/cleaning/audio/delete_multi_speaker.py) - deletes audio file if more than 1 speaker (optimizing for one-way monologues). Disabled by default because you may want to model audio events.
* [remove_silence](https://github.com/jim-schwoebel/allie/blob/master/datasets/cleaning/audio/remove_silence.py) - removes silence from audio files 
* [normalize_volume](https://github.com/jim-schwoebel/allie/blob/master/datasets/cleaning/audio/normalize_volume.py) - normalizes the volume of all audio files (at end)
* [clean audio files](https://github.com/meokz/looking-to-listen) - background noise removal 

### Text
* Coming soon.

### Image 
* Coming soon.

### Video
* Coming soon.

### CSV
* Coming soon.


## Future

### Audio 
* [RNN Noise XIPH](https://github.com/xiph/rnnoise) - eliminates all noise events from environment (e.g. typing)
* Diarization?? - can implement this here if necessary 
* [PB_BSS](https://github.com/fgnt/pb_bss)
* [Norbert](https://github.com/sigsep/norbert) - Weiner filter for source separation of audio signals (multichannel). Use a source truth and separate into Channel A and Channel B.
* [Deep Audio Prior](https://github.com/adobe/Deep-Audio-Prior) - can separate 2 noises without any training data
* [Microsoft noise dataset](https://github.com/microsoft/MS-SNSD) - MS-SNSD
* extract loudest section - https://github.com/petewarden/extract_loudest_section

### Text
* [Text rank/summarization](https://github.com/davidadamojr/TextRank) or [textrank](https://github.com/summanlp/textrank)- 100 word summary, Number of keywords extracted is relative to the size of the text (a third of the number of nodes in the graph) - implented alpha 
* [Textacy](https://chartbeat-labs.github.io/textacy/build/html/api_reference/text_processing.html) - preprocessing like removing punctuation and repeats, etc. - implemented this with multiple settings

### Video
* [Video stabilization](https://github.com/abhiTronix/vidgear#camgear) - video stabilization for the camera / cleaning it
* [Near-duplicate video detection](https://github.com/Chinmay26/Near-Duplicate-Video-Detection)

### CSV
* [Autonormalize](https://github.com/FeatureLabs/autonormalize)
