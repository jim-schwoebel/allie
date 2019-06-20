# Downloads 

Allie can also download data for you to prepare datasets for machine learning.

## Getting started 

All you need to do is call Allie's download script:
```
cd allie/datasets/downloads
python3 download.py 
```

You will then be prompted through a few screens as to what kinds of data you are seeking to download. 

```
what dataset would you like to download? (1-audio, 2-text, 3-image, 4-video, 5-csv)
--> 1

found 35 datasets...
----------------------------
here are the available AUDIO datasets
----------------------------
TIMIT dataset
Parkinson's speech dataset
ISOLET Data Set
AudioSet
Multimodal EmotionLines Dataset (MELD)
Free Spoken Digit Dataset
Speech Accent Archive
2000 HUB5 English
Emotional Voice dataset - Nature
LJ Speech
VoxForge
Million Song Dataset
Free Music Archive
Common Voice
Spoken Commands dataset
Bird audio detection challenge
Environmental audio dataset
Urban Sound Dataset
Ted-LIUM
Noisy Dataset
Librispeech
Emotional Voices Database
CMU Wilderness
Arabic Speech Corpus
Flickr Audio Caption
CHIME
Tatoeba
Freesound dataset
Spoken Wikipeida Corpora
Karoldvl-ESC
Zero Resource Speech Challenge
Speech Commands Dataset
Persian Consonant Vowel Combination (PCVC) Speech Dataset
VoxCeleb

what audio dataset would you like to download?
--> voxceleb
found dataset: VoxCeleb
just confirming, do you want to download the VoxCeleb dataset? (Y - yes, N - no) 
Y
```

Right now this just opens up links to the datasets for you to manually download (the .JSON files were assembled from the README. However, in the future data could be downloaded in the ./allie/datasets/downloads/data folder. 

## References

For a complete list of datasets, visit the main datasets page. Note all these and more are included in the download.py script. Some datasets you may need to sign up for research access, so they cannot be directly downloaded within this interface; however, we may have them intenrally within the NeuroLex team. Just ask us :-) 
