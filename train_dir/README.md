## Train directory 

Use this directory to train machine learning models based on folders of files.

## Getting started 

To get started, you just need to make at least 2 folders containing the same type of file (e.g. audio .WAV files).
| folder 1 | folder 2 | 
| ------- | ------- | 
| ![](https://github.com/jim-schwoebel/allie/blob/master/training/helpers/train_1.png) | ![](https://github.com/jim-schwoebel/allie/blob/master/training/helpers/train_2.png)  | 

Now, you just need to run model.py and continue on to train a machine learning model. Additional instructions can be found [here](https://github.com/jim-schwoebel/allie/tree/master/training).

Note you can edit the settings.json to change the default featurizer for model training. It is important to train using standard arrays if you plan to put models into production environments, as our database only takes in standard_features for audio files. 

## Supported file formats

Here are the supported file formats for the load directory. 

| File type | extension | recommended format | 
| ------------- |-------------| -------------| 
| audio file | .WAV, .MP3 | .WAV | 
| text file | .TXT | .TXT | 
| image file | .PNG, .JPG | .PNG | 
| video file | .MP4 | .MP4 | 
| CSV file | .CSV | .CSV | 



