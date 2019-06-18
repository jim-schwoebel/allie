## Train directory 

Use this directory to train machine learning models based on folders of files.

## Getting started 

To get started, you just need to make at least 2 folders containing the same type of file (e.g. audio .WAV files). Ideally, these folders will have the same number of classes; otherwise, the classes will automatically balance to the lower number of classes during model training.

![](https://github.com/jim-schwoebel/allie/blob/master/training/helpers/train_1.png) 

![](https://github.com/jim-schwoebel/allie/blob/master/training/helpers/train_2.png)  

Now you need to run model.py:

```
cd allie/training
python3 model.py 
```

You then will be asked a few questions regarding the training process (in terms of data type, number of classes, and the name of the model). Note that --> indicates typed responses. 

```
what problem are you solving? (1-audio, 2-text, 3-image, 4-video, 5-csv)
--> 1

 OK cool, we got you modeling audio files 

how many classes would you like to model? (2 available) 
--> 2
these are the available classes: 
['one', 'two']
what is class #1 
--> one
what is class #2 
--> two
what is the 1-word common name for the problem you are working on? (e.g. gender for male/female classification) 
--> test
is this a classification (c) or regression (r) problem? 
--> c
```

Additional instructions can be found [here](https://github.com/jim-schwoebel/allie/tree/master/training).

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



