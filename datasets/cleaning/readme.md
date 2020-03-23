# Cleaning

This part of Allie's skills relates to data cleaning.

Data cleansing is the process of making clean datasets - like removing noise in audio files. It allows for data with a higher signal-to-noise ratio for modeling, increasing robustness of models.

Common data cleaning things include
- class auto balance (this is done by default in the model.py script)
- removing file duplicates 
- normalizing volume in the audio files
- trimming silence out of voice files 

## How to use cleaning scripts

You can call from the command line fairly easily by doing something like the format:

```
python3 clean.py --clean_directory --file_directory
```

Where clean_directory is this directory path and file_directory is the path of the folder that you'd like to clean the data.

Here is a quick example:

```
python3 clean.py /Users/jimschwoebel/allie/datasets/cleaning /Users/jimschwoebel/allie/train_dir/one
```

Once you run a script like this, you will receive a terminal output like:
```
-----------------------------
       REMOVING SILENCE      
-----------------------------
Jims-MBP:cleaning jimschwoebel$ python3 clean.py /Users/jimschwoebel/allie/datasets/cleaning /Users/jimschwoebel/allie/train_dir/two
audiofolder detected!
-----------------
-----------------------------
   DELETING DUPLICATE FILES  
-----------------------------
deleted the files below
[]
/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Using TensorFlow backend.
-----------------------------
       REMOVING SILENCE      
-----------------------------
```

Note that the audio file type ('audio','text','image','video','csv') is automatically detected in the folder and there are some universal data cleaning scripts (e.g. removing file duplicates) that can be used for pre-processing.

## Supported data cleaning scripts 

### All file types 
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
* [Deep Audio Prior](https://github.com/adobe/Deep-Audio-Prior) - can separate 2 noises without any training data

### Text
* [Text rank/summarization](https://github.com/davidadamojr/TextRank) - 100 word summary, Number of keywords extracted is relative to the size of the text (a third of the number of nodes in the graph)
* [Textacy](https://chartbeat-labs.github.io/textacy/build/html/api_reference/text_processing.html) - preprocessing like removing punctuation and repeats, etc.
