# voice_modeling
Master repository for modeling voice files. Transformed from NLX-model.

1. Get dataset (assess bias).
2. feature selection (https://machinelearningmastery.com/feature-selection-machine-learning-python/). 
3. modeling - SC, TPOT, Keras, Ludwig.
4. visualize models (Yellowbrick) - feature selection / etc. 
5. compress models for production.
6. server deployment. 

## getting started 
```
git clone
cd voice_modeling 
python3 setup.py 
```

## settings 

settings.json

```
default_audio_features=audio_features
default_text_features=text_features
default_image_features=image_features
default_video_features=video_features
transcribe_audio=True 
default_audio_transcriber=pocketsphinx
transcribe_images=True 
transcribe_videos=True
default_training_script=keras
augment_data=True 
visualize_data=True 
create_YAML=True 
model_compress=True
```

Typical augmentation scheme is to take 50% of the data and augment it and leave the rest the same. This is what they did in Tacotron2 architecture. 

## types of data

Load folders and data script type based on principal type of file.

* Audio --> .WAV / .MP3 --> .WAV 
* Text --> .WAV (transcribes) / .TXT / .PPT / .DOCX --> .TXT
* Images --> .PNG / .JPG --> .PNG 
* Video --> .MP4 / .M4A --> .MP4 
* CSV --> .CSV --> loads categorical and numerical data into .JSON 

This aligns well with how we define 'samples' in our pipeline.

## Features to add
### Bias discovery
* apply machine learning models for each types of data (audio, text, images, video, .CSV) to auto detect things like ages, genders, etc. 

### Datasets
* [PyDataset](https://github.com/iamaziz/PyDataset) - numerical datasets
* [AudioSet_download]() - download link to AudioSet 
* [CommonVoice]() - download link to Common Voice 
* [Training datasets - .JSON]() - accent detection, etc. 

### Visualization
* [Yellowbrick](https://www.scikit-yb.org/en/latest/)

### Modeling 
* TPOT
* AutoKeras
* Luwdig  

### Model compression
* [Model compression papers](https://github.com/sun254/awesome-model-compression-and-acceleration)
* [Keras compression](https://github.com/DwangoMediaVillage/keras_compressor) - should be good for Keras.
* [Scikit-small-compression](https://github.com/stewartpark/scikit-small-ensemble) - should be good for TPOT I believe.
