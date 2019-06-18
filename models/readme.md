# Models

Use this section of the repository to make model predctions in the ./load_dir.

## Getting started
```python3
cd voice_modeling/models
python3 load_model.py
```

This will then load the directory and apply all models in the directory according to sample types (audio_models, text_models, image_models, video_models, csv_models).

## Model schema 

Recall that the standard sample schema is as follows:

```python3
data={'sample', sampletype,
      'features', features,
      'transcriptions': transcripts,
      'labels': labels,
      'models': models}
```

The models part of this schema is as follows:

```python3
models={'audio': audio_models,
        'text': text_models,
        'image': image_models,
        'video': video_models,
        'csv': csv_models}
```

This allows for a flexible definition of models as arrays. As features are put into the load_dir, featurized, and then modeled, they can be updated with classes in the array such that new labels can be tagged onto the data samples.

## Audio models
N/A - no audio models trained yet. 

## Text models 
N/A - no text models trained yet. 

## Image models
N/A - no image models trained yet. 

## Video models
N/A - no video models trained yet. 

## CSV models 
N/A - no .CSV models trained yet. 

## References
* [AudioSet models](https://github.com/jim-schwoebel/audioset_models) - 📊 Easily apply 527 machine learning models trained on AudioSet.
