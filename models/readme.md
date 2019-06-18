# Models

Use this section of the repository to make model predctions in the ./load_dir.

## Getting started
```python3
cd voice_modeling/models
python3 load_model.py
```

This will then load the directory and apply all models in the directory according to sample types (audio_models, text_models, image_models, video_models, csv_models).

## Model schema 
```python3
data={'sample', sampletype,
      'features', features,
      'transcriptions': transcripts,
      'labels': labels,
      'models': models}
```

Models --> each sample can apply models with a feature embedding, etc.

```python3
models={'audio': audio_models,
        'text': text_models,
        'image': image_models,
        'video': video_models,
        'csv': csv_models}
```

## Settings
N/A

## Audio models
N/A 

## Text models 
N/A

## Image models
N/A

## Video models
N/A

## CSV models 
N/A

## References
* [AudioSet models](https://github.com/jim-schwoebel/audioset_models) - 📊 Easily apply 527 machine learning models trained on AudioSet.
