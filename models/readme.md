# Models

Here are a list of active models: model name, accuracy, etc. 

## Model schema 
```
data={'sample', sampletype,
      'features', features,
      'transcriptions': transcripts,
      'labels': labels,
      'models': models
      }
```

Models --> each sample can apply models with a feature embedding, etc.

```
models={'audio': audio_models,
        'text': text_models,
        'image': image_models,
        'video': video_models,
        'csv': csv_models
        }
```

Each model includes various information on all models including:

```
audio_models: []
```

each model includes

```
model_name: some1_some2_TPOT.pickle
feature_set: default_audio features
prediction: _____
prediction_time: ______
problemtype: 'regression' | 'classification'
metrics: {'MSE': ____
          'Accuracy': _____
         } 
```

## Settings
* if compress=True, only use compressed model (if available).

## Audio models
* NLX-model-gender --> Gender detection --> {feature array} --> 98% +/- 0.002. 
* Age 
* Stress

## Text models 
* TBA 

## Image models
* TBA (age and gender) 

## Video models
* TBA 

## CSV models 
* TBA 
