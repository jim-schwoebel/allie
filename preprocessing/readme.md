# Preprocessing scripts

![](https://github.com/jim-schwoebel/allie/blob/master/annotation/helpers/assets/model.png)

This is a folder for manipulating and pre-processing [features extracted](https://github.com/jim-schwoebel/allie/tree/master/features) from audio, text, image, video, or .CSV files as part of the machine learning modeling process. 

This is done via a convention for transformers, which are in the proper folders (e.g. audio files --> audio_transformers). There are three main feature transformation techniques: [feature scaling](https://github.com/jim-schwoebel/allie/blob/master/preprocessing/feature_scale.py), [feature selection](https://github.com/jim-schwoebel/allie/blob/master/preprocessing/feature_select.py), and [dimensionality reduction](https://github.com/jim-schwoebel/allie/blob/master/preprocessing/feature_reduce.py).

In this way, we can appropriately create transformers for various sample data types. 

## How to transform folders of featurized files

To transform an entire folder of a featurized files, you can run:

```
cd ~ 
cd allie/preprocessing/audio_transformers
python3 transform.py text c onetwo one two
```

The code above will transform all the featurized text files (in .JSON files, folder ONE and folder TWO) via a classification script with a common name ONETWO. 

## Settings

Note you can extend this to any of the feature types. The table below overviews how you could call each as a featurizer. In the code below, you must be in the proper folder (e.g. ./allie/features/audio_features for audio files, ./allie/features/image_features for image files, etc.) for the scripts to work properly.

| Data type | Settings.json | Options | Call to featurizer a folder | Current directory must be | 
| --------- |  --------- |  --------- | --------- | --------- | 
| scaling data | scale_data: True | default_scalers: ["binarizer", "one_hot_encoder", "normalize", "power_transformer", "poly", "quantile_transformer", "standard_scaler" ] |  ```python3 feature_scale.py [folderpath] [options]``` | ./allie/preprocessing | 
| feature selection | select_features: True | default_feature_selectors: ["rfe", "lasso", "chi", "kbest", "variance"] | ```python3 feature_select.py [folderpath] [options]``` | ./allie/preprocessing | 
| dimensionality reduction | reduce_dimensions: True | default_dimensionality_reduction: ["pca", "lda", "tsne", "plda","autoencoder"] | ```python3 feature_reduce.py [folderpath] [options]``` | ./allie/preprocessing  | 

## Definitions

### Preproprocessing parameters adaptable in settings.json
| setting | description | default setting | all options | 
|------|------|------|------| 
| reduce_dimensions | if True, reduce dimensions via the default_dimensionality_reducer (or set of dimensionality reducers) | False | True, False |
| default_dimensionality_reducer | the default dimensionality reducer or set of dimensionality reducers | ["pca"] | ["pca", "lda", "tsne", "plda","autoencoder"] | 
| select_features | if True, select features via the default_feature_selector (or set of feature selectors) | False | True, False | 
| default_feature_selector | the default feature selector or set of reature selectors | ["lasso"] | ["lasso", "rfe", "chi", "kbest", "variance"] | 
| scale_features | if True, scales features via the default_scaler (or set of scalers) | False | True, False | 
| default_scaler | the default scaler (e.g. StandardScalar) to pre-process data | ["standard_scaler"] | ["binarizer", "one_hot_encoder", "normalize", "power_transformer", "poly", "quantile_transformer", "standard_scaler"]|

## References
* [sophia](https://github.com/jiankaiwang/sophia) - tutorials in feature preprocessing and ML algorithms
