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

The code above will transform all the featurized text files (in .JSON files, folder ONE and folder TWO) via a classification script with a common name ONETWO. For clarity, the command line arguments are further elaborated upon below along with all possible options to help you use the transformers API. Note that folder ONE and folder TWO are assumed to be in the [train_dir folder](https://github.com/jim-schwoebel/allie/tree/master/train_dir).

| setting | sample | description | all options | 
|------|------|------|------| 
| sys.argv[2] | 'text' | the sample type of file preprocessed by the transformer | ['audio', 'text', 'image', 'video', 'csv'] | 
| sys.argv[3] | 'c' | classification or regression problems | ['c','r'] | 
| sys.argv[4] | 'onetwo' | the common name for the modeling file | can be any string | 
| sys.argv[5], sys.argv[6], sys.argv[n] | 'one' | classes that you seek to model in the [train_dir folder](https://github.com/jim-schwoebel/allie/tree/master/train_dir) | any string folder name |

## Settings

Here are the relevant settings in Allie related to preprocessing that you can change in the [settings.json file](https://github.com/jim-schwoebel/allie/blob/master/settings.json) (along with the default settings).

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
