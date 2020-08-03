# Preprocessing scripts

This is a folder for manipulating and pre-processing features from audio, text, image, video, or .CSV files. 

This is done via a convention for transformers, which are in the proper folders (e.g. audio files --> audio_transformers). In this way, we can appropriately create transformers for various sample data types. 

These are added during the modeling process.

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

### Scaling data
https://scikit-learn.org/stable/modules/preprocessing.html - all preprocessing techniques 
--> default scaling to quantile transformer could work here 
--> visualization could use which feature selection strategy works best in terms of model accuracy

* scaler = preprocessing.StandardScaler().fit(X_train)
* quantile_transformer = preprocessing.QuantileTransformer(random_state=0)
* pt = preprocessing.PowerTransformer(method='box-cox', standardize=False)
* [X_normalized = preprocessing.normalize(X, norm='l2')]() - Normalization is the process of scaling individual samples to have unit norm.
* enc = preprocessing.OneHotEncoder(categories=[genders, locations, browsers]) - for turning into numbers
* binarizer = preprocessing.Binarizer().fit(X)  # fit does nothing - Feature binarization is the process of thresholding numerical features to get boolean values.
* poly = PolynomialFeatures(2) // poly.fit_transform(X)

### Feature selection 

* [scikit-learn](https://scikit-learn.org/stable/modules/preprocessing.html)
* RFE (https://www.scikit-yb.org/en/latest/_modules/yellowbrick/features/rfecv.html) 
* LASSO - 

### Dimensionality reduction techniques

![](https://github.com/jim-schwoebel/allie/blob/master/preprocessing/helpers/dimensionality_reduction.png)

* PCA - PCA - dimensionality reduction
* LDA - Linear discriminant analysis (LDA)
* tSNE - t-distributed stochastic neighbor embedding 
* pLDA - probabilistic LDA 
* Neural autoencoder 

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
