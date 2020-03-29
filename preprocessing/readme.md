# Preprocessing scripts

This is a folder for manipulating and pre-processing features from audio, text, image, video, or .CSV files. 

This is done via a convention for transformers, which are in the proper folders (e.g. audio files --> audio_transformers). In this way, we can appropriately create transformers for various sample data types. 

These are added during the modeling process.

## Scaling data

* scaler = preprocessing.StandardScaler().fit(X_train)
* quantile_transformer = preprocessing.QuantileTransformer(random_state=0)
* pt = preprocessing.PowerTransformer(method='box-cox', standardize=False)
* [X_normalized = preprocessing.normalize(X, norm='l2')]() - Normalization is the process of scaling individual samples to have unit norm.
* enc = preprocessing.OneHotEncoder(categories=[genders, locations, browsers]) - for turning into numbers
* binarizer = preprocessing.Binarizer().fit(X)  # fit does nothing - Feature binarization is the process of thresholding numerical features to get boolean values.
* poly = PolynomialFeatures(2) // poly.fit_transform(X)

## Feature selection 

* [scikit-learn](https://scikit-learn.org/stable/modules/preprocessing.html)
* RFE (https://www.scikit-yb.org/en/latest/_modules/yellowbrick/features/rfecv.html) 
* LASSO - 

## Dimensionality reduction techniques

[https://github.com/jim-schwoebel/allie/blob/master/preprocessing/helpers/dimensionality_reduction.png]()

* PCA - PCA - dimensionality reduction
* LDA - Linear discriminant analysis (LDA)
* tSNE - t-distributed stochastic neighbor embedding 
* pLDA - probabilistic LDA 
* Neural autoencoder 

## How to transform folders of featurized files

To transform an entire folder of a featurized files, you can run:

```
cd ~ 
cd allie/preprocessing/audio_transformers
python3 transform.py /Users/jimschwoebel/allie/train_dir/classA
```

The code above will transform all the featurized audio files (.JSON) in the folderpath via the default_transformer specified in the settings.json file (e.g. 'standard_transformer'). 

If you'd like to use a different transformer you can specify it optionally:

```
cd ~ 
cd allie/features/audio_transformer
python3 featurize.py /Users/jimschwoebel/allie/load_dir standard_scalar
```

Note you can extend this to any of the feature types. The table below overviews how you could call each as a featurizer. In the code below, you must be in the proper folder (e.g. ./allie/features/audio_features for audio files, ./allie/features/image_features for image files, etc.) for the scripts to work properly.

| Data type | Settings.json | Options | Call to featurizer a folder | Current directory must be | 
| --------- |  --------- |  --------- | --------- | --------- | 
| scaling data | scale_data: true | default_scalers: ['standard_scaler','quantile_transformer', 'power_transformer','one_hot_encoder','binarizer','poly'] |  ```python3 feature_scale.py [folderpath] [options]``` | ./allie/preprocessing | 
| feature selection | select_features: true | default_feature_selectors: ['rfe', 'lasso'] | ```python3 feature_select.py [folderpath] [options]``` | ./allie/preprocessing | 
| dimensionality reduction | reduce_dimensions: true | default_dimensionality_reduction: ['PCA', 'LDA', 'tSNE', 'pLDA','autoencoder'] | ```python3 feature_reduce.py [folderpath] [options]``` | ./allie/preprocessing  | 

