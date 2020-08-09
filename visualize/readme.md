## Visualization

![](https://github.com/jim-schwoebel/allie/raw/master/annotation/helpers/assets/model.png)

This is a univeral visualizer for all types of data as a part of model training. 

Note that [this is a setting](https://github.com/jim-schwoebel/allie/blob/master/settings.json) in the Allie Framework (e.g. "visualize_data": true). 

## Getting started
To get started, you first need to featurize some data using featurizations scripts. This data must be in the [train_dir folder](https://github.com/jim-schwoebel/allie/tree/master/train_dir) in the form of directories. To read more about featurization, [see this page](https://github.com/jim-schwoebel/allie/tree/master/features).

After you have featurized your data, go to this current folder (./allie/visualize') and run the visualize.py script:
```
python3 visualize.py [problemtype] [folder A] [folder B] ... [folder N]
```
Note you need to pass through the problem_type (e.g. 'audio'|'text'|'image'|'video'|'csv') and also all the relevant folders featurizations. In this case, we are looking at audio files separating males from females (e.g. featurizations exist) in the [train_dir folder](https://github.com/jim-schwoebel/allie/tree/master/train_dir).
```
python3 visualize.py audio males females 
```

This then generates a tree structure of graphs, for example below:

```
├── classes.png
├── clustering
│   ├── isomap.png
│   ├── lle.png
│   ├── mds.png
│   ├── modified.png
│   ├── pca.png
│   ├── spectral.png
│   ├── tsne.png
│   └── umap.png
├── feature_ranking
│   ├── feature_importance.png
│   ├── feature_plots
│   │   └── 128_mfcc_10_std.png
            ... [all feature plots (many files)]
│   ├── heatmap.png
│   ├── heatmap_clean.png
│   ├── lasso.png
│   ├── pearson.png
│   └── shapiro.png
└── model_selection
    ├── calibration.png
    ├── cluster_distance.png
    ├── elbow.png
    ├── ks.png
    ├── learning_curve.png
    ├── logr_percentile_plot.png
    ├── outliers.png
    ├── pca_explained_variance.png
    ├── precision-recall.png
    ├── prediction_error.png
    ├── residuals.png
    ├── roc_curve.png
    ├── roc_curve_train.png
    ├── siloutte.png
    └── thresholds.png
```

## Output graphs

Once you run this script, you output many visualizations. These visualizations can be customized within the script itself with some simple modifications. See below for some of the visualizations you can make.

Note that this script considers whether or not to balance datasets (e.g. "balance_data": true in settings.json) - so make sure you adjust your settings as to whether or not you'd like to balance the data before running the script above. These were the settings used to create the visualizations below:

```
{
  "version": "1.0.0",
  "augment_data": false,
  "balance_data": true,
  "clean_data": false,
  "create_YAML": true,
  "create_csv": true,
  "default_audio_features": [ "pspeech_features", "praat_features", "sox_features" ],
  "default_audio_transcriber": ["deepspeech_dict"],
  "default_csv_features": [ "csv_features" ],
  "default_csv_transcriber": ["raw text"],
  "default_dimensionality_reducer": [ "pca" ],
  "default_feature_selector": [ "lasso" ],
  "default_image_features": [ "image_features" ],
  "default_image_transcriber": ["tesseract"],
  "default_scaler": [ "standard_scaler" ],
  "default_text_features": [ "nltk_features" ],
  "default_text_transcriber": "raw text",
  "default_training_script": [ "tpot" ],
  "default_video_features": [ "video_features" ],
  "default_video_transcriber": [ "tesseract (averaged over frames)" ],
  "feature_number": 20,
  "model_compress": false,
  "reduce_dimensions": false,
  "scale_features": true,
  "select_features": false,
  "test_size": 0.10,
  "transcribe_audio": false,
  "transcribe_csv": true,
  "transcribe_image": true,
  "transcribe_text": true,
  "transcribe_video": true,
  "visualize_data": true
}
```
### numbers in each class
![](https://github.com/jim-schwoebel/allie/blob/master/visualize/data/classes.png)

## Clustering
Quickly iterate and see which cluster method works best with your dataset.
```
├── clustering
│   ├── isomap.png
│   ├── lle.png
│   ├── mds.png
│   ├── modified.png
│   ├── pca.png
│   ├── spectral.png
│   ├── tsne.png
│   └── umap.png
```

### Isomap embedding
![](https://github.com/jim-schwoebel/allie/blob/master/visualize/data/clustering/isomap.png)
### LLE embedding
![](https://github.com/jim-schwoebel/allie/blob/master/visualize/data/clustering/lle.png)
### MDS embedding
![](https://github.com/jim-schwoebel/allie/blob/master/visualize/data/clustering/mds.png)
### Modified embedding
![](https://github.com/jim-schwoebel/allie/blob/master/visualize/data/clustering/modified.png)
### PCA embedding
![](https://github.com/jim-schwoebel/allie/blob/master/visualize/data/clustering/pca.png)
### Spectral embedding
![](https://github.com/jim-schwoebel/allie/blob/master/visualize/data/clustering/spectral.png)
### tSNE embedding
![](https://github.com/jim-schwoebel/allie/blob/master/visualize/data/clustering/tsne.png)
### UMAP embedding
![](https://github.com/jim-schwoebel/allie/blob/master/visualize/data/clustering/umap.png)

## Feature ranking

```
├── feature_ranking
│   ├── feature_importance.png
│   ├── feature_plots
│   │   └── 128_mfcc_10_std.png
            ... [all feature plots (many files)]
│   ├── heatmap.png
│   ├── heatmap_clean.png
│   ├── lasso.png
│   ├── pearson.png
│   └── shapiro.png
```

### Feature importances (top 20 features)
![](https://github.com/jim-schwoebel/allie/blob/master/visualize/data/feature_ranking/feature_importance2.png)

### Feature_plots
Easily plots all the features via violin plots (to spot distributions).

![](https://github.com/jim-schwoebel/allie/blob/master/visualize/data/feature_ranking/feature_plots/326_meanF0.png)

### Lasso plot 
![](https://github.com/jim-schwoebel/allie/blob/master/visualize/data/feature_ranking/lasso2.png)

### Heatmaps
Heatmap with correlated variables
![](https://github.com/jim-schwoebel/allie/blob/master/visualize/data/feature_ranking/heatmap2.png)

Heatmap with removed correlated variables 
![](https://github.com/jim-schwoebel/allie/blob/master/visualize/data/feature_ranking/heatmap_clean2.png)

### Pearson ranking plot
![](https://github.com/jim-schwoebel/allie/blob/master/visualize/data/feature_ranking/pearson2.png)

### Shapiro plot 
![](https://github.com/jim-schwoebel/allie/blob/master/visualize/data/feature_ranking/shapiro2.png)

## Modeling graphs

```
└── model_selection
    ├── calibration.png
    ├── cluster_distance.png
    ├── elbow.png
    ├── ks.png
    ├── learning_curve.png
    ├── logr_percentile_plot.png
    ├── outliers.png
    ├── pca_explained_variance.png
    ├── precision-recall.png
    ├── prediction_error.png
    ├── residuals.png
    ├── roc_curve.png
    ├── roc_curve_train.png
    ├── siloutte.png
    └── thresholds.png
```

### Calibration plot
![](https://github.com/jim-schwoebel/allie/blob/master/visualize/data/modeling/calibration.png)

### Cluster distance 
![](https://github.com/jim-schwoebel/allie/blob/master/visualize/data/modeling/cluster_distance.png)

### Elbow plot
![](https://github.com/jim-schwoebel/allie/blob/master/visualize/data/modeling/elbow.png)

### KS stat plot
![](https://github.com/jim-schwoebel/allie/blob/master/visualize/data/modeling/ks.png)

### Learning curve
![](https://github.com/jim-schwoebel/allie/blob/master/visualize/data/modeling/learning_curve.png)

### logr percentile plot
![](https://github.com/jim-schwoebel/allie/blob/master/visualize/data/modeling/logr_percentile_plot.png)

### Outlier detection 
![](https://github.com/jim-schwoebel/allie/blob/master/visualize/data/modeling/outliers.png)

### PCA explained variance plot
![](https://github.com/jim-schwoebel/allie/blob/master/visualize/data/modeling/pca_explained_variance.png)

### Precision/recall graphs 
![](https://github.com/jim-schwoebel/allie/blob/master/visualize/data/modeling/precision-recall.png)

### Prediction error graphs
![](https://github.com/jim-schwoebel/allie/blob/master/visualize/data/modeling/prediction_error.png)

### Residuals 
![](https://github.com/jim-schwoebel/allie/blob/master/visualize/data/modeling/residuals.png)

### ROC curve_train
![](https://github.com/jim-schwoebel/allie/blob/master/visualize/data/modeling/roc_curve_train.png)

### ROC curve_test
![](https://github.com/jim-schwoebel/allie/blob/master/visualize/data/modeling/roc_curve.png)

### siloutte graph
![](https://github.com/jim-schwoebel/allie/blob/master/visualize/data/modeling/siloutte.png)

### Threshold graph 
![](https://github.com/jim-schwoebel/allie/blob/master/visualize/data/modeling/thresholds.png)
