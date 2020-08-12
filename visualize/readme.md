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
│   ├── correlation.png
│   ├── data.csv
│   ├── feature_importance.png
│   ├── feature_plots
│   │   ├── 0_F0semitoneFrom27.5Hz_sma3nz_amean.png
│   │   ├── 10_loudness_sma3_amean.png
│   │   ├── 11_loudness_sma3_stddevNorm.png
│   │   ├── 12_loudness_sma3_percentile20.0.png
│   │   ├── 13_loudness_sma3_percentile50.0.png
│   │   ├── 14_loudness_sma3_percentile80.0.png
│   │   ├── 15_loudness_sma3_pctlrange0-2.png
│   │   ├── 16_loudness_sma3_meanRisingSlope.png
│   │   ├── 17_loudness_sma3_stddevRisingSlope.png
│   │   ├── 18_loudness_sma3_meanFallingSlope.png
│   │   ├── 19_loudness_sma3_stddevFallingSlope.png
│   │   ├── 1_F0semitoneFrom27.5Hz_sma3nz_stddevNorm.png
│   │   ├── 2_F0semitoneFrom27.5Hz_sma3nz_percentile20.0.png
│   │   ├── 3_F0semitoneFrom27.5Hz_sma3nz_percentile50.0.png
│   │   ├── 4_F0semitoneFrom27.5Hz_sma3nz_percentile80.0.png
│   │   ├── 5_F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2.png
│   │   ├── 6_F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope.png
│   │   ├── 7_F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope.png
│   │   ├── 8_F0semitoneFrom27.5Hz_sma3nz_meanFallingSlope.png
│   │   ├── 9_F0semitoneFrom27.5Hz_sma3nz_stddevFallingSlope.png
│   │   └── feature_plots_transformed
│   │       ├── 0_F0semitoneFrom27.5Hz_sma3nz_amean.png
│   │       ├── 10_loudness_sma3_amean.png
│   │       ├── 11_loudness_sma3_stddevNorm.png
│   │       ├── 12_loudness_sma3_percentile20.0.png
│   │       ├── 13_loudness_sma3_percentile50.0.png
│   │       ├── 14_loudness_sma3_percentile80.0.png
│   │       ├── 15_loudness_sma3_pctlrange0-2.png
│   │       ├── 16_loudness_sma3_meanRisingSlope.png
│   │       ├── 17_loudness_sma3_stddevRisingSlope.png
│   │       ├── 18_loudness_sma3_meanFallingSlope.png
│   │       ├── 19_loudness_sma3_stddevFallingSlope.png
│   │       ├── 1_F0semitoneFrom27.5Hz_sma3nz_stddevNorm.png
│   │       ├── 2_F0semitoneFrom27.5Hz_sma3nz_percentile20.0.png
│   │       ├── 3_F0semitoneFrom27.5Hz_sma3nz_percentile50.0.png
│   │       ├── 4_F0semitoneFrom27.5Hz_sma3nz_percentile80.0.png
│   │       ├── 5_F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2.png
│   │       ├── 6_F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope.png
│   │       ├── 7_F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope.png
│   │       ├── 8_F0semitoneFrom27.5Hz_sma3nz_meanFallingSlope.png
│   │       └── 9_F0semitoneFrom27.5Hz_sma3nz_stddevFallingSlope.png
│   ├── heatmap.png
│   ├── heatmap_clean.png
│   ├── lasso.png
│   ├── pearson.png
│   ├── radviz.png
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
  "default_audio_features": [ "opensmile_features" ],
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
### Numbers in each class
![](https://github.com/jim-schwoebel/allie/blob/master/visualize/data/classes.png)

You can also view the [raw data](https://github.com/jim-schwoebel/allie/blob/master/visualize/data/feature_ranking/data3.csv) in the visualization session.

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
|   └── umap.png
```

### Isomap embedding
![](https://github.com/jim-schwoebel/allie/blob/master/visualize/data/clustering/isomap3.png)
### LLE embedding
![](https://github.com/jim-schwoebel/allie/blob/master/visualize/data/clustering/lle3.png)
### MDS embedding
![](https://github.com/jim-schwoebel/allie/blob/master/visualize/data/clustering/mds3.png)
### Modified embedding
![](https://github.com/jim-schwoebel/allie/blob/master/visualize/data/clustering/modified3.png)
### PCA embedding
![](https://github.com/jim-schwoebel/allie/blob/master/visualize/data/clustering/pca3.png)
### Spectral embedding
![](https://github.com/jim-schwoebel/allie/blob/master/visualize/data/clustering/spectral3.png)
### tSNE embedding
![](https://github.com/jim-schwoebel/allie/blob/master/visualize/data/clustering/tsne3.png)
### UMAP embedding
![](https://github.com/jim-schwoebel/allie/blob/master/visualize/data/clustering/umap3.png)

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
![](https://github.com/jim-schwoebel/allie/blob/master/visualize/data/feature_ranking/feature_importance3.png)

### Feature_plots
Easily plots the top 20 features via violin plots (to spot distributions).
![](https://github.com/jim-schwoebel/allie/blob/master/visualize/data/feature_ranking/feature_plots/0_F0semitoneFrom27.5Hz_sma3nz_amean.png)

### Lasso plot 
![](https://github.com/jim-schwoebel/allie/blob/master/visualize/data/feature_ranking/lasso3.png)

### Heatmaps
Heatmap with correlated variables
![](https://github.com/jim-schwoebel/allie/blob/master/visualize/data/feature_ranking/heatmap3.png)

Heatmap with removed correlated variables 
![](https://github.com/jim-schwoebel/allie/blob/master/visualize/data/feature_ranking/heatmap_clean3.png)

### Pearson ranking plot
![](https://github.com/jim-schwoebel/allie/blob/master/visualize/data/feature_ranking/pearson3.png)

### Shapiro plot 
![](https://github.com/jim-schwoebel/allie/blob/master/visualize/data/feature_ranking/shapiro3.png)

### RadViz plot
![](https://github.com/jim-schwoebel/allie/blob/master/visualize/data/feature_ranking/radviz3.png)

### Feature correlation plot
![](https://github.com/jim-schwoebel/allie/blob/master/visualize/data/feature_ranking/correlation.png)

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
![](https://github.com/jim-schwoebel/allie/blob/master/visualize/data/modeling/calibration3.png)

### Cluster distance 
![](https://github.com/jim-schwoebel/allie/blob/master/visualize/data/modeling/cluster_distance3.png)

### Elbow plot
![](https://github.com/jim-schwoebel/allie/blob/master/visualize/data/modeling/elbow3.png)

### KS stat plot
![](https://github.com/jim-schwoebel/allie/blob/master/visualize/data/modeling/ks3.png)

### Learning curve
![](https://github.com/jim-schwoebel/allie/blob/master/visualize/data/modeling/learning_curve3.png)

### logr percentile plot
![](https://github.com/jim-schwoebel/allie/blob/master/visualize/data/modeling/logr_percentile_plot3.png)

### Outlier detection 
![](https://github.com/jim-schwoebel/allie/blob/master/visualize/data/modeling/outliers3.png)

### PCA explained variance plot
![](https://github.com/jim-schwoebel/allie/blob/master/visualize/data/modeling/pca_explained_variance3.png)

### Precision/recall graphs 
![](https://github.com/jim-schwoebel/allie/blob/master/visualize/data/modeling/precision-recall3.png)

### Prediction error graphs
![](https://github.com/jim-schwoebel/allie/blob/master/visualize/data/modeling/prediction_error3.png)

### Residuals 
![](https://github.com/jim-schwoebel/allie/blob/master/visualize/data/modeling/residuals3.png)

### ROC curve_train
![](https://github.com/jim-schwoebel/allie/blob/master/visualize/data/modeling/roc_curve_train3.png)

### ROC curve_test
![](https://github.com/jim-schwoebel/allie/blob/master/visualize/data/modeling/roc_curve3.png)

### Siloutte graph
![](https://github.com/jim-schwoebel/allie/blob/master/visualize/data/modeling/siloutte3.png)

### Threshold graph 
![](https://github.com/jim-schwoebel/allie/blob/master/visualize/data/modeling/thresholds3.png)
