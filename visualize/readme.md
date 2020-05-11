## Visualization

This is a univeral visualizer for all types of data. 

Note that this is a setting in the Allie Framework (e.g. "visualize_data": true). 

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
│   │   ├── 0_log_spectrogram_mean_feature_1.png
│   │   ├── 100_log_spectrogram_mean_feature_101.png
        ... [all the features in the current feature set]
│   ├── heatmap.png
│   ├── heatmap_clean.png
│   ├── lasso.png
│   ├── pearson.png
│   └── shapiro.png
└── modeling
    ├── cluster_distance.png
    ├── logr_percentile_plot.png
    ├── outliers.png
    ├── precision-recall.png
    ├── prediction_error.png
    ├── residuals.png
    ├── roc_curve_train.png
    ├── siloutte.png
    └── thresholds.png
```

## Output graphs

Once you run this script, you output many visualizations. These visualizations can be customized within the script itself with some simple modifications. See below for some of the visualizations you can make.

Note that this script considers whether or not to balance datasets (e.g. "balance_data": true in settings.json) - so make sure you adjust your settings as to whether or not you'd like to balance the data before running the script above. These were the settings used to create the visualizations below:

```
{
  "augment_data": false,
  "balance_data": true,
  "clean_data": false,
  "create_YAML": true,
  "default_audio_features": [ "librosa_features", "pyworld_features" ],
  "default_audio_transcriber": "pocketsphinx",
  "default_csv_features": [ "csv_features" ],
  "default_csv_transcriber": "raw text",
  "default_dimensionality_reducer": [ "pca" ],
  "default_feature_selector": [ "lasso" ],
  "default_image_features": [ "image_features" ],
  "default_image_transcriber": "tesseract",
  "default_scaler": [ "standard_scaler" ],
  "default_text_features": [ "nltk_features" ],
  "default_text_transcriber": "raw text",
  "default_training_script": [ "tpot" ],
  "default_video_features": [ "video_features" ],
  "default_video_transcriber": "tesseract (averaged over frames)",
  "model_compress": false,
  "reduce_dimensions": true,
  "scale_features": true,
  "select_features": true,
  "test_size": 0.25,
  "transcribe_audio": false,
  "transcribe_csv": true,
  "transcribe_image": true,
  "transcribe_text": true,
  "transcribe_videos": true,
  "visualize_data": true
}
```
### numbers in each class
![](https://github.com/jim-schwoebel/allie/blob/master/visualize/data/classes.png)

## top 10 features (if 10 features exist)
![](https://github.com/jim-schwoebel/allie/blob/master/visualize/data/feature_importance.png)

## PCA plot
![](https://github.com/jim-schwoebel/allie/blob/master/visualize/data/pca.png)

## Pearson ranking plot
![](https://github.com/jim-schwoebel/allie/blob/master/visualize/data/pearson.png)

## tSNE plot
![](https://github.com/jim-schwoebel/allie/blob/master/visualize/data/tsne.png)

## Working on now

- https://umap-learn.readthedocs.io/en/latest/basic_usage.html
- https://github.com/sepandhaghighi/pycm - confusion matrix / ROC curve
- https://github.com/wcipriano/pretty-print-confusion-matrix

## Using streamlit

```
streamlit run test.py
```
Outputs various graphs on the data.

## Featurization tutorials (Readmes)
* [Feature selection techniques in Python](https://towardsdatascience.com/feature-selection-techniques-in-machine-learning-with-python-f24e7da3f36e)
* [How to visualize anything in Yellowbrick](https://medium.com/analytics-vidhya/how-to-visualize-anything-in-machine-learning-using-yellowbrick-and-mlxtend-39c45e1e9e9f)
* [Intro to data visualization in Python](https://gilberttanner.com/blog/introduction-to-data-visualization-inpython)
* [Matplotlib bar chart](https://pythonspot.com/matplotlib-bar-chart/)
* [Visualize machine learning Pandas](https://machinelearningmastery.com/visualize-machine-learning-data-python-pandas/)
* [Top 50 Matplotlib visualizations](https://www.machinelearningplus.com/plots/top-50-matplotlib-visualizations-the-master-plots-python/)
* [Dimensionality reduction techniques](https://www.analyticsvidhya.com/blog/2018/08/dimensionality-reduction-techniques-python/)

## Additional documentation
* [MLXtend](https://github.com/rasbt/mlxtend) - for visualizing models
* [Plotly learning curves](https://github.com/mitmedialab/vizml/blob/master/notebooks/Plotly%20Performance.ipynb)
* [Snowflake](https://github.com/doubledherin/Audio_Snowflake) - audio song visualization
* [Streamit](https://github.com/streamlit/streamlit)
* [Voicebook: Chapter 6 Visualization](https://github.com/jim-schwoebel/voicebook/tree/master/chapter_6_visualization)
* [Yellowbrick](https://www.scikit-yb.org/en/latest/)
