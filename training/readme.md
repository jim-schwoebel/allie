## Training scripts 

Quickly train according to the default_training_script using model.py.

![](https://github.com/jim-schwoebel/Allie/blob/master/training/helpers/train.gif)

## Getting started

All you need to do to get started is go to this repository and run upgrade.py followed by model.py:

```
cd allie/training
python3 upgrade.py
python3 model.py 
```

You then will be asked a few questions regarding the training process (in terms of data type, number of classes, and the name of the model). Note that --> indicates typed responses. 

### regression model example 
For regression model training, you need to insert a .CSV file for training. You can then specify the target classes here from the spreadsheet and the models will then be trained with the specified model trainers.

### classification model example 
You can take in any list of files in folders for classification problems. 

```
what problem are you solving? (1-audio, 2-text, 3-image, 4-video, 5-csv)
--> 1

 OK cool, we got you modeling audio files 

how many classes would you like to model? (2 available) 
--> 2
these are the available classes: 
['one', 'two']
what is class #1 
--> one
what is class #2 
--> two
what is the 1-word common name for the problem you are working on? (e.g. gender for male/female classification) 
--> test
is this a classification (c) or regression (r) problem? 
--> c
```

After this, the model will be trained and placed in the models/[sampletype_models] directory. For example, if you trained an audio model with TPOT, the model will be placed in the allie/models/audio_models/ directory. 

For automated training, you can alternatively pass through sys.argv[] inputs as follows:

```
python3 model.py audio 2 c male female
```
Where:
- audio = audio file type 
- 2 = 2 classes 
- c = classification (r for regression)
- male = first class
- female = second class [via N number of classes]

Goal is to make an output folder like this:
```
└── commonname_tpot
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
    ├── data
    │   ├── m_f_all.csv
    │   ├── m_f_all_transformed.csv
    │   ├── m_f_test.csv
    │   ├── m_f_test_transformed.csv
    │   ├── m_f_train.csv
    │   └── m_f_train_transformed.csv
    ├── feature_ranking
    │   ├── feature_importance.png
    │   ├── feature_plots
    │   │   ├── 0_onset_length.png
    │   │   ├── 100_mfcc_4_minv.png
    │   │   ├── 101_mfcc_4_median.png
    │   │   ├── 102_mfcc_5_mean.png
    │   │   ├── 103_mfcc_5_std.png
    │   │   ├── 104_mfcc_5_maxv.png
    │   │   ├── 105_mfcc_5_minv.png
    │   │   ├── 106_mfcc_5_median.png
    │   │   ├── 107_mfcc_6_mean.png
    │   │   ├── 108_mfcc_6_std.png
    │   │   ├── 109_mfcc_6_maxv.png
    │   │   ├── 10_onset_strength_minv.png
    │   │   ├── 110_mfcc_6_minv.png
    │   │   ├── 111_mfcc_6_median.png
    │   │   ├── 112_mfcc_7_mean.png
    │   │   ├── 113_mfcc_7_std.png
    │   │   ├── 114_mfcc_7_maxv.png
    │   │   ├── 115_mfcc_7_minv.png
    │   │   ├── 116_mfcc_7_median.png
    │   │   ├── 117_mfcc_8_mean.png
    │   │   ├── 118_mfcc_8_std.png
    │   │   ├── 119_mfcc_8_maxv.png
    │   │   ├── 11_onset_strength_median.png
    │   │   ├── 120_mfcc_8_minv.png
    │   │   ├── 121_mfcc_8_median.png
    │   │   ├── 122_mfcc_9_mean.png
    │   │   ├── 123_mfcc_9_std.png
    │   │   ├── 124_mfcc_9_maxv.png
    │   │   ├── 125_mfcc_9_minv.png
    │   │   ├── 126_mfcc_9_median.png
    │   │   ├── 127_mfcc_10_mean.png
    │   │   ├── 128_mfcc_10_std.png
    │   │   ├── 129_mfcc_10_maxv.png
    │   │   ├── 12_rhythm_0_mean.png
    │   │   ├── 130_mfcc_10_minv.png
    │   │   ├── 131_mfcc_10_median.png
    │   │   ├── 132_mfcc_11_mean.png
    │   │   ├── 133_mfcc_11_std.png
    │   │   ├── 134_mfcc_11_maxv.png
    │   │   ├── 135_mfcc_11_minv.png
    │   │   ├── 136_mfcc_11_median.png
    │   │   ├── 137_mfcc_12_mean.png
    │   │   ├── 138_mfcc_12_std.png
    │   │   ├── 139_mfcc_12_maxv.png
    │   │   ├── 13_rhythm_0_std.png
    │   │   ├── 140_mfcc_12_minv.png
    │   │   ├── 141_mfcc_12_median.png
    │   │   ├── 142_poly_0_mean.png
    │   │   ├── 143_poly_0_std.png
    │   │   ├── 144_poly_0_maxv.png
    │   │   ├── 145_poly_0_minv.png
    │   │   ├── 146_poly_0_median.png
    │   │   ├── 147_poly_1_mean.png
    │   │   ├── 148_poly_1_std.png
    │   │   ├── 149_poly_1_maxv.png
    │   │   ├── 14_rhythm_0_maxv.png
    │   │   ├── 150_poly_1_minv.png
    │   │   ├── 151_poly_1_median.png
    │   │   ├── 152_spectral_centroid_mean.png
    │   │   ├── 153_spectral_centroid_std.png
    │   │   ├── 154_spectral_centroid_maxv.png
    │   │   ├── 155_spectral_centroid_minv.png
    │   │   ├── 156_spectral_centroid_median.png
    │   │   ├── 157_spectral_bandwidth_mean.png
    │   │   ├── 158_spectral_bandwidth_std.png
    │   │   ├── 159_spectral_bandwidth_maxv.png
    │   │   ├── 15_rhythm_0_minv.png
    │   │   ├── 160_spectral_bandwidth_minv.png
    │   │   ├── 161_spectral_bandwidth_median.png
    │   │   ├── 162_spectral_contrast_mean.png
    │   │   ├── 163_spectral_contrast_std.png
    │   │   ├── 164_spectral_contrast_maxv.png
    │   │   ├── 165_spectral_contrast_minv.png
    │   │   ├── 166_spectral_contrast_median.png
    │   │   ├── 167_spectral_flatness_mean.png
    │   │   ├── 168_spectral_flatness_std.png
    │   │   ├── 169_spectral_flatness_maxv.png
    │   │   ├── 16_rhythm_0_median.png
    │   │   ├── 170_spectral_flatness_minv.png
    │   │   ├── 171_spectral_flatness_median.png
    │   │   ├── 172_spectral_rolloff_mean.png
    │   │   ├── 173_spectral_rolloff_std.png
    │   │   ├── 174_spectral_rolloff_maxv.png
    │   │   ├── 175_spectral_rolloff_minv.png
    │   │   ├── 176_spectral_rolloff_median.png
    │   │   ├── 177_zero_crossings_mean.png
    │   │   ├── 178_zero_crossings_std.png
    │   │   ├── 179_zero_crossings_maxv.png
    │   │   ├── 17_rhythm_1_mean.png
    │   │   ├── 180_zero_crossings_minv.png
    │   │   ├── 181_zero_crossings_median.png
    │   │   ├── 182_RMSE_mean.png
    │   │   ├── 183_RMSE_std.png
    │   │   ├── 184_RMSE_maxv.png
    │   │   ├── 185_RMSE_minv.png
    │   │   ├── 186_RMSE_median.png
    │   │   ├── 18_rhythm_1_std.png
    │   │   ├── 19_rhythm_1_maxv.png
    │   │   ├── 1_onset_detect_mean.png
    │   │   ├── 20_rhythm_1_minv.png
    │   │   ├── 21_rhythm_1_median.png
    │   │   ├── 22_rhythm_2_mean.png
    │   │   ├── 23_rhythm_2_std.png
    │   │   ├── 24_rhythm_2_maxv.png
    │   │   ├── 25_rhythm_2_minv.png
    │   │   ├── 26_rhythm_2_median.png
    │   │   ├── 27_rhythm_3_mean.png
    │   │   ├── 28_rhythm_3_std.png
    │   │   ├── 29_rhythm_3_maxv.png
    │   │   ├── 2_onset_detect_std.png
    │   │   ├── 30_rhythm_3_minv.png
    │   │   ├── 31_rhythm_3_median.png
    │   │   ├── 32_rhythm_4_mean.png
    │   │   ├── 33_rhythm_4_std.png
    │   │   ├── 34_rhythm_4_maxv.png
    │   │   ├── 35_rhythm_4_minv.png
    │   │   ├── 36_rhythm_4_median.png
    │   │   ├── 37_rhythm_5_mean.png
    │   │   ├── 38_rhythm_5_std.png
    │   │   ├── 39_rhythm_5_maxv.png
    │   │   ├── 3_onset_detect_maxv.png
    │   │   ├── 40_rhythm_5_minv.png
    │   │   ├── 41_rhythm_5_median.png
    │   │   ├── 42_rhythm_6_mean.png
    │   │   ├── 43_rhythm_6_std.png
    │   │   ├── 44_rhythm_6_maxv.png
    │   │   ├── 45_rhythm_6_minv.png
    │   │   ├── 46_rhythm_6_median.png
    │   │   ├── 47_rhythm_7_mean.png
    │   │   ├── 48_rhythm_7_std.png
    │   │   ├── 49_rhythm_7_maxv.png
    │   │   ├── 4_onset_detect_minv.png
    │   │   ├── 50_rhythm_7_minv.png
    │   │   ├── 51_rhythm_7_median.png
    │   │   ├── 52_rhythm_8_mean.png
    │   │   ├── 53_rhythm_8_std.png
    │   │   ├── 54_rhythm_8_maxv.png
    │   │   ├── 55_rhythm_8_minv.png
    │   │   ├── 56_rhythm_8_median.png
    │   │   ├── 57_rhythm_9_mean.png
    │   │   ├── 58_rhythm_9_std.png
    │   │   ├── 59_rhythm_9_maxv.png
    │   │   ├── 5_onset_detect_median.png
    │   │   ├── 60_rhythm_9_minv.png
    │   │   ├── 61_rhythm_9_median.png
    │   │   ├── 62_rhythm_10_mean.png
    │   │   ├── 63_rhythm_10_std.png
    │   │   ├── 64_rhythm_10_maxv.png
    │   │   ├── 65_rhythm_10_minv.png
    │   │   ├── 66_rhythm_10_median.png
    │   │   ├── 67_rhythm_11_mean.png
    │   │   ├── 68_rhythm_11_std.png
    │   │   ├── 69_rhythm_11_maxv.png
    │   │   ├── 6_tempo.png
    │   │   ├── 70_rhythm_11_minv.png
    │   │   ├── 71_rhythm_11_median.png
    │   │   ├── 72_rhythm_12_mean.png
    │   │   ├── 73_rhythm_12_std.png
    │   │   ├── 74_rhythm_12_maxv.png
    │   │   ├── 75_rhythm_12_minv.png
    │   │   ├── 76_rhythm_12_median.png
    │   │   ├── 77_mfcc_0_mean.png
    │   │   ├── 78_mfcc_0_std.png
    │   │   ├── 79_mfcc_0_maxv.png
    │   │   ├── 7_onset_strength_mean.png
    │   │   ├── 80_mfcc_0_minv.png
    │   │   ├── 81_mfcc_0_median.png
    │   │   ├── 82_mfcc_1_mean.png
    │   │   ├── 83_mfcc_1_std.png
    │   │   ├── 84_mfcc_1_maxv.png
    │   │   ├── 85_mfcc_1_minv.png
    │   │   ├── 86_mfcc_1_median.png
    │   │   ├── 87_mfcc_2_mean.png
    │   │   ├── 88_mfcc_2_std.png
    │   │   ├── 89_mfcc_2_maxv.png
    │   │   ├── 8_onset_strength_std.png
    │   │   ├── 90_mfcc_2_minv.png
    │   │   ├── 91_mfcc_2_median.png
    │   │   ├── 92_mfcc_3_mean.png
    │   │   ├── 93_mfcc_3_std.png
    │   │   ├── 94_mfcc_3_maxv.png
    │   │   ├── 95_mfcc_3_minv.png
    │   │   ├── 96_mfcc_3_median.png
    │   │   ├── 97_mfcc_4_mean.png
    │   │   ├── 98_mfcc_4_std.png
    │   │   ├── 99_mfcc_4_maxv.png
    │   │   └── 9_onset_strength_maxv.png
    │   ├── heatmap.png
    │   ├── heatmap_clean.png
    │   ├── lasso.png
    │   ├── pearson.png
    │   └── shapiro.png
    ├── modeling
    │   ├── calibration.png
    │   ├── cluster_distance.png
    │   ├── elbow.png
    │   ├── ks.png
    │   ├── learning_curve.png
    │   ├── logr_percentile_plot.png
    │   ├── outliers.png
    │   ├── pca_explained_variance.png
    │   ├── precision-recall.png
    │   ├── prediction_error.png
    │   ├── residuals.png
    │   ├── roc_curve.png
    │   ├── roc_curve_train.png
    │   ├── siloutte.png
    │   └── thresholds.png
    └── models
        ├── m_f_librosa_features.png
        ├── m_f_librosa_features_tpotclassifier.json
        ├── m_f_librosa_features_tpotclassifier.pickle
        ├── m_f_librosa_features_tpotclassifier.py
        └── m_f_librosa_features_tpotclassifier_transform.pickle
```

Now you're ready to go to load these models and [make predictions](https://github.com/jim-schwoebel/allie/tree/master/models).

## Model training scripts

Here is a quick review of all the potential default_training_script settings:

| Setting | License | Accurate? | Quick? | Good docs? | Classification | Regression | Description |
| --------- |  --------- |  --------- | --------- | --------- | --------- | --------- |  --------- | 
| '[alphapy](https://alphapy.readthedocs.io/en/latest/user_guide/pipelines.html#model-object-creation)' | Apache 2.0 |  ✅ | ❌ | ✅ | ✅ | ✅  | Highly customizable setttings for data science pipelines/feature selection. |
| '[atm](https://github.com/HDI-Project/ATM)' | MIT License | ✅ | ✅ | ✅ | ✅ | ✅  | give ATM a classification problem and a dataset as a CSV file, and ATM will build the best model it can. |
| '[autogbt](https://github.com/pfnet-research/autogbt-alt)' | MIT License |  ✅ | ✅ | ✅ | ✅ | ✅  | An experimental Python package that reimplements AutoGBT using LightGBM and Optuna. |
| '[autogluon](https://github.com/awslabs/autogluon)'| Apache 2.0 | ✅ | ✅ | ✅ | ✅ | ✅  | AutoGluon: AutoML Toolkit for Deep Learning. |
| '[autokaggle](https://github.com/datamllab/autokaggle)' | Apache 2.0 | ❌ | ✅ | ❌ | ✅ | ✅  | Automated ML system trained using gbdt (regression and classification). |
| '[autokeras](https://autokeras.com/)'| MIT License | ✅ | ❌ | ✅ | ✅ | ✅ | Automatic optimization of a neural network using neural architecture search (takes a very long time) - consistently has problems associated with saving and loading models in keras. |
| '[autopytorch](https://github.com/automl/Auto-PyTorch)' | Apache 2.0 | ❌ | ✅ | ❌ | ✅ | ✅  | Brute-Force all sklearn models with all parameters using .fit/.predict. |
| '[btb](https://github.com/HDI-Project/BTB)' | MIT License | ❌ | ✅ | ❌ | ✅ | ✅  | Hyperparameter tuning with various ML algorithms in scikit-learn using genetic algorithms. |
| '[cvopt](https://github.com/genfifth/cvopt)' | BSD 2-Clause "Simplified" License| ✅ | ✅ | ❌ | ✅ | ✅  | Machine learning parameter search / feature selection module with visualization. |
| '[devol](https://github.com/joeddav/devol)' | MIT License |  ✅ | ❌ | ❌ | ✅ | ✅  | Genetic programming keras cnn layers. |
| '[gama](https://github.com/PGijsbers/gama)' | Apache 2.0 |  ✅ | ✅ | ✅ | ✅ | ✅  | An automated machine learning tool based on genetic programming. |
| '[hungabunga](https://github.com/ypeleg/HungaBunga)' | MIT License | ❌ | ✅ | ❌ | ✅ | ✅  | HungaBunga: Brute-Force all sklearn models with all parameters using .fit .predict! |
| '[hyperband](https://github.com/thuijskens/scikit-hyperband)' | BSD 3-Clause "New" or "Revised" License |  ✅ | ✅ | ✅ | ✅ | ✅  | Implements a class HyperbandSearchCV that works exactly as GridSearchCV and RandomizedSearchCV from scikit-learn do, except that it runs the hyperband algorithm under the hood. |
| '[hypsklearn](https://github.com/hyperopt/hyperopt-sklearn)' | [BSD 3-Clause "New" or "Revised" License](https://github.com/hyperopt/hyperopt-sklearn/blob/master/LICENSE.txt) |  ✅ | ✅ | ✅ | ✅ | ✅  | Hyperparameter optimization on scikit-learn models. |
| '[imbalance](https://pypi.org/project/imbalanced-learn/)' | MIT License |  ✅ | ✅ | ✅ | ✅ | ✅  | Imbalance learn different ML techniques to work on data with different numbers of samples. |
| '[keras](https://keras.io/getting-started/faq/)' | MIT License |  ✅ | ✅ | ✅ | ✅ | ✅  | Simple MLP network architecture (quick prototype - if works may want to use autoML settings). |
| '[ludwig](https://github.com/uber/ludwig)' | Apache 2.0 | ✅ | ❌ | ✅ | ✅ | ✅  | Deep learning (simple ludwig). - convert every feature to numerical data. |
| '[mlblocks](https://github.com/HDI-Project/MLBlocks)' | MIT License | ✅ | ❌ | ❌ | ✅ | ✅  | Most recent framework @ MIT, regression and classification. |
| '[neuraxle](https://github.com/Neuraxio/Neuraxle)' | Apache 2.0 | ✅ | ✅ | ❌ | ❌ | ✅  | A Sklearn-like Framework for Hyperparameter Tuning and AutoML in Deep Learning projects. |
| '[safe](https://github.com/ModelOriented/SAFE)' | MIT License | ❌ | ✅ | ❌ | ✅ | ✅  | Black box trainer / helps reduce opacity of ML models while increasing accuracy. |
| '[scsr](https://github.com/jim-schwoebel/voicebook/blob/master/chapter_4_modeling/train_audioregression.py)' | Apache 2.0 |  ❌ | ✅ | ✅ | ✅ | ✅  | Simple classification / regression (built by Jim from NLX-model). |
| '[tpot](https://github.com/EpistasisLab/tpot)' (default) | LGPL-3.0 |  ❌ | ✅ | ✅ | ✅ | ✅  | TPOT classification / regression (autoML). |

Note that you can customize the default_training_script in the settings.json. If you include multiple default training scripts in series e.g. ['keras','tpot'] it will go through and model each of these sessions serially. A sample settings.json with the ['tpot'] setting is shown below, for reference (this is the default setting):

```python3
{
  "version": "1.0.0",
  "augment_data": false,
  "balance_data": true,
  "clean_data": false,
  "create_YAML": true,
  "default_audio_features": [ "librosa_features" ],
  "default_audio_transcriber": ["pocketsphinx"],
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
  "default_video_transcriber": ["tesseract (averaged over frames)"],
  "model_compress": false,
  "reduce_dimensions": false,
  "scale_features": true,
  "select_features": false,
  "test_size": 0.25,
  "transcribe_audio": true,
  "transcribe_csv": true,
  "transcribe_image": true,
  "transcribe_text": true,
  "transcribe_videos": true,
  "visualize_data": false
}
```

## [Metrics](https://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics)
Metrics are standardized across all model training methods to allow for interoperability across the various AutoML frameworks used. These methods differ between classification and regression models, and use the [scikit-learn metrics API](https://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics).

### Classification
See the [Classification metrics section](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics) of the user guide for further details.

- accuracy	sklearn.metrics.accuracy_score
- balanced_accuracy metrics.balanced_accuracy_score
- precision	sklearn.metrics.precision_score
- recall	sklearn.metrics.recall_score
- f1	sklearn.metrics.f1_score (pos_label=1)
- f1Micro	sklearn.metrics.f1_score(average='micro')
- f1Macro	sklearn.metrics.f1_score(average='macro')
- rocAuc	sklearn.metrics.roc_auc_score
- rocAucMicro	sklearn.metrics.roc_auc_score(average='micro')
- rocAucMacro	sklearn.metrics.roc_auc_score(average='macro')
- confusion matrix

The output .JSON with metrics will look something like this:

```
{"sample type": "audio", "created date": "2020-05-18 17:10:24.747250", "session id": "6f81e898-bd03-4ba8-91e7-caf281748b83", "classes": ["males", "females"], "model type": "classification", "model name": "gender_atm.pickle", "metrics": {"accuracy": 0.8076923076923077, "balanced_accuracy": 0.8212121212121212, "precision": 0.7142857142857143, "recall": 0.9090909090909091, "f1_score": 0.8, "f1_micro": 0.8076923076923077, "f1_macro": 0.8074074074074074, "roc_auc": 0.8212121212121213, "roc_auc_micro": 0.8212121212121213, "roc_auc_macro": 0.8212121212121213, "confusion_matrix": [[22, 8], [2, 20]], "classification_report": "              precision    recall  f1-score   support\n\n       males       0.92      0.73      0.81        30\n     females       0.71      0.91      0.80        22\n\n    accuracy                           0.81        52\n   macro avg       0.82      0.82      0.81        52\nweighted avg       0.83      0.81      0.81        52\n"}, "settings": {"version": 1.0, "augment_data": false, "balance_data": true, "clean_data": false, "create_YAML": true, "default_audio_features": ["librosa_features"], "default_audio_transcriber": ["pocketsphinx"], "default_csv_features": ["csv_features"], "default_csv_transcriber": ["raw text"], "default_dimensionality_reducer": ["pca"], "default_feature_selector": ["lasso"], "default_image_features": ["image_features"], "default_image_transcriber": ["tesseract"], "default_scaler": ["standard_scaler"], "default_text_features": ["nltk_features"], "default_text_transcriber": "raw text", "default_training_script": ["atm"], "default_video_features": ["video_features"], "default_video_transcriber": ["tesseract (averaged over frames)"], "model_compress": false, "reduce_dimensions": false, "scale_features": true, "select_features": false, "test_size": 0.25, "transcribe_audio": false, "transcribe_csv": true, "transcribe_image": true, "transcribe_text": true, "transcribe_videos": true, "visualize_data": false}, "transformer name": "gender_atm_transform.pickle", "training data": ["gender_all.csv", "gender_train.csv", "gender_test.csv", "gender_all_transformed.csv", "gender_train_transformed.csv", "gender_test_transformed.csv"], "sample X_test": [-1.1922838749534224, -1.1929940219927937, -1.2118533388328951, -1.19399037815855, -0.0021316583752660208, -1.190766733291579, 3.849427576362443, -1.9204306379246512, -1.125006509709126, -1.545343386974132, 0.0, -1.9143559991789347, 0.0, 0.0, 0.0, 0.0, 0.0, 0.393398281153827, -1.3419713201673347, -0.31267044157215834, 0.7145414374550684, 0.4408029245727885, -0.08794721338696684, -1.2908803928098331, -0.7686039200276, 0.27355089960768003, -0.023868957327515483, -0.33493242828747644, -1.3534071112752446, -1.0663649542614346, 0.07710406397104422, -0.2620966016717233, 0.07442950075672321, -1.3586158282078342, -0.7229877787428609, 0.47998843890208265, 0.14338269561742617, 1.0289942793010851, -1.372750144952516, 0.16682659474904094, 1.3103462536843684, 1.0852524978109923, 1.3053083662553129, -1.3204536680340357, 0.44443569253872167, 1.5316554076888196, 1.3536334361962563, -0.008584151637748075, -1.3044478087345126, -0.7798374876334464, 0.4391526047447745, 0.0328217238195433, -1.6588812589246522, -1.321496997694275, -2.3058731114054853, -0.9428316620058456, -1.6054110631290472, -1.9015638712936165, -1.3181264260875947, -2.3400689045953977, -1.2257820801881496, -1.8580680903896456, -1.6875097168794329, -1.4113531659207812, -2.07877796917062, -1.0679177417254373, -1.6568549096503389, -1.7221630518945865, -1.3474956240767961, -2.135089810138456, -1.1077177226463242, -1.6750981593832228, -2.0781689670816696, -1.407674752516947, -2.466670316340356, -1.4224935683039404, -2.0228930807342236, -0.64807477922011, -0.8067816782034948, -0.6897448732292194, 0.666399646568167, -0.758504945965009, 1.686482678278769, -0.4026076385117994, 0.9932683897094231, 1.7178647246613565, 1.5792208834827586, 1.4390951796584497, -0.8982625212821831, 0.8965596316516141, 2.1707288819224306, 1.161596616542146, 0.23397581925946248, 1.5058743230731093, -0.3256390312368899, -0.1997911700368024, 0.3913640764940668, -0.23106922325335622, 0.2688364420481832, 0.6136517467870749, 0.9853170289935439, -0.45156833200238317, -0.4039027759848379, -1.4047903060201576, -1.6199422985321121, 0.6428248152306695, -0.5021531991290126, 0.6164464311350407, -1.8395730917192974, -1.1262962931860703, 1.6725808902325139, 0.5842734033850393, 0.7649135160627001, -0.2873589631759208, -0.17297527885266198, 1.2832640412296985, 0.8199787566304052, 0.17266635859916557, -1.530303895687936, -1.035762063808967, 1.2059152644116997, 0.05560008103368553, -1.4240650565517237, 0.1317016024108744, -1.5473895633630874, -0.044686224198341895, -1.622607473245282, -0.7811768474081445, -1.3574788179509854, -1.4277185112430948, 0.8006696574528496, -0.739635158959708, -0.43596672823397514, -0.9941316877059948, -1.5527334792494474, 1.0241352281932175, -0.7322079093175251, -0.6928023654465668, 2.256128519734752, -0.6674105694489904, -0.5118001652350106, -1.119649264235433, 0.41747338869357786, -0.6721303658862441, -0.6423304904495852, 0.721819384516983, 0.3675016869618989, -0.4590847986919945, -0.6915104068484828, -0.7457704134904531, 0.5219531770125143, -0.4113120309505084, -1.722127309595865, -1.6950722496795612, -2.4638679454105468, -0.8850791263479258, -1.3195505366838527, -1.3493943022990107, -0.14490564220224492, -0.431207418652445, -0.5543954310823219, -1.1040016294903932, 1.7500526600682629, -1.0972072350519622, 0.117984869151784, 3.8430207941961694, 1.7439873570898108, -0.3527571572132666, -0.29344427565369596, -0.30383194101857836, -0.27273455031977006, -0.4812192589932978, -1.9936313049319827, -1.1758892116636155, -2.7207295590332827, -0.619693296869786, -1.753005830475332, -1.6994742556679854, -1.8507272006073032, -2.2045026156861076, -0.805498984978142, -1.3546162014501322, -0.3736833281263094, -0.7998460443253107, -0.7198574596318982, 0.9646067905470591, -0.3263749779375788], "sample y_test": 0}
```
### Regression

See the [Regression metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics) section of the user guide for further details.

- max_error metric calculates the maximum residual error.
- Mean absolute error regression loss
- Mean squared error regression loss
- Median absolute error regression loss
- R^2 (coefficient of determination) regression score function.

The output .JSON will look something like this:

```
{"sample type": "audio", "created date": "2020-05-18 16:54:24.805218", "session id": "0d88a075-a7ab-487c-b9c2-3c7c1ae09d03", "classes": ["males", "females"], "model type": "regression", "model name": "alsdfjlsajdf_autokeras.pickle", "metrics": {"mean_absolute_error": 0.1077826815442397, "mean_squared_error": 0.07462779294115354, "median_absolute_error": 1.601874828338623e-07, "r2_score": 0.6942521937683649}, "settings": {"version": 1.0, "augment_data": false, "balance_data": true, "clean_data": false, "create_YAML": true, "default_audio_features": ["librosa_features"], "default_audio_transcriber": ["pocketsphinx"], "default_csv_features": ["csv_features"], "default_csv_transcriber": ["raw text"], "default_dimensionality_reducer": ["pca"], "default_feature_selector": ["lasso"], "default_image_features": ["image_features"], "default_image_transcriber": ["tesseract"], "default_scaler": ["standard_scaler"], "default_text_features": ["nltk_features"], "default_text_transcriber": "raw text", "default_training_script": ["autokeras"], "default_video_features": ["video_features"], "default_video_transcriber": ["tesseract (averaged over frames)"], "model_compress": false, "reduce_dimensions": false, "scale_features": true, "select_features": false, "test_size": 0.25, "transcribe_audio": false, "transcribe_csv": true, "transcribe_image": true, "transcribe_text": true, "transcribe_videos": true, "visualize_data": false}, "transformer name": "alsdfjlsajdf_autokeras_transform.pickle", "training data": ["alsdfjlsajdf_all.csv", "alsdfjlsajdf_train.csv", "alsdfjlsajdf_test.csv", "alsdfjlsajdf_all_transformed.csv", "alsdfjlsajdf_train_transformed.csv", "alsdfjlsajdf_test_transformed.csv"], "sample X_test": [0.1617091154403997, 0.0720205638907688, 0.18343183536874122, 0.1672803364150215, -0.6768015341469388, 0.017543357215669842, -0.6984277765173735, 0.38615567322347166, 0.16113189024852953, 0.4001187632663553, 0.0, 0.4873511913307515, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5387127579642551, 0.41066147715164214, -0.19825322401743353, -0.5706310390061365, -0.4986647848248428, -0.49189625109547513, -0.17493751600011975, -0.4014143051606473, -0.3465660754407285, -0.45260975302246853, -0.45589495937639435, -0.4819237375486541, -0.5896133670337916, -0.330215482212022, -0.44043279851095235, -0.5234836122938473, -0.5272903362078499, -0.7088703332128212, -0.41531341914812164, -0.48748939273270225, -0.2618951732682184, -0.9856411619824048, -0.9400354843020243, -0.08771221511743747, -0.18482942654692902, -0.24959459599220649, -0.461204392217036, -0.72057074420697, -0.24348964155889352, -0.147406186520558, -0.9162291667305105, -0.85242473112346, -1.504207019296117, -0.580649405797514, -0.8350039253357034, -1.0312312405349793, -0.977872259346789, -1.601090597322523, -0.7066534998040032, -0.9583625521744131, -0.5346719885538214, -0.741154613709866, -0.9176221407192556, -0.4018183816122035, -0.425862924649035, -0.3009786272341064, -0.8443910549014533, -0.7325823664165042, -0.23267512513445213, -0.23968657015625028, -0.3224171248346584, -0.8056733325409708, -0.6135216108652851, -0.13110858180526833, -0.28141136962232266, -0.18871553667615804, -0.42816764163386845, -0.2448096902479657, -0.21662295115147656, -0.18750337165969908, 1.5811135800822052, 1.0748087568347513, 1.6267559896630304, 0.8394627144655394, 1.678672293821748, 0.8181658095231239, 0.4339603292131989, 0.7004815738393242, 0.17877752619514414, 0.9398282245795986, 0.5457165094370632, -0.5397498486758648, 0.8230854055311116, 0.684484079509053, 0.5435214979128995, -0.4720442259612947, 0.4558838955539394, 0.8613304491649149, -0.144243205486765, -0.6468251880219985, 0.9274784274905465, 0.9159086289445996, 1.1015505829020054, -0.30645164342333514, 0.9433508117147407, -0.005069235321852924, -0.6259481967763515, 0.27171487175342257, 0.22411722556396624, 0.02865051782674097, -0.00913579700429947, 0.7227844383541874, 0.1264126455463073, 0.05306725012946615, -0.09772130396457164, -0.6091616462319488, 0.7263804750235214, -0.35747120634336915, -1.2508217851043137, -0.52900673597145, 0.7117290727433093, -0.4008435053731832, 0.25951657943776824, 1.0586025389442921, 0.7687242828311188, 0.10088671872271811, 0.3803402834415134, 0.21974339908525978, -0.3821105808822284, -0.027745943999639745, 0.7102387217713214, 0.6062578953010683, 0.8163296719569486, 0.5948418634657724, 0.5621170094004995, 0.4145319391371441, 0.25245737630504206, 0.23658108163017624, 0.36976632978770213, 0.4676565049792853, 0.4816786341414505, -0.05879637131605244, 0.19011971656748472, 0.7265638597457408, 0.4231243844206609, -2.859067353164431, 3.5425423206598614, -0.11121182154990267, -2.8269772874698686, -2.8586061324370533, 2.714673144897261, 3.448792952538426, 2.7455492759982465, 0.029867744349596773, 2.7262814780550926, -1.0338981213024936, -0.3891019561013453, -0.31294708873454485, -0.9377285645101899, -1.0164696231472012, -1.1425686362538043, 0.47521014698574116, -0.4525955078651535, -0.9138467786910969, -1.2621235383545972, 0.11086650741096651, -0.48742707748473124, -0.23350661020516406, 0.4736030753566866, 0.17481630054955186, -0.4646954170267768, -0.31775626788395456, -0.3101992352332374, -0.4453270996903978, -0.5292092668805664, -1.0696347050704478, 0.037624033706366376, -0.1588525625765062, -0.7051766164562955, -1.0721717209500161, -0.9504316566439719, -0.5537952472654982, -0.2727396701260328, -0.6747603035701513, -0.9216852217847525, 3.2027683754576968, 3.771258992684294, 3.109624582693844, 0.10004242630174563, 3.2109347346201025], "sample y_test": 0}
```

## Actively working on (in future)

### 😉 Exploring now
* [metric_learn](https://github.com/scikit-learn-contrib/metric-learn)
* [hyperparameter tuning w/ Ray](https://github.com/ray-project/ray) - tuning hyperparameters 
* '[adanet](https://github.com/tensorflow/adanet)' - Google's AutoML framework in tensorflow (https://github.com/tensorflow/adanet).
* [MLBox](https://github.com/AxeldeRomblay/MLBox) - State-of-the art predictive models for classification and regression (Deep Learning, Stacking, LightGBM,…).
* [SMAC3](https://github.com/automl/SMAC3) - SMAC performs Bayesian Optimization in combination with a aggressive racing mechanism to efficiently decide which of two configuration performs better.
* [Keras-tuner](https://github.com/keras-team/keras-tuner) - Hyperparameter tuning for humans.
* '[gentun](https://github.com/gmontamat/gentun)' - genetic algorithm approach with distributed training capability.
* '[python-sherpa](https://github.com/sherpa-ai/sherpa)' - sherpa bayesian hyperparameter optimization
* [Misvm](https://github.com/garydoranjr/misvm)
* [Tacotron architecture](https://github.com/KinglittleQ/GST-Tacotron) 
* getting into various other ML architectures for keyword spotting, etc.

#### 🗜️ Model compression 
* [PocketFlow](https://github.com/Tencent/PocketFlow) - allow for ludwig model compression.
* '[keras-inference-time-optimizer](https://github.com/ZFTurbo/Keras-inference-time-optimizer) - restructure keras neural network to reduce inference time without reducing accuracy.
* '[tensorflow-model-optimization](https://github.com/tensorflow/model-optimization) - keras-compatible model compression platform using quantization techniques

### 👎 Discarded (in archived folder)

Note these are in the 'archived' folder in case you'd like to expand upon them and use these training back-ends in an expeirmental way.

* '[autobazaar](https://github.com/HDI-Project/AutoBazaar)' - AutoBazaar: An AutoML System from the Machine Learning Bazaar (from the Data to AI Lab at MIT). - this was a bit hard to extract a machine learning model from and the schema was a little hard to make interoperable with Allie; may get to this later if the framework is further supported. 
* '[autokeras](https://autokeras.com/)' - automatic optimization of a neural network using neural architecture search (takes a very long time) - consistently has problems associated with saving and loading models in keras. 
* '[autosklearn](https://github.com/automl/auto-sklearn)' - segmentation faults are common, thus archived. If documentation and community improves, may be good to add back in. 
* '[pLDA](https://github.com/RaviSoji/plda)' - this works only for symmetrical images (as it cannot compute eigenvector for many of the feature arrays we have created). For this reason, it is probably not a good idea to use this as a standard training method. 

### 💻 Data modeling notebooks (jupyter)
* Audio file training example
* Text file training example 
* Image file training example
* Video file training example 
* CSV file training example

## Additional documentation
* [Automated Machine Learning tools](https://www.kdnuggets.com/2019/11/github-repo-raider-automated-machine-learning.html)
* [Keras compression](https://github.com/DwangoMediaVillage/keras_compressor)
* [Ludwig](https://uber.github.io/ludwig/examples/#time-series-forecasting)
* [Model compression](https://www.slideshare.net/AnassBensrhirDatasci/deploying-machine-learning-models-to-production)
* [Model compression papers](https://github.com/sun254/awesome-model-compression-and-acceleration)
* [Scikit-small-compression](https://github.com/stewartpark/scikit-small-ensemble)
* [TPOT](https://epistasislab.github.io/tpot/)
* [Voicebook modeling chapter](https://github.com/jim-schwoebel/voicebook/tree/master/chapter_4_modeling)
