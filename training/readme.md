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

## V1 training scripts

There are 21 potential training script settings (customized in the 'settings.json'). Recommended setting is TPOT. 

Note some of the deep learning autoML techniques can take days for optimization, and there are compromises in accuracy vs. speed in training.
* '[auto-pytorch](https://github.com/automl/Auto-PyTorch)' - automated machine learning with the PyTorch framework. &#x2611;

## V2 architecture model training

* '[alphapy](https://alphapy.readthedocs.io/en/latest/user_guide/pipelines.html#model-object-creation)' - highly customizable setttings for data science pipelines/feature selection. &#x2611;
* '[atm](https://github.com/HDI-Project/ATM)' -  give ATM a classification problem and a dataset as a CSV file, and ATM will build the best model it can. &#x2611;
* '[autogbt](https://github.com/pfnet-research/autogbt-alt)' - an experimental Python package that reimplements AutoGBT using LightGBM and Optuna. &#x2611;
* '[autogluon](https://github.com/awslabs/autogluon)' - AutoGluon: AutoML Toolkit for Deep Learning. &#x2611;
* '[autokaggle](https://github.com/datamllab/autokaggle)' - automated ML system trained using gbdt (regression and classification). &#x2611;
* '[btb](https://github.com/HDI-Project/BTB)' - hyperparameter tuning with various ML algorithms in scikit-learn using genetic algorithms. &#x2611;
* '[cvopt](https://github.com/genfifth/cvopt)' - Machine learning parameter search / feature selection module with visualization. &#x2611;
* '[devol](https://github.com/joeddav/devol)' - genetic programming keras cnn layers. &#x2611;
* '[gama](https://github.com/PGijsbers/gama)' - An automated machine learning tool based on genetic programming. &#x2611; 
* '[hungabunga](https://github.com/ypeleg/HungaBunga)' - brute-Force all sklearn models with all parameters using .fit .predict &#x2611;
* '[hyperband](https://github.com/thuijskens/scikit-hyperband)' - implements a class HyperbandSearchCV that works exactly as GridSearchCV and RandomizedSearchCV from scikit-learn do, except that it runs the hyperband algorithm under the hood. &#x2611;
* '[hypsklearn](https://github.com/hyperopt/hyperopt-sklearn)' - seems stable - hyperparameter optimization of the data. &#x2611;
* '[imbalance-learn](https://pypi.org/project/imbalanced-learn/)' - imbalance learn different ML techniques to work on data with different numbers of samples. (could use expanding a bit). &#x2611;
* '[keras](https://keras.io/getting-started/faq/)' - simple MLP network architecture (quick prototype - if works may want to use autoML settings). &#x2611;
* '[ludwig](https://github.com/uber/ludwig)' - deep learning (simple ludwig). - convert every feature to numerical data. &#x2611; 
* '[mlblocks](https://github.com/HDI-Project/MLBlocks)' - most recent framework @ MIT, regression and classification. &#x2611;
* '[neuraxle](https://github.com/Neuraxio/Neuraxle)' - A sklearn-like Framework for Hyperparameter Tuning and AutoML in Deep Learning projects. &#x2611;
* '[safe](https://github.com/ModelOriented/SAFE)' - black box trainer / helps reduce opacity of ML models while increasing accuracy.&#x2611;
* '[scsr](https://github.com/jim-schwoebel/voicebook/blob/master/chapter_4_modeling/train_audioregression.py)' - simple classification / regression (built by Jim from NLX-model). &#x2611;
* **'[tpot](https://epistasislab.github.io/tpot/)'** - TPOT classification / regression (autoML). &#x2611;

## [Metrics](https://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics)
Various [metrics](https://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics) are used based on the problem type. 

Here are the standard metrics calculated for every model trained.

### Classification
See the [Classification metrics section](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics) of the user guide for further details.

- accuracy	sklearn.metrics.accuracy_score
- precision	sklearn.metrics.precision_score
- recall	sklearn.metrics.recall_score
- f1	sklearn.metrics.f1_score (pos_label=1)
- f1Micro	sklearn.metrics.f1_score(average='micro')
- f1Macro	sklearn.metrics.f1_score(average='macro')
- rocAuc	sklearn.metrics.roc_auc_score
- rocAucMicro	sklearn.metrics.roc_auc_score(average='micro')
- rocAucMacro	sklearn.metrics.roc_auc_score(average='macro')

### Regression

See the [Regression metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics) section of the user guide for further details.

- max_error metric calculates the maximum residual error.
- Mean absolute error regression loss
- Mean squared error regression loss
- Median absolute error regression loss
- R^2 (coefficient of determination) regression score function.

## Qualtitative observations 

Rank on these axes:
- ease of setup / future support (setup properly) 
- accuracy on most problems (less accurate --> more accurate) 
- speed of modeling process (slow --> fast) 
- documentation quality (poor --> great) 



```diff
- red
+ green
! orange
# gray
```

```
Metrics = https://scikit-learn.org/stable/modules/model_evaluation.html
from sklearn.metrics import classification_report
```

| Modeling script | License | Accuracy | Speed | Documentation quality | Classification | Regression | 
| --------- |  --------- |  --------- | --------- | --------- | --------- | --------- | 
| alphapy | Apache 2.0 |  5/5 | 4/5 | 5/5 | ✅ | ✅  |
| atm | Apache 2.0 |  5/5 | 4/5 | 5/5 | ✅ | ✅  |
| autogbt | Apache 2.0 |  5/5 | 4/5 | 5/5 | ✅ | ✅  |
| autogluon | Apache 2.0 |  5/5 | 4/5 | 5/5 | ✅ | ✅  |
| autokaggle | Apache 2.0 |  5/5 | 4/5 | 5/5 | ✅ | ✅  |
| autokeras | Apache 2.0 |  5/5 | 4/5 | 5/5 | ✅ | ✅  |
| autopytorch | Apache 2.0 |  5/5 | 4/5 | 5/5 | ✅ | ✅  |
| btb | Apache 2.0 |  5/5 | 4/5 | 5/5 | ✅ | ✅  |
| cvopt | Apache 2.0 |  5/5 | 4/5 | 5/5 | ✅ | ✅  |
| devol | Apache 2.0 |  5/5 | 4/5 | 5/5 | ✅ | ✅  |
| gama | Apache 2.0 |  5/5 | 4/5 | 5/5 | ✅ | ✅  |
| hungabunga | Apache 2.0 |  5/5 | 4/5 | 5/5 | ✅ | ✅  |
| hyperband | Apache 2.0 |  5/5 | 4/5 | 5/5 | ✅ | ✅  |
| hypsklearn | Apache 2.0 |  5/5 | 4/5 | 5/5 | ✅ | ✅  |
| imbalance | Apache 2.0 |  5/5 | 4/5 | 5/5 | ✅ | ✅  |
| keras | Apache 2.0 |  5/5 | 4/5 | 5/5 | ✅ | ✅  |
| ludwig | Apache 2.0 |  5/5 | 4/5 | 5/5 | ✅ | ✅  |
| mlblocks | Apache 2.0 |  5/5 | 4/5 | 5/5 | ✅ | ✅  |
| neuraxle | Apache 2.0 |  5/5 | 4/5 | 5/5 | ❌ | ✅  |
| safe | Apache 2.0 |  5/5 | 4/5 | 5/5 | ✅ | ✅  |
| scsr | Apache 2.0 |  5/5 | 4/5 | 5/5 | ✅ | ✅  |
| [tpot](https://github.com/EpistasisLab/tpot) | LGPL-3.0 |  5/5 | 5/5 | 5/5 | ✅ | ✅  |

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

### 👎 Discarded (due to lack of functionality)

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
