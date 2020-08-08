# Allie 
[![Tweet](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/intent/tweet?text=Check%20out%20Allie,%20an%20awesome%20new%20AutoML%20framework%20@%20https://github.com/jim-schwoebel/allie.&hashtags=machinelearning,automl,deeplearning) [<iframe src="https://ghbtns.com/github-btn.html?user=twbs&repo=bootstrap&type=star&count=true" frameborder="0" scrolling="0" width="150" height="20" title="GitHub"></iframe>](https://github.com/jim-schwoebel/allie)

Allie is a framework for building machine learning models from audio, text, image, video, or .CSV files. 

Intended for both beginners and experts, Allie is designed to be easy-to-use for rapid prototyping and easy-to-extend to your desired modeling strategy.

Here are some things that Allie can do:
- [find](https://github.com/jim-schwoebel/allie/tree/master/datasets) and download datasets (for quick experiments)
- [annotate](https://github.com/jim-schwoebel/allie/tree/master/annotation), [clean](https://github.com/jim-schwoebel/allie/tree/master/cleaning), and/or [augment](https://github.com/jim-schwoebel/allie/tree/master/augmentation) audio, text, image, or video datasets (to prepare data for modeling)
- [featurize](https://github.com/jim-schwoebel/allie/tree/master/features) files using a [standard format](https://github.com/jim-schwoebel/allie/blob/master/features/standard_array.py) (via audio, text, image, video, or csv featurizers)
- [transform](https://github.com/jim-schwoebel/allie/tree/master/preprocessing) features (via scikit-learn preprocessing techniques)
- [visualize](https://github.com/jim-schwoebel/allie/tree/master/visualize) featurized datasets (via yellowbrick, scikit-learn, and matplotlib libraries)
- [train](https://github.com/jim-schwoebel/allie/tree/master/training) classification or regression machine learning models (via tpot, autokeras, autopytorch, ludwig, and 15+ other training scripts)
- [make predictions](https://github.com/jim-schwoebel/allie/tree/master/load_dir) from machine learning models (with all models trained in ./models directory)
- [export data](https://github.com/jim-schwoebel/allie/tree/master/training) in .CSV file formats (for repeatable machine learning experiments across frameworks)
- [compress](https://github.com/jim-schwoebel/allie/tree/master/training) machine learning models for deployment (including repositories with readmes)

![](https://media.giphy.com/media/20NLMBm0BkUOwNljwv/giphy.gif)

You can read more about Allie in the [wiki documentation](https://github.com/jim-schwoebel/allie/wiki).

## getting started (Mac or Linux) 

First, clone the repository:
```
git clone git@github.com:jim-schwoebel/allie.git
cd allie 
```
Set up virtual environment (to ensure consistent operating mode across operating systems).
```
python3 -m pip install --user virtualenv
python3 -m venv env
source env/bin/activate
```
Now install required dependencies and perform unit tests to make sure everything works:
```
python3 setup.py
```

Note the installatin process and unit tests above takes roughly ~10-15 minutes to complete and makes sure that you can featurize, model, and load model files (to make predictions) via your default featurizers and modeling techniques. It may be best to go grab lunch or coffee while waiting. :-)

After everything is done, you can use the [Allie CLI](https://github.com/jim-schwoebel/allie/blob/master/allie.py) by typing in:
```
python3 allie.py -h
```

Which should output some ways you can use Allie with commands in the API:
```
Usage: allie.py [options]

Options:
  -h, --help            show this help message and exit
  --c=command, --command=command
                        the target command (annotate API = 'annotate',
                        augmentation API = 'augment',  cleaning API = 'clean',
                        datasets API = 'data',  features API = 'features',
                        model prediction API = 'predict',  preprocessing API =
                        'transform',  model training API = 'train',  testing
                        API = 'test',  visualize API = 'visualize',
                        list/change default settings = 'settings')
  --p=problemtype, --problemtype=problemtype
                        specify the problem type ('c' = classification or 'r'
                        = regression)
  --s=sampletype, --sampletype=sampletype
                        specify the type files that you'd like to operate on
                        (e.g. 'audio', 'text', 'image', 'video', 'csv')
  --n=common_name, --name=common_name
                        specify the common name for the model (e.g. 'gender'
                        for a male/female problem)
  --i=class_, --class=class_
                        specify the class that you wish to annotate (e.g.
                        'male')
  --d=dir, --dir=dir    an array of the target directory (or directories) that
                        contains sample files for the annotation API,
                        prediction API, features API, augmentation API,
                        cleaning API, and preprocessing API (e.g.
                        '/Users/jim/desktop/allie/train_dir/teens/')
```

For more information on how to use the Allie CLI, check out the [Allie CLI tutorial](https://github.com/jim-schwoebel/allie/wiki/5.-Command-Line-Interface).

## Quick examples

The below examples assume the default [settings.json configuration](https://github.com/jim-schwoebel/allie/blob/master/settings.json) in Allie. Results may vary if you change this file.

### [Collecting data](https://github.com/jim-schwoebel/allie/tree/master/datasets)

To illustrate a quick example, we can pull some sample audio data from this GitHub repository, separating males (x50) from females (x50). 

This [seed_test.py script](https://github.com/jim-schwoebel/allie/blob/master/datasets/seed_test.py) creates two datasets in the train_dir folder, one full of audio files of males and the other full of audio files of females. This data will be used for the rest of the demo sections listed here.

```python3
cd /Users/jim/desktop/allie
cd datasets
python3 seed_test.py
---------------
Cloning into 'sample_voice_data'...
remote: Enumerating objects: 119, done.
remote: Counting objects: 100% (119/119), done.
remote: Compressing objects: 100% (115/115), done.
remote: Total 119 (delta 5), reused 108 (delta 2), pack-reused 0
Receiving objects: 100% (119/119), 18.83 MiB | 7.43 MiB/s, done.
Resolving deltas: 100% (5/5), done.
```
You can easily test if the files are in there with:
```
cd ..
cd train_dir
ls
```
Which should output:
```
jim@Jims-MBP train_dir % ls
README.md		delete_json.py		females
delete_features.py	featurize_csv.py	males
```

Click the .GIF below to follow along this example in a video format:

[![](https://github.com/jim-schwoebel/allie/blob/master/annotation/helpers/assets/collecting.gif)](https://drive.google.com/file/d/1YYniwEJWZFpxTFNwJSGYGP0eeCAgxcvU/view?usp=sharing)

### [Annotating data](https://github.com/jim-schwoebel/allie/tree/master/annotation)

You can simply annotate by typing this into the terminal:

```python3
cd /Users/jim/desktop/allie
cd annotation
python3 annotate.py -d /Users/jim/desktop/allie/train_dir/males/ -s audio -c male -p classification
```

What results is annotated folders in .JSON format following the [standard dictionary](https://github.com/jim-schwoebel/allie/blob/master/features/standard_array.py).

```python3
{"sampletype": "audio", "transcripts": {"audio": {}, "text": {}, "image": {}, "video": {}, "csv": {}}, "features": {"audio": {}, "text": {}, "image": {}, "video": {}, "csv": {}}, "models": {"audio": {}, "text": {}, "image": {}, "video": {}, "csv": {}}, "labels": [{"male": {"value": 1.0, "datetime": "2020-08-03 14:06:53.101763", "filetype": "audio", "file": "1.wav", "problemtype": "classification", "annotate_dir": "/Users/jim/desktop/allie/train_dir/males"}}], "errors": [], "settings": {"version": "1.0.0", "augment_data": false, "balance_data": true, "clean_data": false, "create_csv": true, "default_audio_augmenters": ["augment_tsaug"], "default_audio_cleaners": ["clean_mono16hz"], "default_audio_features": ["librosa_features"], "default_audio_transcriber": ["deepspeech_dict"], "default_csv_augmenters": ["augment_ctgan_regression"], "default_csv_cleaners": ["clean_csv"], "default_csv_features": ["csv_features"], "default_csv_transcriber": ["raw text"], "default_dimensionality_reducer": ["pca"], "default_feature_selector": ["rfe"], "default_image_augmenters": ["augment_imaug"], "default_image_cleaners": ["clean_greyscale"], "default_image_features": ["image_features"], "default_image_transcriber": ["tesseract"], "default_outlier_detector": ["isolationforest"], "default_scaler": ["standard_scaler"], "default_text_augmenters": ["augment_textacy"], "default_text_cleaners": ["remove_duplicates"], "default_text_features": ["nltk_features"], "default_text_transcriber": ["raw text"], "default_training_script": ["tpot"], "default_video_augmenters": ["augment_vidaug"], "default_video_cleaners": ["remove_duplicates"], "default_video_features": ["video_features"], "default_video_transcriber": ["tesseract (averaged over frames)"], "dimension_number": 2, "feature_number": 20, "model_compress": false, "reduce_dimensions": false, "remove_outliers": true, "scale_features": true, "select_features": true, "test_size": 0.1, "transcribe_audio": true, "transcribe_csv": true, "transcribe_image": true, "transcribe_text": true, "transcribe_video": true, "visualize_data": false, "transcribe_videos": true}}
```

After you annotate, you can create [a nicely formatted .CSV](https://github.com/jim-schwoebel/allie/blob/master/annotation/helpers/male_data.csv) for machine learning:

```python3
cd /Users/jim/desktop/allie
cd annotation
python3 create_csv.py -d /Users/jim/desktop/allie/train_dir/males/ -s audio -c male -p classification
```

Click the .GIF below to follow along this example in a video format:

[![](https://github.com/jim-schwoebel/allie/blob/master/annotation/helpers/annotation.gif)](https://drive.google.com/file/d/1Xn7A61XWY8oCAfMmjSMpwEjvItiNp5ev/view?usp=sharing)

### [Augmenting data](https://github.com/jim-schwoebel/allie/tree/master/augmentation)

To augment folders of files, type this into the terminal: 

```python3
cd /Users/jim/desktop/allie
cd augmentation/audio_augmentation
python3 augment.py /Users/jim/desktop/allie/train_dir/males/
python3 augment.py /Users/jim/desktop/allie/train_dir/females/
```

You should now have 2x the data in each folder. Here is a sample audio file and augmented audio file (in females) folder, for reference:
* [Non-augmented file sample](https://drive.google.com/file/d/1kvdoKn0IjBXhBEtjDq9AK8CjD14nIC35/view?usp=sharing) (female speaker)
* [Augmented file sample](https://drive.google.com/file/d/1EsSHx1m_zxrdTjnRMhYKOLLjiKi5gRgY/view?usp=sharing) (female speaker)

Click the .GIF below to follow along this example in a video format:

[![](https://github.com/jim-schwoebel/allie/blob/master/annotation/helpers/assets/augment.gif)](https://drive.google.com/file/d/1j-rGRCgVDifIzoPx3YNuux_k9H-Gb-TD/view?usp=sharing)

### [Cleaning data](https://github.com/jim-schwoebel/allie/tree/master/cleaning)

To clean folders of files, type this into the terminal:

```python3
cd /Users/jim/desktop/allie
cd cleaning/audio_cleaning
python3 clean.py /Users/jim/desktop/allie/train_dir/males/
python3 clean.py /Users/jim/desktop/allie/train_dir/females/
```

Click the .GIF below to follow along this example in a video format:

[![](https://github.com/jim-schwoebel/allie/blob/master/annotation/helpers/assets/clean.gif)](https://drive.google.com/file/d/1gqEHb_3WYFZNnBYdiwJZL--1Aw5KYLUR/view?usp=sharing)

### [Featurizing data](https://github.com/jim-schwoebel/allie/tree/master/features)

To featurize folders of files, type this into the terminal:

```python3
cd /Users/jim/desktop/allie
cd features/audio_features
python3 featurize.py /Users/jim/desktop/allie/train_dir/males/
```

What results are featurized audio files in .JSON format, following the [standard dictionary](https://github.com/jim-schwoebel/allie/blob/master/features/standard_array.py). In this case, librosa_features ahve been added to the audio features dictionary element:
```
{"sampletype": "audio", "transcripts": {"audio": {"deepspeech_dict": "no time to fear with which segnatura"}, "text": {}, "image": {}, "video": {}, "csv": {}}, "features": {"audio": {"librosa_features": {"features": [26.0, 91.6923076923077, 51.970234995079515, 168.0, 3.0, 91.5, 151.99908088235293, 1.4653875809266168, 1.034836765896441, 6.041317273120036, 0.0, 1.1416813305224416, 1.0, 0.0, 1.0, 1.0, 1.0, 0.8034238194742891, 0.011214064753885858, 0.825403850923042, 0.7916342653642267, 0.799591131419195, 0.6655924418356907, 0.011330648308197012, 0.6913809809074571, 0.6541308272200052, 0.661418338298795, 0.6733986003222069, 0.015596609024721262, 0.7120299672460608, 0.6587366189344313, 0.6663189274679151, 0.6465513378063461, 0.025089929067247278, 0.7051369537517457, 0.6222968652297333, 0.6355192254778563, 0.6149336085508313, 0.018343891143310215, 0.6627315206635452, 0.5977577122027029, 0.607582407218563, 0.647661411671485, 0.008459482998415297, 0.6713073144386539, 0.6386408867240191, 0.6452831890619676, 0.699964101062811, 0.007615472147513634, 0.7189362743395604, 0.6881770842137894, 0.6979190256767938, 0.7291497355453966, 0.00940911939072975, 0.7401396411729065, 0.7017699200214637, 0.7331228695983505, 0.7531264693954148, 0.01329521814226255, 0.763276239791762, 0.7116522984632485, 0.7590587373746718, 0.7350729652806612, 0.03153112064046178, 0.7660970571880834, 0.657206917912684, 0.7470611927003584, 0.6322639720282205, 0.017938497171685913, 0.6615311610839943, 0.5941319788201291, 0.6343007353689181, 0.5778206400146562, 0.017037307606750474, 0.6125509247436416, 0.5514687732413751, 0.5754841853855134, -201.3533386630615, 47.15330192057996, -102.18535523700777, -502.49179220012513, -201.31156601864114, 125.25603463854729, 18.082299070656465, 162.0310670688952, 63.261254376704564, 127.81548416353635, -71.79877076064429, 25.856723357072436, 0.00849312238186889, -135.65333506936406, -70.68911674995219, 36.022947214229916, 17.88923654191366, 88.64728797287137, -4.5486056006807685, 32.97698049436967, -43.81875187482729, 14.559312843201848, -11.695474808994188, -92.5127540007665, -43.3715458030925, -2.8686813657134924, 15.41813093478462, 45.52354221029802, -46.72785054356835, -3.558833340504383, -27.922593394058616, 10.927878425731766, -1.4375539667895811, -55.265896216864945, -26.81749586891489, -5.505759781242237, 11.987368006311469, 29.15642000598597, -32.415122640721314, -5.511091433999935, -15.823860504423996, 9.451418989514549, 5.241598458345977, -37.9588985258688, -15.755382749864523, -0.4895271220805938, 9.193396110591832, 21.927443663882503, -17.64191390126784, -1.8424862936634168, -9.022651931057775, 9.128291169735268, 18.549517520727974, -30.5660411806481, -8.34285803735538, -9.278437145276117, 8.646155204335475, 14.76706895840456, -27.949477962308606, -9.825154645141577, -0.5851781562774249, 7.5172603841211965, 15.958013214521532, -19.188487080387258, -1.4245433010743693, -0.0002136159172391847, 0.0001244699595478107, -4.3705042965701695e-06, -0.0006147477962819087, -0.00018150223332853152, 1.798108486545567, 1.0501312811041865, 5.152526911465508, 0.0364132947716482, 1.5099976502325472, 1974.162593266179, 480.2110921639575, 3767.7561717054723, 1207.1282130698876, 1914.9902608895832, 1597.477406458397, 300.1835303257206, 2611.5607852021894, 1096.7664088833956, 1524.2274402971789, 19.140024880982267, 5.679892757483778, 63.887253040125415, 7.787955276296776, 18.524605940498716, 0.00018343908595852554, 0.0003219500940758735, 0.003092152765020728, 9.228861017618328e-06, 7.437301246682182e-05, 3716.331787109375, 970.2260872230195, 6556.8603515625, 1894.921875, 3692.9443359375, 0.11654459635416667, 0.0477755604304587, 0.28662109375, 0.04052734375, 0.1044921875, 0.07000089436769485, 0.042519740760326385, 0.18653172254562378, 0.010408219881355762, 0.06168703734874725], "labels": ["onset_length", "onset_detect_mean", "onset_detect_std", "onset_detect_maxv", "onset_detect_minv", "onset_detect_median", "tempo", "onset_strength_mean", "onset_strength_std", "onset_strength_maxv", "onset_strength_minv", "onset_strength_median", "rhythm_0_mean", "rhythm_0_std", "rhythm_0_maxv", "rhythm_0_minv", "rhythm_0_median", "rhythm_1_mean", "rhythm_1_std", "rhythm_1_maxv", "rhythm_1_minv", "rhythm_1_median", "rhythm_2_mean", "rhythm_2_std", "rhythm_2_maxv", "rhythm_2_minv", "rhythm_2_median", "rhythm_3_mean", "rhythm_3_std", "rhythm_3_maxv", "rhythm_3_minv", "rhythm_3_median", "rhythm_4_mean", "rhythm_4_std", "rhythm_4_maxv", "rhythm_4_minv", "rhythm_4_median", "rhythm_5_mean", "rhythm_5_std", "rhythm_5_maxv", "rhythm_5_minv", "rhythm_5_median", "rhythm_6_mean", "rhythm_6_std", "rhythm_6_maxv", "rhythm_6_minv", "rhythm_6_median", "rhythm_7_mean", "rhythm_7_std", "rhythm_7_maxv", "rhythm_7_minv", "rhythm_7_median", "rhythm_8_mean", "rhythm_8_std", "rhythm_8_maxv", "rhythm_8_minv", "rhythm_8_median", "rhythm_9_mean", "rhythm_9_std", "rhythm_9_maxv", "rhythm_9_minv", "rhythm_9_median", "rhythm_10_mean", "rhythm_10_std", "rhythm_10_maxv", "rhythm_10_minv", "rhythm_10_median", "rhythm_11_mean", "rhythm_11_std", "rhythm_11_maxv", "rhythm_11_minv", "rhythm_11_median", "rhythm_12_mean", "rhythm_12_std", "rhythm_12_maxv", "rhythm_12_minv", "rhythm_12_median", "mfcc_0_mean", "mfcc_0_std", "mfcc_0_maxv", "mfcc_0_minv", "mfcc_0_median", "mfcc_1_mean", "mfcc_1_std", "mfcc_1_maxv", "mfcc_1_minv", "mfcc_1_median", "mfcc_2_mean", "mfcc_2_std", "mfcc_2_maxv", "mfcc_2_minv", "mfcc_2_median", "mfcc_3_mean", "mfcc_3_std", "mfcc_3_maxv", "mfcc_3_minv", "mfcc_3_median", "mfcc_4_mean", "mfcc_4_std", "mfcc_4_maxv", "mfcc_4_minv", "mfcc_4_median", "mfcc_5_mean", "mfcc_5_std", "mfcc_5_maxv", "mfcc_5_minv", "mfcc_5_median", "mfcc_6_mean", "mfcc_6_std", "mfcc_6_maxv", "mfcc_6_minv", "mfcc_6_median", "mfcc_7_mean", "mfcc_7_std", "mfcc_7_maxv", "mfcc_7_minv", "mfcc_7_median", "mfcc_8_mean", "mfcc_8_std", "mfcc_8_maxv", "mfcc_8_minv", "mfcc_8_median", "mfcc_9_mean", "mfcc_9_std", "mfcc_9_maxv", "mfcc_9_minv", "mfcc_9_median", "mfcc_10_mean", "mfcc_10_std", "mfcc_10_maxv", "mfcc_10_minv", "mfcc_10_median", "mfcc_11_mean", "mfcc_11_std", "mfcc_11_maxv", "mfcc_11_minv", "mfcc_11_median", "mfcc_12_mean", "mfcc_12_std", "mfcc_12_maxv", "mfcc_12_minv", "mfcc_12_median", "poly_0_mean", "poly_0_std", "poly_0_maxv", "poly_0_minv", "poly_0_median", "poly_1_mean", "poly_1_std", "poly_1_maxv", "poly_1_minv", "poly_1_median", "spectral_centroid_mean", "spectral_centroid_std", "spectral_centroid_maxv", "spectral_centroid_minv", "spectral_centroid_median", "spectral_bandwidth_mean", "spectral_bandwidth_std", "spectral_bandwidth_maxv", "spectral_bandwidth_minv", "spectral_bandwidth_median", "spectral_contrast_mean", "spectral_contrast_std", "spectral_contrast_maxv", "spectral_contrast_minv", "spectral_contrast_median", "spectral_flatness_mean", "spectral_flatness_std", "spectral_flatness_maxv", "spectral_flatness_minv", "spectral_flatness_median", "spectral_rolloff_mean", "spectral_rolloff_std", "spectral_rolloff_maxv", "spectral_rolloff_minv", "spectral_rolloff_median", "zero_crossings_mean", "zero_crossings_std", "zero_crossings_maxv", "zero_crossings_minv", "zero_crossings_median", "RMSE_mean", "RMSE_std", "RMSE_maxv", "RMSE_minv", "RMSE_median"]}}, "text": {}, "image": {}, "video": {}, "csv": {}}, "models": {"audio": {}, "text": {}, "image": {}, "video": {}, "csv": {}}, "labels": ["females"], "errors": [], "settings": {"version": "1.0.0", "augment_data": false, "balance_data": true, "clean_data": false, "create_csv": true, "default_audio_augmenters": ["augment_tsaug"], "default_audio_cleaners": ["clean_mono16hz"], "default_audio_features": ["librosa_features"], "default_audio_transcriber": ["deepspeech_dict"], "default_csv_augmenters": ["augment_ctgan_regression"], "default_csv_cleaners": ["clean_csv"], "default_csv_features": ["csv_features"], "default_csv_transcriber": ["raw text"], "default_dimensionality_reducer": ["pca"], "default_feature_selector": ["rfe"], "default_image_augmenters": ["augment_imaug"], "default_image_cleaners": ["clean_greyscale"], "default_image_features": ["image_features"], "default_image_transcriber": ["tesseract"], "default_outlier_detector": ["isolationforest"], "default_scaler": ["standard_scaler"], "default_text_augmenters": ["augment_textacy"], "default_text_cleaners": ["remove_duplicates"], "default_text_features": ["nltk_features"], "default_text_transcriber": ["raw text"], "default_training_script": ["tpot"], "default_video_augmenters": ["augment_vidaug"], "default_video_cleaners": ["remove_duplicates"], "default_video_features": ["video_features"], "default_video_transcriber": ["tesseract (averaged over frames)"], "dimension_number": 2, "feature_number": 20, "model_compress": false, "reduce_dimensions": true, "remove_outliers": true, "scale_features": true, "select_features": true, "test_size": 0.1, "transcribe_audio": true, "transcribe_csv": true, "transcribe_image": true, "transcribe_text": true, "transcribe_video": true, "visualize_data": false, "transcribe_videos": true}}
```

Click the .GIF below to follow along this example in a video format:

[![](https://github.com/jim-schwoebel/allie/blob/master/annotation/helpers/assets/featurize.gif)](https://drive.google.com/file/d/1YQXgnAvKHVjSgYteIFz-9H0znr1UKCLl/view?usp=sharing)

### [Transforming data](https://github.com/jim-schwoebel/allie/tree/master/preprocessing)

To transform two folders of files with a classification modeling goal, type this into the terminal:

```python3
cd /Users/jim/desktop/allie
cd preprocessing
python3 transform.py audio c gender males females
```

What results are a few files, with a summary .JSON file (c_gender_standard_scaler_rfe.json) that elaborates upon the transformer in the ./preprocessing/audio_transformer directory:
```
{"estimators": "[('standard_scaler', StandardScaler(copy=True, with_mean=True, with_std=True)), ('rfe', RFE(estimator=SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1,\n                  gamma='scale', kernel='linear', max_iter=-1, shrinking=True,\n                  tol=0.001, verbose=False),\n    n_features_to_select=20, step=1, verbose=0))]", "settings": {"version": "1.0.0", "augment_data": false, "balance_data": true, "clean_data": false, "create_csv": true, "default_audio_augmenters": ["augment_tsaug"], "default_audio_cleaners": ["clean_mono16hz"], "default_audio_features": ["librosa_features"], "default_audio_transcriber": ["deepspeech_dict"], "default_csv_augmenters": ["augment_ctgan_regression"], "default_csv_cleaners": ["clean_csv"], "default_csv_features": ["csv_features"], "default_csv_transcriber": ["raw text"], "default_dimensionality_reducer": ["pca"], "default_feature_selector": ["rfe"], "default_image_augmenters": ["augment_imaug"], "default_image_cleaners": ["clean_greyscale"], "default_image_features": ["image_features"], "default_image_transcriber": ["tesseract"], "default_outlier_detector": ["isolationforest"], "default_scaler": ["standard_scaler"], "default_text_augmenters": ["augment_textacy"], "default_text_cleaners": ["remove_duplicates"], "default_text_features": ["nltk_features"], "default_text_transcriber": ["raw text"], "default_training_script": ["tpot"], "default_video_augmenters": ["augment_vidaug"], "default_video_cleaners": ["remove_duplicates"], "default_video_features": ["video_features"], "default_video_transcriber": ["tesseract (averaged over frames)"], "dimension_number": 2, "feature_number": 20, "model_compress": false, "reduce_dimensions": false, "remove_outliers": true, "scale_features": true, "select_features": true, "test_size": 0.1, "transcribe_audio": true, "transcribe_csv": true, "transcribe_image": true, "transcribe_text": true, "transcribe_video": true, "visualize_data": false, "transcribe_videos": true}, "classes": [0, 1], "sample input X": [7.0, 23.428571428571427, 13.275725891987362, 40.0, 3.0, 29.0, 143.5546875, 0.9958894161283733, 0.548284548384031, 2.9561698164853145, 0.0, 0.9823586435889371, 1.0, 0.0, 1.0, 1.0, 1.0, 0.9563696862586734, 0.004556745090440225, 0.96393999788138, 0.9484953491224131, 0.956450500708107, 0.9159480905407004, 0.0083874210259623, 0.9299425651546245, 0.9015187648369452, 0.9160637212676059, 0.8944452328439548, 0.010436308983995553, 0.9118600181268972, 0.876491746933975, 0.8945884359248597, 0.8770112296385062, 0.012213153676505805, 0.897368483560061, 0.8559758923011408, 0.8771914323071244, 0.8905940594153822, 0.010861457873386392, 0.9086706269489068, 0.871855264076524, 0.8907699627836017, 0.8986618946665619, 0.010095245635220757, 0.9154303336418514, 0.8812084302562171, 0.8988437925782, 0.8940183738781617, 0.01039025341119034, 0.9112831285473728, 0.8760607413483972, 0.8942023162840362, 0.8980899592148434, 0.010095670257151154, 0.9148198685317376, 0.8805923053101389, 0.8982937440953425, 0.888882132228931, 0.01127171631616655, 0.9075145588083179, 0.8692983803153955, 0.8891346978561219, 0.8803505690032647, 0.012156972724147449, 0.9004327206206673, 0.8592151658555779, 0.8806302144175665, 0.8783421648443998, 0.012494808554290946, 0.8989303042965345, 0.8565635990020571, 0.8786581988736766, 0.8633477274070439, 0.013675039594980561, 0.8859283087034104, 0.8395630641775639, 0.8636673866109066, -313.23363896251993, 18.946764320029068, -265.4153881359699, -352.35669009191434, -314.88964335810385, 136.239475379525, 13.46457033057532, 155.28229790095634, 96.67729067600845, 138.31307847807975, -60.109940589659594, 8.90651546650511, -43.00250224341745, -77.22644879310883, -61.59614027580888, 59.959525194997426, 11.49266912690683, 79.92823038661382, 22.593262641790204, 60.14384367187341, -54.39960148805922, 12.978670142489454, -16.69391321594054, -78.18044376664089, -54.04351001558572, 30.023862498118685, 8.714431771268103, 45.984861607171624, -7.969418151448695, 30.779899533210106, -48.79826737490987, 9.404798307829793, -17.32746858770041, -67.85565811008664, -48.67558954047166, 16.438960903373093, 6.676108733267705, 24.75100641115554, -1.8759098025429237, 18.27300445180957, -24.239093865617573, 6.8313516276284245, -9.56759656295116, -40.92277771655667, -24.18878158134608, 3.2516761928923215, 4.2430222382933085, 10.37732827872848, -6.461490621772226, 3.393567465008272, -4.1570109920127685, 5.605424304597271, 5.78957218995748, -18.10767695295411, -3.8190369770110664, -9.46159588572396, 5.81772077466229, 2.7763746636679323, -20.054279810217025, -10.268401482915364, 9.197482271105386, 5.755721680320874, 18.46922506683798, -6.706210697210241, 10.044558505805792, -4.748126927006937e-05, 1.0575334143974938e-05, -2.4722594252240076e-05, -6.952111317908028e-05, -4.773507820227446e-05, 0.40672100602206257, 0.08467855992898438, 0.5757090803234001, 0.22579515457526012, 0.4087367660373401, 2210.951610581014, 212.91019021101542, 2791.2529330926723, 1845.3115106685345, 2223.07457835522, 2063.550185470081, 111.14828141425747, 2287.23164419421, 1816.298268701022, 2073.585928819859, 12.485818860644423, 4.014563343823625, 25.591622605877692, 7.328069768561837, 11.33830589622713, 0.0021384726278483868, 0.004675689153373241, 0.020496303215622902, 0.00027283065719529986, 0.0006564159411936998, 4814.383766867898, 483.0045387722584, 5857.03125, 3682.177734375, 4888.037109375, 0.12172629616477272, 0.0227615872259035, 0.1875, 0.0732421875, 0.1181640625, 0.011859399266541004, 0.0020985447335988283, 0.015743320807814598, 0.006857675965875387, 0.012092416174709797], "sample input Y": 1, "sample transformed X": [1.403390013815334, -0.46826787078753823, -1.1449558304944494, 0.16206820805906216, 0.29026766476533494, 1.121884315543298, 0.6068005729105246, 0.8938752562067679, 1.321370886418425, 0.003118933840524784, 0.9096755549417692, 1.6455655890558802, 0.29214439046692964, 0.47100284419336996, -0.3969793022778641, 0.3178025570312262, -0.5484053895612412, -0.34677929105059246, -0.6946185268998197, -0.6088183224560646], "sample transformed y": 1}
```

Click the .GIF below to follow along this example in a video format:

[![](https://github.com/jim-schwoebel/allie/blob/master/annotation/helpers/assets/transform.gif)](https://drive.google.com/file/d/1wV-yeM9FTRKrGAebhREXpY4gfWPmcL0U/view?usp=sharing)

### [Modeling data](https://github.com/jim-schwoebel/allie/tree/master/training)
#### classification problem

To now model both males and females as a binary gender classification problem, type this into the terminal:

```python3
cd /Users/jim/desktop/allie
cd training
python3 model.py
```

The resulting output in the terminal will be something like:
```
                                                                             
                                                                             
               AAA               lllllll lllllll   iiii                      
              A:::A              l:::::l l:::::l  i::::i                     
             A:::::A             l:::::l l:::::l   iiii                      
            A:::::::A            l:::::l l:::::l                             
           A:::::::::A            l::::l  l::::l iiiiiii     eeeeeeeeeeee    
          A:::::A:::::A           l::::l  l::::l i:::::i   ee::::::::::::ee  
         A:::::A A:::::A          l::::l  l::::l  i::::i  e::::::eeeee:::::ee
        A:::::A   A:::::A         l::::l  l::::l  i::::i e::::::e     e:::::e
       A:::::A     A:::::A        l::::l  l::::l  i::::i e:::::::eeeee::::::e
      A:::::AAAAAAAAA:::::A       l::::l  l::::l  i::::i e:::::::::::::::::e 
     A:::::::::::::::::::::A      l::::l  l::::l  i::::i e::::::eeeeeeeeeee  
    A:::::AAAAAAAAAAAAA:::::A     l::::l  l::::l  i::::i e:::::::e           
   A:::::A             A:::::A   l::::::ll::::::li::::::ie::::::::e          
  A:::::A               A:::::A  l::::::ll::::::li::::::i e::::::::eeeeeeee  
 A:::::A                 A:::::A l::::::ll::::::li::::::i  ee:::::::::::::e  
AAAAAAA                   AAAAAAAlllllllllllllllliiiiiiii    eeeeeeeeeeeeee  
                                                                             
                                                                             
                                                                             
                                                                             
                                                                             
                                                                             
                                                                             

is this a classification (c) or regression (r) problem? 
c
what problem are you solving? (1-audio, 2-text, 3-image, 4-video, 5-csv)
1

 OK cool, we got you modeling audio files 

how many classes would you like to model? (2 available) 
2
these are the available classes: 
['females', 'males']
what is class #1 
males
what is class #2 
females
what is the 1-word common name for the problem you are working on? (e.g. gender for male/female classification) 
gender
-----------------------------------
          LOADING MODULES          
-----------------------------------
Requirement already satisfied: scikit-learn==0.22.2.post1 in /usr/local/lib/python3.7/site-packages (0.22.2.post1)
Requirement already satisfied: numpy>=1.11.0 in /usr/local/lib/python3.7/site-packages (from scikit-learn==0.22.2.post1) (1.18.4)
Requirement already satisfied: scipy>=0.17.0 in /usr/local/lib/python3.7/site-packages (from scikit-learn==0.22.2.post1) (1.4.1)
Requirement already satisfied: joblib>=0.11 in /usr/local/lib/python3.7/site-packages (from scikit-learn==0.22.2.post1) (0.15.1)
-----------------------------------
______ _____  ___ _____ _   _______ _____ ___________ _   _ _____ 
|  ___|  ___|/ _ \_   _| | | | ___ \_   _|___  /_   _| \ | |  __ \
| |_  | |__ / /_\ \| | | | | | |_/ / | |    / /  | | |  \| | |  \/
|  _| |  __||  _  || | | | | |    /  | |   / /   | | | . ` | | __ 
| |   | |___| | | || | | |_| | |\ \ _| |_./ /____| |_| |\  | |_\ \
\_|   \____/\_| |_/\_/  \___/\_| \_|\___/\_____/\___/\_| \_/\____/
                                                                  
                                                                  
______  ___ _____ ___  
|  _  \/ _ \_   _/ _ \ 
| | | / /_\ \| |/ /_\ \
| | | |  _  || ||  _  |
| |/ /| | | || || | | |
|___/ \_| |_/\_/\_| |_/
                       
                       

-----------------------------------
-----------------------------------
           FEATURIZING MALES
-----------------------------------
males: 100%|█████████████████████████████████| 204/204 [00:00<00:00, 432.04it/s]
-----------------------------------
           FEATURIZING FEMALES
-----------------------------------
females: 100%|███████████████████████████████| 204/204 [00:00<00:00, 792.07it/s]
-----------------------------------
 _____ ______ _____  ___ _____ _____ _   _ _____ 
/  __ \| ___ \  ___|/ _ \_   _|_   _| \ | |  __ \
| /  \/| |_/ / |__ / /_\ \| |   | | |  \| | |  \/
| |    |    /|  __||  _  || |   | | | . ` | | __ 
| \__/\| |\ \| |___| | | || |  _| |_| |\  | |_\ \
 \____/\_| \_\____/\_| |_/\_/  \___/\_| \_/\____/
                                                 
                                                 
 ___________  ___  _____ _   _ _____ _   _ _____  ______  ___ _____ ___  
|_   _| ___ \/ _ \|_   _| \ | |_   _| \ | |  __ \ |  _  \/ _ \_   _/ _ \ 
  | | | |_/ / /_\ \ | | |  \| | | | |  \| | |  \/ | | | / /_\ \| |/ /_\ \
  | | |    /|  _  | | | | . ` | | | | . ` | | __  | | | |  _  || ||  _  |
  | | | |\ \| | | |_| |_| |\  |_| |_| |\  | |_\ \ | |/ /| | | || || | | |
  \_/ \_| \_\_| |_/\___/\_| \_/\___/\_| \_/\____/ |___/ \_| |_/\_/\_| |_/
                                                                         
                                                                         

-----------------------------------
-----------------------------------
			REMOVING OUTLIERS
-----------------------------------
<class 'list'>
<class 'int'>
193
11
(204, 187)
(204,)
(193, 187)
(193,)
males greater than minlength (94) by 5, equalizing...
males greater than minlength (94) by 4, equalizing...
males greater than minlength (94) by 3, equalizing...
males greater than minlength (94) by 2, equalizing...
males greater than minlength (94) by 1, equalizing...
males greater than minlength (94) by 0, equalizing...
gender_ALL.CSV
gender_TRAIN.CSV
gender_TEST.CSV
----------------------------------
 ___________  ___   _   _  ___________ ______________  ________ _   _ _____ 
|_   _| ___ \/ _ \ | \ | |/  ___|  ___|  _  | ___ \  \/  |_   _| \ | |  __ \
  | | | |_/ / /_\ \|  \| |\ `--.| |_  | | | | |_/ / .  . | | | |  \| | |  \/
  | | |    /|  _  || . ` | `--. \  _| | | | |    /| |\/| | | | | . ` | | __ 
  | | | |\ \| | | || |\  |/\__/ / |   \ \_/ / |\ \| |  | |_| |_| |\  | |_\ \
  \_/ \_| \_\_| |_/\_| \_/\____/\_|    \___/\_| \_\_|  |_/\___/\_| \_/\____/
                                                                            
                                                                            
______  ___ _____ ___  
|  _  \/ _ \_   _/ _ \ 
| | | / /_\ \| |/ /_\ \
| | | |  _  || ||  _  |
| |/ /| | | || || | | |
|___/ \_| |_/\_/\_| |_/
                       
                       

----------------------------------
Requirement already satisfied: scikit-learn==0.22.2.post1 in /usr/local/lib/python3.7/site-packages (0.22.2.post1)
Requirement already satisfied: joblib>=0.11 in /usr/local/lib/python3.7/site-packages (from scikit-learn==0.22.2.post1) (0.15.1)
Requirement already satisfied: scipy>=0.17.0 in /usr/local/lib/python3.7/site-packages (from scikit-learn==0.22.2.post1) (1.4.1)
Requirement already satisfied: numpy>=1.11.0 in /usr/local/lib/python3.7/site-packages (from scikit-learn==0.22.2.post1) (1.18.4)
making transformer...
python3 transform.py audio c gender  males females
Requirement already satisfied: scikit-learn==0.22.2.post1 in /usr/local/lib/python3.7/site-packages (0.22.2.post1)
Requirement already satisfied: scipy>=0.17.0 in /usr/local/lib/python3.7/site-packages (from scikit-learn==0.22.2.post1) (1.4.1)
Requirement already satisfied: joblib>=0.11 in /usr/local/lib/python3.7/site-packages (from scikit-learn==0.22.2.post1) (0.15.1)
Requirement already satisfied: numpy>=1.11.0 in /usr/local/lib/python3.7/site-packages (from scikit-learn==0.22.2.post1) (1.18.4)
/Users/jim/Desktop/allie
True
False
True
['standard_scaler']
['pca']
['rfe']
['males']
['males', 'females']
----------LOADING MALES----------
100%|███████████████████████████████████████| 102/102 [00:00<00:00, 3633.53it/s]
----------LOADING FEMALES----------
100%|███████████████████████████████████████| 102/102 [00:00<00:00, 3253.67it/s]
[29.0, 115.48275862068965, 61.08593194873256, 222.0, 8.0, 115.0, 135.99917763157896, 2.3062946170604843, 2.4523724313147683, 14.511204587940831, 0.0, 1.5839709225558054, 1.0, 0.0, 1.0, 1.0, 1.0, 0.726785565165924, 0.07679976746065657, 0.8995575540236156, 0.6637538362298933, 0.6773691100707808, 0.5192872926710033, 0.14462205416039903, 0.8356997271918719, 0.39185128881625986, 0.43825294226507827, 0.47576864823687653, 0.15489798102781285, 0.8172667015988133, 0.33745619224839324, 0.3880062539044845, 0.5039810096886804, 0.14210116297622033, 0.8222246027925175, 0.3891757588007075, 0.41656986998926515, 0.5373808945288242, 0.13048669284410083, 0.8278757499272521, 0.42749020156019385, 0.4603386494395968, 0.576968162842806, 0.12481555452009631, 0.8326178822159916, 0.4151756068434901, 0.5352195483012459, 0.5831458034458475, 0.12993864932034976, 0.8408276288260831, 0.4086771989157548, 0.5442120810410158, 0.5991899975086484, 0.12087077223741394, 0.8512938716671109, 0.4582549119597623, 0.548583257282575, 0.6043430612098373, 0.0967011928572277, 0.833318501846635, 0.5289767977628144, 0.5386694936622187, 0.5901015163274542, 0.09036115342542611, 0.8132908481041344, 0.5183061555498681, 0.5348989197039723, 0.5577690938261909, 0.10440904667022494, 0.8051028496830286, 0.47654504571917944, 0.48986432314249784, 0.5222194992298572, 0.12036460062379664, 0.7973796198667512, 0.4203428200207039, 0.4509165396076157, -324.9346584335899, 63.40977505468484, -239.1113566990693, -536.0134915419532, -312.1029273499763, 158.69053618273082, 35.26991926915998, 224.33444977809518, 25.32709426551922, 164.09725638069276, -41.76686834014897, 36.193666229738035, 46.89516424897592, -116.64411005852219, -48.46812935048629, 40.93271834758351, 30.292873365157128, 110.30488966414437, -34.8296992058053, 40.54577852540313, -13.68074476738829, 23.578831857611142, 44.5288579949288, -81.96856948352185, -13.20824575924119, 28.01017666759282, 19.911510776447017, 79.48989729266256, -30.98446467042396, 27.506651161152135, -26.204186150756332, 16.325928650509297, 17.379122853402333, -64.23824041845967, -27.70833256772887, 14.638824890367118, 14.030449436317777, 49.746826625863726, -24.54064068297873, 12.937758655225592, 1.8907564192378423, 12.717800756091705, 32.81143480306558, -34.17480821652823, 3.017798387908008, -9.890548990017422, 11.275154613335049, 16.431256434502732, -41.48821773570883, -10.025098347722025, -2.238522066589343, 11.50921922011025, 25.053143314110734, -36.57309603680529, -2.0110753582118464, -11.28338110558961, 10.092676771445209, 12.359297810934656, -45.39044308667263, -9.744274595029339, 6.634597233918086, 8.23910310866827, 31.12846300160725, -11.374600658849563, 6.9929843274651455, -9.01995244948718e-05, 4.4746520830831e-05, -9.463474948151087e-06, -0.00017975655569362465, -9.088812250730835e-05, 0.7226232345166703, 0.3516279383632571, 1.4144159675282353, 0.082382838783397, 0.7190915579267225, 1451.143003087302, 642.4123068137746, 4547.085433482971, 395.8091024437681, 1324.495426192924, 1453.4773788957211, 426.3531685791114, 2669.3033744664745, 747.1557882330682, 1339.431286902565, 17.70035758898037, 4.253697620372516, 33.22776254607448, 8.885816697236287, 17.70216728565277, 0.00011313906725263223, 0.00018414952501188964, 0.001124512287788093, 5.7439488045929465e-06, 4.929980423185043e-05, 2670.3094482421875, 1335.5639439562065, 6836.7919921875, 355.2978515625, 2282.51953125, 0.07254682268415179, 0.04210112258843493, 0.27783203125, 0.01513671875, 0.062255859375, 0.028742285445332527, 0.011032973416149616, 0.05006047338247299, 0.005435430910438299, 0.029380839318037033]
['onset_length', 'onset_detect_mean', 'onset_detect_std', 'onset_detect_maxv', 'onset_detect_minv', 'onset_detect_median', 'tempo', 'onset_strength_mean', 'onset_strength_std', 'onset_strength_maxv', 'onset_strength_minv', 'onset_strength_median', 'rhythm_0_mean', 'rhythm_0_std', 'rhythm_0_maxv', 'rhythm_0_minv', 'rhythm_0_median', 'rhythm_1_mean', 'rhythm_1_std', 'rhythm_1_maxv', 'rhythm_1_minv', 'rhythm_1_median', 'rhythm_2_mean', 'rhythm_2_std', 'rhythm_2_maxv', 'rhythm_2_minv', 'rhythm_2_median', 'rhythm_3_mean', 'rhythm_3_std', 'rhythm_3_maxv', 'rhythm_3_minv', 'rhythm_3_median', 'rhythm_4_mean', 'rhythm_4_std', 'rhythm_4_maxv', 'rhythm_4_minv', 'rhythm_4_median', 'rhythm_5_mean', 'rhythm_5_std', 'rhythm_5_maxv', 'rhythm_5_minv', 'rhythm_5_median', 'rhythm_6_mean', 'rhythm_6_std', 'rhythm_6_maxv', 'rhythm_6_minv', 'rhythm_6_median', 'rhythm_7_mean', 'rhythm_7_std', 'rhythm_7_maxv', 'rhythm_7_minv', 'rhythm_7_median', 'rhythm_8_mean', 'rhythm_8_std', 'rhythm_8_maxv', 'rhythm_8_minv', 'rhythm_8_median', 'rhythm_9_mean', 'rhythm_9_std', 'rhythm_9_maxv', 'rhythm_9_minv', 'rhythm_9_median', 'rhythm_10_mean', 'rhythm_10_std', 'rhythm_10_maxv', 'rhythm_10_minv', 'rhythm_10_median', 'rhythm_11_mean', 'rhythm_11_std', 'rhythm_11_maxv', 'rhythm_11_minv', 'rhythm_11_median', 'rhythm_12_mean', 'rhythm_12_std', 'rhythm_12_maxv', 'rhythm_12_minv', 'rhythm_12_median', 'mfcc_0_mean', 'mfcc_0_std', 'mfcc_0_maxv', 'mfcc_0_minv', 'mfcc_0_median', 'mfcc_1_mean', 'mfcc_1_std', 'mfcc_1_maxv', 'mfcc_1_minv', 'mfcc_1_median', 'mfcc_2_mean', 'mfcc_2_std', 'mfcc_2_maxv', 'mfcc_2_minv', 'mfcc_2_median', 'mfcc_3_mean', 'mfcc_3_std', 'mfcc_3_maxv', 'mfcc_3_minv', 'mfcc_3_median', 'mfcc_4_mean', 'mfcc_4_std', 'mfcc_4_maxv', 'mfcc_4_minv', 'mfcc_4_median', 'mfcc_5_mean', 'mfcc_5_std', 'mfcc_5_maxv', 'mfcc_5_minv', 'mfcc_5_median', 'mfcc_6_mean', 'mfcc_6_std', 'mfcc_6_maxv', 'mfcc_6_minv', 'mfcc_6_median', 'mfcc_7_mean', 'mfcc_7_std', 'mfcc_7_maxv', 'mfcc_7_minv', 'mfcc_7_median', 'mfcc_8_mean', 'mfcc_8_std', 'mfcc_8_maxv', 'mfcc_8_minv', 'mfcc_8_median', 'mfcc_9_mean', 'mfcc_9_std', 'mfcc_9_maxv', 'mfcc_9_minv', 'mfcc_9_median', 'mfcc_10_mean', 'mfcc_10_std', 'mfcc_10_maxv', 'mfcc_10_minv', 'mfcc_10_median', 'mfcc_11_mean', 'mfcc_11_std', 'mfcc_11_maxv', 'mfcc_11_minv', 'mfcc_11_median', 'mfcc_12_mean', 'mfcc_12_std', 'mfcc_12_maxv', 'mfcc_12_minv', 'mfcc_12_median', 'poly_0_mean', 'poly_0_std', 'poly_0_maxv', 'poly_0_minv', 'poly_0_median', 'poly_1_mean', 'poly_1_std', 'poly_1_maxv', 'poly_1_minv', 'poly_1_median', 'spectral_centroid_mean', 'spectral_centroid_std', 'spectral_centroid_maxv', 'spectral_centroid_minv', 'spectral_centroid_median', 'spectral_bandwidth_mean', 'spectral_bandwidth_std', 'spectral_bandwidth_maxv', 'spectral_bandwidth_minv', 'spectral_bandwidth_median', 'spectral_contrast_mean', 'spectral_contrast_std', 'spectral_contrast_maxv', 'spectral_contrast_minv', 'spectral_contrast_median', 'spectral_flatness_mean', 'spectral_flatness_std', 'spectral_flatness_maxv', 'spectral_flatness_minv', 'spectral_flatness_median', 'spectral_rolloff_mean', 'spectral_rolloff_std', 'spectral_rolloff_maxv', 'spectral_rolloff_minv', 'spectral_rolloff_median', 'zero_crossings_mean', 'zero_crossings_std', 'zero_crossings_maxv', 'zero_crossings_minv', 'zero_crossings_median', 'RMSE_mean', 'RMSE_std', 'RMSE_maxv', 'RMSE_minv', 'RMSE_median']
STANDARD_SCALER
RFE - 20 features
[('standard_scaler', StandardScaler(copy=True, with_mean=True, with_std=True)), ('rfe', RFE(estimator=SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1,
                  gamma='scale', kernel='linear', max_iter=-1, shrinking=True,
                  tol=0.001, verbose=False),
    n_features_to_select=20, step=1, verbose=0))]
21
21
transformed training size
[ 1.40339001 -0.46826787 -1.14495583  0.16206821  0.29026766  1.12188432
  0.60680057  0.89387526  1.32137089  0.00311893  0.90967555  1.64556559
  0.29214439  0.47100284 -0.3969793   0.31780256 -0.54840539 -0.34677929
 -0.69461853 -0.60881832]
/Users/jim/Desktop/allie/preprocessing
c_gender_standard_scaler_rfe.pickle
----------------------------------
-%-$-V-|-%-$-V-|-%-$-V-|-%-$-V-|-%-$-
         TRANSFORMATION           
-%-$-V-|-%-$-V-|-%-$-V-|-%-$-V-|-%-$-
----------------------------------
[7.0, 23.428571428571427, 13.275725891987362, 40.0, 3.0, 29.0, 143.5546875, 0.9958894161283733, 0.548284548384031, 2.9561698164853145, 0.0, 0.9823586435889371, 1.0, 0.0, 1.0, 1.0, 1.0, 0.9563696862586734, 0.004556745090440225, 0.96393999788138, 0.9484953491224131, 0.956450500708107, 0.9159480905407004, 0.0083874210259623, 0.9299425651546245, 0.9015187648369452, 0.9160637212676059, 0.8944452328439548, 0.010436308983995553, 0.9118600181268972, 0.876491746933975, 0.8945884359248597, 0.8770112296385062, 0.012213153676505805, 0.897368483560061, 0.8559758923011408, 0.8771914323071244, 0.8905940594153822, 0.010861457873386392, 0.9086706269489068, 0.871855264076524, 0.8907699627836017, 0.8986618946665619, 0.010095245635220757, 0.9154303336418514, 0.8812084302562171, 0.8988437925782, 0.8940183738781617, 0.01039025341119034, 0.9112831285473728, 0.8760607413483972, 0.8942023162840362, 0.8980899592148434, 0.010095670257151154, 0.9148198685317376, 0.8805923053101389, 0.8982937440953425, 0.888882132228931, 0.01127171631616655, 0.9075145588083179, 0.8692983803153955, 0.8891346978561219, 0.8803505690032647, 0.012156972724147449, 0.9004327206206673, 0.8592151658555779, 0.8806302144175665, 0.8783421648443998, 0.012494808554290946, 0.8989303042965345, 0.8565635990020571, 0.8786581988736766, 0.8633477274070439, 0.013675039594980561, 0.8859283087034104, 0.8395630641775639, 0.8636673866109066, -313.23363896251993, 18.946764320029068, -265.4153881359699, -352.35669009191434, -314.88964335810385, 136.239475379525, 13.46457033057532, 155.28229790095634, 96.67729067600845, 138.31307847807975, -60.109940589659594, 8.90651546650511, -43.00250224341745, -77.22644879310883, -61.59614027580888, 59.959525194997426, 11.49266912690683, 79.92823038661382, 22.593262641790204, 60.14384367187341, -54.39960148805922, 12.978670142489454, -16.69391321594054, -78.18044376664089, -54.04351001558572, 30.023862498118685, 8.714431771268103, 45.984861607171624, -7.969418151448695, 30.779899533210106, -48.79826737490987, 9.404798307829793, -17.32746858770041, -67.85565811008664, -48.67558954047166, 16.438960903373093, 6.676108733267705, 24.75100641115554, -1.8759098025429237, 18.27300445180957, -24.239093865617573, 6.8313516276284245, -9.56759656295116, -40.92277771655667, -24.18878158134608, 3.2516761928923215, 4.2430222382933085, 10.37732827872848, -6.461490621772226, 3.393567465008272, -4.1570109920127685, 5.605424304597271, 5.78957218995748, -18.10767695295411, -3.8190369770110664, -9.46159588572396, 5.81772077466229, 2.7763746636679323, -20.054279810217025, -10.268401482915364, 9.197482271105386, 5.755721680320874, 18.46922506683798, -6.706210697210241, 10.044558505805792, -4.748126927006937e-05, 1.0575334143974938e-05, -2.4722594252240076e-05, -6.952111317908028e-05, -4.773507820227446e-05, 0.40672100602206257, 0.08467855992898438, 0.5757090803234001, 0.22579515457526012, 0.4087367660373401, 2210.951610581014, 212.91019021101542, 2791.2529330926723, 1845.3115106685345, 2223.07457835522, 2063.550185470081, 111.14828141425747, 2287.23164419421, 1816.298268701022, 2073.585928819859, 12.485818860644423, 4.014563343823625, 25.591622605877692, 7.328069768561837, 11.33830589622713, 0.0021384726278483868, 0.004675689153373241, 0.020496303215622902, 0.00027283065719529986, 0.0006564159411936998, 4814.383766867898, 483.0045387722584, 5857.03125, 3682.177734375, 4888.037109375, 0.12172629616477272, 0.0227615872259035, 0.1875, 0.0732421875, 0.1181640625, 0.011859399266541004, 0.0020985447335988283, 0.015743320807814598, 0.006857675965875387, 0.012092416174709797]
-->
[[-1.08260965 -0.98269388 -0.60797492 -0.75483856 -0.81280646 -0.89654763
  -0.2878008  -0.57018752  0.31999349  0.91470661 -0.79709927 -0.39215548
  -0.52523377  0.54936626 -0.85596512  0.88348636  0.96310551  0.00975297
   1.56752723 -0.81022666]]
----------------------------------
gender_ALL_TRANSFORMED.CSV
converting csv...: 100%|█████████████████████| 187/187 [00:00<00:00, 570.70it/s]
     transformed_feature_0  ...  class_
0                -1.109610  ...       0
1                 1.125489  ...       0
2                -1.496244  ...       0
3                -0.971811  ...       0
4                -0.961863  ...       0
..                     ...  ...     ...
182              -1.358248  ...       1
183              -0.376670  ...       1
184              -0.840383  ...       1
185              -0.662551  ...       1
186              -0.580909  ...       1

[187 rows x 21 columns]
writing csv file...
gender_TRAIN_TRANSFORMED.CSV
converting csv...: 100%|█████████████████████| 168/168 [00:00<00:00, 472.53it/s]
     transformed_feature_0  ...  class_
0                 1.378101  ...       0
1                -0.866300  ...       1
2                 1.860016  ...       1
3                -0.124242  ...       1
4                 1.015606  ...       1
..                     ...  ...     ...
163               1.151959  ...       1
164              -0.157868  ...       1
165              -1.179480  ...       0
166              -0.376670  ...       1
167              -0.580909  ...       1

[168 rows x 21 columns]
writing csv file...
gender_TEST_TRANSFORMED.CSV
converting csv...: 100%|███████████████████████| 19/19 [00:00<00:00, 419.95it/s]
    transformed_feature_0  ...  class_
0                0.194916  ...       1
1               -0.818428  ...       0
2               -0.089688  ...       1
3               -0.432771  ...       0
4               -0.457341  ...       0
5               -1.054777  ...       1
6               -0.458814  ...       0
7               -1.011486  ...       1
8                1.028686  ...       1
9                0.753718  ...       1
10              -0.959699  ...       0
11               0.515472  ...       1
12              -1.007477  ...       0
13              -0.552799  ...       1
14              -0.642334  ...       1
15               1.778480  ...       1
16              -0.444939  ...       0
17               0.939541  ...       0
18               1.460974  ...       0

[19 rows x 21 columns]
writing csv file...
----------------------------------
___  ______________ _____ _     _____ _   _ _____  ______  ___ _____ ___  
|  \/  |  _  |  _  \  ___| |   |_   _| \ | |  __ \ |  _  \/ _ \_   _/ _ \ 
| .  . | | | | | | | |__ | |     | | |  \| | |  \/ | | | / /_\ \| |/ /_\ \
| |\/| | | | | | | |  __|| |     | | | . ` | | __  | | | |  _  || ||  _  |
| |  | \ \_/ / |/ /| |___| |_____| |_| |\  | |_\ \ | |/ /| | | || || | | |
\_|  |_/\___/|___/ \____/\_____/\___/\_| \_/\____/ |___/ \_| |_/\_/\_| |_/
                                                                          
                                                                          

----------------------------------
tpot:   0%|                                               | 0/1 [00:00<?, ?it/s]----------------------------------
       .... training TPOT           
----------------------------------
Requirement already satisfied: tpot==0.11.3 in /usr/local/lib/python3.7/site-packages (0.11.3)
Requirement already satisfied: stopit>=1.1.1 in /usr/local/lib/python3.7/site-packages (from tpot==0.11.3) (1.1.2)
Requirement already satisfied: pandas>=0.24.2 in /usr/local/lib/python3.7/site-packages (from tpot==0.11.3) (0.25.3)
Requirement already satisfied: update-checker>=0.16 in /usr/local/lib/python3.7/site-packages (from tpot==0.11.3) (0.17)
Requirement already satisfied: numpy>=1.16.3 in /usr/local/lib/python3.7/site-packages (from tpot==0.11.3) (1.18.4)
Requirement already satisfied: scipy>=1.3.1 in /usr/local/lib/python3.7/site-packages (from tpot==0.11.3) (1.4.1)
Requirement already satisfied: tqdm>=4.36.1 in /usr/local/lib/python3.7/site-packages (from tpot==0.11.3) (4.43.0)
Requirement already satisfied: deap>=1.2 in /usr/local/lib/python3.7/site-packages (from tpot==0.11.3) (1.3.1)
Requirement already satisfied: scikit-learn>=0.22.0 in /usr/local/lib/python3.7/site-packages (from tpot==0.11.3) (0.22.2.post1)
Requirement already satisfied: joblib>=0.13.2 in /usr/local/lib/python3.7/site-packages (from tpot==0.11.3) (0.15.1)
Requirement already satisfied: python-dateutil>=2.6.1 in /usr/local/lib/python3.7/site-packages (from pandas>=0.24.2->tpot==0.11.3) (2.8.1)
Requirement already satisfied: pytz>=2017.2 in /usr/local/lib/python3.7/site-packages (from pandas>=0.24.2->tpot==0.11.3) (2020.1)
Requirement already satisfied: requests>=2.3.0 in /usr/local/lib/python3.7/site-packages (from update-checker>=0.16->tpot==0.11.3) (2.24.0)
Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.7/site-packages (from python-dateutil>=2.6.1->pandas>=0.24.2->tpot==0.11.3) (1.15.0)
Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.7/site-packages (from requests>=2.3.0->update-checker>=0.16->tpot==0.11.3) (2020.4.5.2)
Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.7/site-packages (from requests>=2.3.0->update-checker>=0.16->tpot==0.11.3) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.7/site-packages (from requests>=2.3.0->update-checker>=0.16->tpot==0.11.3) (2.9)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.7/site-packages (from requests>=2.3.0->update-checker>=0.16->tpot==0.11.3) (1.25.9)
Warning: xgboost.XGBClassifier is not available and will not be used by TPOT.
Generation 1 - Current best internal CV score: 0.7976827094474153               
Generation 2 - Current best internal CV score: 0.8096256684491978               
Generation 3 - Current best internal CV score: 0.8096256684491978               
Generation 4 - Current best internal CV score: 0.8096256684491978               
Generation 5 - Current best internal CV score: 0.8096256684491978               
Generation 6 - Current best internal CV score: 0.8215686274509804               
Generation 7 - Current best internal CV score: 0.8215686274509804               
Generation 8 - Current best internal CV score: 0.8219251336898395               
Generation 9 - Current best internal CV score: 0.8219251336898395               
Generation 10 - Current best internal CV score: 0.8276292335115866              
tpot:   0%|                                               | 0/1 [03:58<?, ?it/s]
Best pipeline: LinearSVC(Normalizer(input_matrix, norm=max), C=20.0, dual=True, loss=hinge, penalty=l2, tol=0.0001)

/usr/local/lib/python3.7/site-packages/sklearn/svm/_base.py:947: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  "the number of iterations.", ConvergenceWarning)
saving classifier to disk
[1 0 1 0 0 1 0 1 1 0 0 1 1 1 1 1 0 0 0]

Normalized confusion matrix
error making y_probas
error plotting ROC curve
predict_proba only works for or log loss and modified Huber loss.
tpot: 100%|██████████████████████████████████████| 1/1 [04:08<00:00, 248.61s/it]
```

The result will be a [GitHub repo like this](https://github.com/jim-schwoebel/allie/tree/master/training/helpers/gender_tpot_classifier), defining the model session and summary. Accuracy metrics will be defined as part of the model training process:

```
{'accuracy': 0.8947368421052632, 'balanced_accuracy': 0.8944444444444444, 'precision': 0.9, 'recall': 0.9, 'f1_score': 0.9, 'f1_micro': 0.8947368421052632, 'f1_macro': 0.8944444444444444, 'roc_auc': 0.8944444444444444, 'roc_auc_micro': 0.8944444444444444, 'roc_auc_macro': 0.8944444444444444, 'confusion_matrix': [[8, 1], [1, 9]], 'classification_report': '              precision    recall  f1-score   support\n\n       males       0.89      0.89      0.89         9\n     females       0.90      0.90      0.90        10\n\n    accuracy                           0.89        19\n   macro avg       0.89      0.89      0.89        19\nweighted avg       0.89      0.89      0.89        19\n'}
```

Click the .GIF below to follow along this example in a video format:

[![](https://github.com/jim-schwoebel/allie/blob/master/annotation/helpers/assets/classification.gif)](https://drive.google.com/file/d/1x6gGl6Ag4HjT3MKs6kwz_0gq-Kk8peZA/view?usp=sharing)

#### regression problem

To model a regression problem, you need a .CSV file with annotations and files. It is best to produce these files with the annotate.py script, but you can also use datasets created within Allie for regression modeling.

All you need to do is follow the similar steps for modeling, specifying a regression target and the .CSV file of interest:
```
python3 model.py
                                                                             
                                                                             
               AAA               lllllll lllllll   iiii                      
              A:::A              l:::::l l:::::l  i::::i                     
             A:::::A             l:::::l l:::::l   iiii                      
            A:::::::A            l:::::l l:::::l                             
           A:::::::::A            l::::l  l::::l iiiiiii     eeeeeeeeeeee    
          A:::::A:::::A           l::::l  l::::l i:::::i   ee::::::::::::ee  
         A:::::A A:::::A          l::::l  l::::l  i::::i  e::::::eeeee:::::ee
        A:::::A   A:::::A         l::::l  l::::l  i::::i e::::::e     e:::::e
       A:::::A     A:::::A        l::::l  l::::l  i::::i e:::::::eeeee::::::e
      A:::::AAAAAAAAA:::::A       l::::l  l::::l  i::::i e:::::::::::::::::e 
     A:::::::::::::::::::::A      l::::l  l::::l  i::::i e::::::eeeeeeeeeee  
    A:::::AAAAAAAAAAAAA:::::A     l::::l  l::::l  i::::i e:::::::e           
   A:::::A             A:::::A   l::::::ll::::::li::::::ie::::::::e          
  A:::::A               A:::::A  l::::::ll::::::li::::::i e::::::::eeeeeeee  
 A:::::A                 A:::::A l::::::ll::::::li::::::i  ee:::::::::::::e  
AAAAAAA                   AAAAAAAlllllllllllllllliiiiiiii    eeeeeeeeeeeeee  
                                                                             
                                                                             
                                                                             
                                                                             
                                                                             
                                                                             
                                                                             

is this a classification (c) or regression (r) problem? 
r
what is the name of the spreadsheet (in ./train_dir) used for prediction? 

 available: ['gender_all.csv']

gender_all.csv
how many classes would you like to model? (188 available) 
1
these are the available classes: ['onset_length', 'onset_detect_mean', 'onset_detect_std', 'onset_detect_maxv', 'onset_detect_minv', 'onset_detect_median', 'tempo', 'onset_strength_mean', 'onset_strength_std', 'onset_strength_maxv', 'onset_strength_minv', 'onset_strength_median', 'rhythm_0_mean', 'rhythm_0_std', 'rhythm_0_maxv', 'rhythm_0_minv', 'rhythm_0_median', 'rhythm_1_mean', 'rhythm_1_std', 'rhythm_1_maxv', 'rhythm_1_minv', 'rhythm_1_median', 'rhythm_2_mean', 'rhythm_2_std', 'rhythm_2_maxv', 'rhythm_2_minv', 'rhythm_2_median', 'rhythm_3_mean', 'rhythm_3_std', 'rhythm_3_maxv', 'rhythm_3_minv', 'rhythm_3_median', 'rhythm_4_mean', 'rhythm_4_std', 'rhythm_4_maxv', 'rhythm_4_minv', 'rhythm_4_median', 'rhythm_5_mean', 'rhythm_5_std', 'rhythm_5_maxv', 'rhythm_5_minv', 'rhythm_5_median', 'rhythm_6_mean', 'rhythm_6_std', 'rhythm_6_maxv', 'rhythm_6_minv', 'rhythm_6_median', 'rhythm_7_mean', 'rhythm_7_std', 'rhythm_7_maxv', 'rhythm_7_minv', 'rhythm_7_median', 'rhythm_8_mean', 'rhythm_8_std', 'rhythm_8_maxv', 'rhythm_8_minv', 'rhythm_8_median', 'rhythm_9_mean', 'rhythm_9_std', 'rhythm_9_maxv', 'rhythm_9_minv', 'rhythm_9_median', 'rhythm_10_mean', 'rhythm_10_std', 'rhythm_10_maxv', 'rhythm_10_minv', 'rhythm_10_median', 'rhythm_11_mean', 'rhythm_11_std', 'rhythm_11_maxv', 'rhythm_11_minv', 'rhythm_11_median', 'rhythm_12_mean', 'rhythm_12_std', 'rhythm_12_maxv', 'rhythm_12_minv', 'rhythm_12_median', 'mfcc_0_mean', 'mfcc_0_std', 'mfcc_0_maxv', 'mfcc_0_minv', 'mfcc_0_median', 'mfcc_1_mean', 'mfcc_1_std', 'mfcc_1_maxv', 'mfcc_1_minv', 'mfcc_1_median', 'mfcc_2_mean', 'mfcc_2_std', 'mfcc_2_maxv', 'mfcc_2_minv', 'mfcc_2_median', 'mfcc_3_mean', 'mfcc_3_std', 'mfcc_3_maxv', 'mfcc_3_minv', 'mfcc_3_median', 'mfcc_4_mean', 'mfcc_4_std', 'mfcc_4_maxv', 'mfcc_4_minv', 'mfcc_4_median', 'mfcc_5_mean', 'mfcc_5_std', 'mfcc_5_maxv', 'mfcc_5_minv', 'mfcc_5_median', 'mfcc_6_mean', 'mfcc_6_std', 'mfcc_6_maxv', 'mfcc_6_minv', 'mfcc_6_median', 'mfcc_7_mean', 'mfcc_7_std', 'mfcc_7_maxv', 'mfcc_7_minv', 'mfcc_7_median', 'mfcc_8_mean', 'mfcc_8_std', 'mfcc_8_maxv', 'mfcc_8_minv', 'mfcc_8_median', 'mfcc_9_mean', 'mfcc_9_std', 'mfcc_9_maxv', 'mfcc_9_minv', 'mfcc_9_median', 'mfcc_10_mean', 'mfcc_10_std', 'mfcc_10_maxv', 'mfcc_10_minv', 'mfcc_10_median', 'mfcc_11_mean', 'mfcc_11_std', 'mfcc_11_maxv', 'mfcc_11_minv', 'mfcc_11_median', 'mfcc_12_mean', 'mfcc_12_std', 'mfcc_12_maxv', 'mfcc_12_minv', 'mfcc_12_median', 'poly_0_mean', 'poly_0_std', 'poly_0_maxv', 'poly_0_minv', 'poly_0_median', 'poly_1_mean', 'poly_1_std', 'poly_1_maxv', 'poly_1_minv', 'poly_1_median', 'spectral_centroid_mean', 'spectral_centroid_std', 'spectral_centroid_maxv', 'spectral_centroid_minv', 'spectral_centroid_median', 'spectral_bandwidth_mean', 'spectral_bandwidth_std', 'spectral_bandwidth_maxv', 'spectral_bandwidth_minv', 'spectral_bandwidth_median', 'spectral_contrast_mean', 'spectral_contrast_std', 'spectral_contrast_maxv', 'spectral_contrast_minv', 'spectral_contrast_median', 'spectral_flatness_mean', 'spectral_flatness_std', 'spectral_flatness_maxv', 'spectral_flatness_minv', 'spectral_flatness_median', 'spectral_rolloff_mean', 'spectral_rolloff_std', 'spectral_rolloff_maxv', 'spectral_rolloff_minv', 'spectral_rolloff_median', 'zero_crossings_mean', 'zero_crossings_std', 'zero_crossings_maxv', 'zero_crossings_minv', 'zero_crossings_median', 'RMSE_mean', 'RMSE_std', 'RMSE_maxv', 'RMSE_minv', 'RMSE_median', 'class_']
what is class #1 
class_
what is the 1-word common name for the problem you are working on? (e.g. gender for male/female classification) 
gender
```
It will then output the regression model in the proper folder ([like this](https://github.com/jim-schwoebel/allie/tree/master/training/helpers/gender_tpot_regression)), using TPOT as a model trainer. You will also get some awesome stats on the regression modeling sesssion, like in the [.JSON file below](https://github.com/jim-schwoebel/allie/blob/master/training/helpers/gender_tpot_regression/model/gender_tpot_regression.json):

```
{"sample type": "csv", "created date": "2020-08-03 15:29:43.786976", "device info": {"time": "2020-08-03 15:29", "timezone": ["EST", "EDT"], "operating system": "Darwin", "os release": "19.5.0", "os version": "Darwin Kernel Version 19.5.0: Tue May 26 20:41:44 PDT 2020; root:xnu-6153.121.2~2/RELEASE_X86_64", "cpu data": {"memory": [8589934592, 2577022976, 70.0, 4525428736, 107941888, 2460807168, 2122092544, 2064621568], "cpu percent": 59.1, "cpu times": [22612.18, 0.0, 12992.38, 102624.04], "cpu count": 4, "cpu stats": [110955, 504058, 130337047, 518089], "cpu swap": [2147483648, 1096548352, 1050935296, 51.1, 44743286784, 329093120], "partitions": [["/dev/disk1s6", "/", "apfs", "ro,local,rootfs,dovolfs,journaled,multilabel"], ["/dev/disk1s5", "/System/Volumes/Data", "apfs", "rw,local,dovolfs,dontbrowse,journaled,multilabel"], ["/dev/disk1s4", "/private/var/vm", "apfs", "rw,local,dovolfs,dontbrowse,journaled,multilabel"], ["/dev/disk1s1", "/Volumes/Macintosh HD - Data", "apfs", "rw,local,dovolfs,journaled,multilabel"]], "disk usage": [499963174912, 10985529344, 317145075712, 3.3], "disk io counters": [1689675, 1773144, 52597518336, 34808844288, 1180797, 1136731], "battery": [100, -2, true], "boot time": 1596411904.0}, "space left": 317.145075712}, "session id": "fc54dd66-d5bc-11ea-9c75-acde48001122", "classes": ["class_"], "problem type": "regression", "model name": "gender_tpot_regression.pickle", "model type": "tpot", "metrics": {"mean_absolute_error": 0.37026379788606023, "mean_squared_error": 0.16954440031335424, "median_absolute_error": 0.410668441980656, "r2_score": 0.3199385720764347}, "settings": {"version": "1.0.0", "augment_data": false, "balance_data": true, "clean_data": false, "create_csv": true, "default_audio_augmenters": ["augment_tsaug"], "default_audio_cleaners": ["clean_mono16hz"], "default_audio_features": ["librosa_features"], "default_audio_transcriber": ["deepspeech_dict"], "default_csv_augmenters": ["augment_ctgan_regression"], "default_csv_cleaners": ["clean_csv"], "default_csv_features": ["csv_features"], "default_csv_transcriber": ["raw text"], "default_dimensionality_reducer": ["pca"], "default_feature_selector": ["rfe"], "default_image_augmenters": ["augment_imaug"], "default_image_cleaners": ["clean_greyscale"], "default_image_features": ["image_features"], "default_image_transcriber": ["tesseract"], "default_outlier_detector": ["isolationforest"], "default_scaler": ["standard_scaler"], "default_text_augmenters": ["augment_textacy"], "default_text_cleaners": ["remove_duplicates"], "default_text_features": ["nltk_features"], "default_text_transcriber": ["raw text"], "default_training_script": ["tpot"], "default_video_augmenters": ["augment_vidaug"], "default_video_cleaners": ["remove_duplicates"], "default_video_features": ["video_features"], "default_video_transcriber": ["tesseract (averaged over frames)"], "dimension_number": 2, "feature_number": 20, "model_compress": false, "reduce_dimensions": false, "remove_outliers": true, "scale_features": false, "select_features": false, "test_size": 0.1, "transcribe_audio": false, "transcribe_csv": true, "transcribe_image": true, "transcribe_text": true, "transcribe_video": true, "transcribe_videos": true, "visualize_data": false, "default_dimensionionality_reducer": ["pca"]}, "transformer name": "", "training data": [], "sample X_test": [30.0, 116.1, 68.390715744171, 224.0, 3.0, 115.5, 129.19921875, 1.579895074162117, 1.4053805862299766, 6.915237601339313, 0.0, 1.1654598038099069, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8033179369485901, 0.00438967342343324, 0.8140795129312649, 0.7979309783326958, 0.80255447893579, 0.5772101965904585, 0.025367026843705915, 0.6147904436358145, 0.5452462503889344, 0.5720709525572024, 0.5251607032640779, 0.031273364291655614, 0.5651684602891733, 0.4833782607526296, 0.522481114581999, 0.53067387207457, 0.01636309315550051, 0.5760527497162795, 0.5083941678429416, 0.5308772078223155, 0.5383483269837346, 0.02398538849569036, 0.6138641187358237, 0.5148823529890311, 0.5317355191905834, 0.5590921868458475, 0.018941050706796927, 0.6185565218733067, 0.5391848127954322, 0.5515129204797803, 0.5653692033981255, 0.022886171192539908, 0.6170498591126126, 0.5187020777516459, 0.5693268285980656, 0.5428369240411614, 0.011543007163874491, 0.5837123204211986, 0.5208221399174541, 0.5415414663324902, 0.4946660644711973, 0.021472694373470352, 0.5215764169994959, 0.4640787039752625, 0.4952267598817138, 0.4798469011394895, 0.02593484469896265, 0.5172960598832023, 0.4449712627305569, 0.4777149108114186, 0.4993938744598669, 0.01849048457494309, 0.5651910299787914, 0.4822436630327371, 0.4950261489562563, 0.5363930497563161, 0.0376443504751349, 0.6330907702118795, 0.4816294954352716, 0.5249507027509328, -235.4678661326307, 61.51638081120653, -119.29458629496251, -362.1632462796749, -227.60500825042942, 163.92070611988834, 47.05955903012367, 237.9764586528294, 41.986380826321785, 172.71493170004138, 9.237411399943188, 25.868443694231683, 61.477039729510096, -75.39528620218707, 9.629797757209056, 38.85787728431835, 25.651975918739637, 120.33667371104372, -9.003575689525233, 36.13886469019118, -3.813926397129359, 18.466559976322753, 45.395818864794386, -54.58126572108478, -3.563646356257889, 28.49882430361086, 15.286105184256387, 72.2886732962803, 0.03239718043784112, 26.491533722920998, -19.866746887564343, 16.46528562102129, 9.928420130258688, -61.42422346209003, -17.134010559191154, 4.917765483447672, 13.106589177321654, 36.30054941946764, -28.88492762419697, 4.470641784765922, -7.5214435695300805, 11.456845078656613, 24.68530842159717, -33.23468909518539, -7.800944005694487, 1.7653313822916499, 10.137823325108423, 26.38688279047729, -22.507646864346647, 2.1230603462314384, 2.9722994596741263, 9.920580299259306, 29.09083383516883, -28.462312178142557, 3.1356694281534625, -8.31659816437322, 9.321735116288234, 14.977416272339756, -29.19924207526083, -7.200232618719922, 10.020856138237773, 9.605360863583002, 33.70453001221575, -10.34310153320585, 8.538943192527702, -0.0003117740953455404, 0.0002093530273784296, -3.649852038234921e-05, -0.0008609846033373115, -0.00024944132582088046, 2.427670449088513, 1.573081810523066, 6.574603060966783, 0.2961628052414745, 1.8991203106986, 1122.5579040699354, 895.7957759390358, 4590.354474064802, 349.53842801686966, 800.0437543350607, 1384.7323846043691, 519.4846094956321, 2642.151716668925, 703.4646482237979, 1229.7584170111122, 22.097758701059746, 6.005214057147793, 54.922406822231686, 8.895233246285754, 22.047151155860252, 6.146541272755712e-05, 0.00013457647582981735, 0.0006881643203087151, 5.692067475138174e-07, 8.736528798181098e-06, 2087.3572470319323, 1731.5818839146564, 6535.3271484375, 409.130859375, 1421.19140625, 0.05515445892467249, 0.07680443213522453, 0.46142578125, 0.0078125, 0.0302734375, 0.12412750720977785, 0.07253565639257431, 0.29952874779701233, 0.010528072714805605, 0.10663044452667236], "sample y_test": 0}
```

Click the .GIF below to follow along this example in a video format:

[![](https://github.com/jim-schwoebel/allie/blob/master/annotation/helpers/assets/regression.gif)](https://drive.google.com/file/d/1PQwBABSRKrzS67IrlgvFSjuRBKFFx-XX/view?usp=sharing)

### [Making predictions](https://github.com/jim-schwoebel/allie/tree/master/training)
To make predictions using the machine learning model that you just trained, type this into the terminal:

```python3
cd /Users/jim/desktop/allie
cd models
python3 load.py
```

What results is featurized folders with model predictions in .JSON format following the [standard dictionary](https://github.com/jim-schwoebel/allie/blob/master/features/standard_array.py), as shown below:

```
{"sampletype": "audio", "transcripts": {"audio": {"deepspeech_dict": "this is some testator testing once you three i am testing for ale while machinery framework to see his accretion the new sample test"}, "text": {}, "image": {}, "video": {}, "csv": {}}, "features": {"audio": {"librosa_features": {"features": [48.0, 204.70833333333334, 114.93240011947118, 396.0, 14.0, 216.5, 103.359375, 1.8399430940147428, 1.9668646890049444, 11.385597904924593, 0.0, 1.0559244294941723, 1.0, 0.0, 1.0, 1.0, 1.0, 0.7548547717034835, 0.010364651324331484, 0.7781876333922462, 0.7413898270731765, 0.7531146388573189, 0.5329470301378001, 0.017993121900389288, 0.5625446261639591, 0.49812177879164954, 0.5362475050227643, 0.5019325152826378, 0.014070606086479512, 0.523632811868841, 0.46894673564752315, 0.5016733890463346, 0.47905132092405744, 0.02472744944944913, 0.5170032241543769, 0.408032583763636, 0.481510183943566, 0.47211244429901056, 0.018043067999864118, 0.4986083755424441, 0.4084943475419884, 0.47487594610615297, 0.4698145764425497, 0.02481760404009747, 0.5249353704873062, 0.4293713399428569, 0.4722612320098678, 0.45261508487773017, 0.026497663310172545, 0.49848564789957234, 0.40880741566998524, 0.4507269108715201, 0.4486803104555413, 0.058559460166888094, 0.5292193402660791, 0.3401144705267203, 0.44999945618536435, 0.47497774707770735, 0.06545659069313127, 0.5736851049778624, 0.37925421500129, 0.4734915617563768, 0.4650337731799947, 0.06320658864729298, 0.5675856011170606, 0.3828128296325481, 0.45491284769941215, 0.4336677569640048, 0.06580364398487831, 0.5229786825561087, 0.3254973876934075, 0.4435446804048719, 0.4510229261935718, 0.0716424867984984, 0.5607997826027251, 0.3319068941555564, 0.45336899240905365, -378.4693712461592, 123.45005738361948, -131.02074973363048, -645.6119532302674, -365.0407612849682, 108.01722016743142, 78.5850621057939, 244.19279156346005, -109.89544987268641, 113.87757464191944, -18.990339871317058, 38.97227759803155, 80.46313291288668, -113.14922433281748, -19.5478460234633, 25.85348830525823, 36.66801973350443, 140.72102808980202, -59.74682246793187, 18.3196627309548, 25.890819294565695, 28.110070916600474, 109.71209190044716, -32.50655086525428, 24.126562365562382, -12.77779324195114, 25.980150189124338, 37.34024720564918, -89.18596268298815, -14.092855104596493, -14.213402047550273, 17.851386217883952, 24.416921204215857, -53.80916251929509, -15.616460366626296, -11.056262156053059, 18.131479541957944, 25.019042211813467, -65.95011982036516, -10.115261093647717, -2.111560667454096, 11.800353875032327, 33.815281150727785, -35.047615612670526, -2.4632489045982657, -12.855548041442455, 13.841955462451525, 26.49950045235625, -54.65905146286438, -12.258563565004795, -5.991988010947961, 11.560727147262314, 26.699383611419385, -46.86210002294128, -5.08389478450145, -11.905883972886778, 13.110884275285521, 18.96898208296976, -55.222181120197234, -8.889351847151506, -13.282554457300717, 9.363802595261776, 13.125079504552438, -42.40351688080857, -12.904730673116855, -6.647081175227956e-05, 6.790962221819154e-05, 3.898538767970233e-05, -0.0003530532719088282, -5.176161063821292e-05, 0.5775604310470552, 0.5363262958114443, 3.0171051694951547, 0.005876029108677461, 0.447613631105005, 2196.6402149427804, 1460.1082170800585, 6848.122696727527, 474.45532202867423, 1779.7575344580457, 1879.6573011499802, 758.0548156953982, 3968.436183431614, 710.7057371268927, 1783.9133839417857, 25.057721821734972, 7.417488037600184, 48.54069273302066, 7.980294433517432, 26.382808285840404, 0.02705797180533409, 0.049401603639125824, 0.22989588975906372, 2.3204531316878274e-05, 0.0016842428594827652, 3896.30511090472, 2618.9936438064337, 9829.9072265625, 484.4970703125, 2993.115234375, 0.13837594696969696, 0.11062751539644003, 0.62060546875, 0.01220703125, 0.10009765625, 0.025540588423609734, 0.02010413259267807, 0.09340725094079971, 0.00015651443391107023, 0.02306547947227955], "labels": ["onset_length", "onset_detect_mean", "onset_detect_std", "onset_detect_maxv", "onset_detect_minv", "onset_detect_median", "tempo", "onset_strength_mean", "onset_strength_std", "onset_strength_maxv", "onset_strength_minv", "onset_strength_median", "rhythm_0_mean", "rhythm_0_std", "rhythm_0_maxv", "rhythm_0_minv", "rhythm_0_median", "rhythm_1_mean", "rhythm_1_std", "rhythm_1_maxv", "rhythm_1_minv", "rhythm_1_median", "rhythm_2_mean", "rhythm_2_std", "rhythm_2_maxv", "rhythm_2_minv", "rhythm_2_median", "rhythm_3_mean", "rhythm_3_std", "rhythm_3_maxv", "rhythm_3_minv", "rhythm_3_median", "rhythm_4_mean", "rhythm_4_std", "rhythm_4_maxv", "rhythm_4_minv", "rhythm_4_median", "rhythm_5_mean", "rhythm_5_std", "rhythm_5_maxv", "rhythm_5_minv", "rhythm_5_median", "rhythm_6_mean", "rhythm_6_std", "rhythm_6_maxv", "rhythm_6_minv", "rhythm_6_median", "rhythm_7_mean", "rhythm_7_std", "rhythm_7_maxv", "rhythm_7_minv", "rhythm_7_median", "rhythm_8_mean", "rhythm_8_std", "rhythm_8_maxv", "rhythm_8_minv", "rhythm_8_median", "rhythm_9_mean", "rhythm_9_std", "rhythm_9_maxv", "rhythm_9_minv", "rhythm_9_median", "rhythm_10_mean", "rhythm_10_std", "rhythm_10_maxv", "rhythm_10_minv", "rhythm_10_median", "rhythm_11_mean", "rhythm_11_std", "rhythm_11_maxv", "rhythm_11_minv", "rhythm_11_median", "rhythm_12_mean", "rhythm_12_std", "rhythm_12_maxv", "rhythm_12_minv", "rhythm_12_median", "mfcc_0_mean", "mfcc_0_std", "mfcc_0_maxv", "mfcc_0_minv", "mfcc_0_median", "mfcc_1_mean", "mfcc_1_std", "mfcc_1_maxv", "mfcc_1_minv", "mfcc_1_median", "mfcc_2_mean", "mfcc_2_std", "mfcc_2_maxv", "mfcc_2_minv", "mfcc_2_median", "mfcc_3_mean", "mfcc_3_std", "mfcc_3_maxv", "mfcc_3_minv", "mfcc_3_median", "mfcc_4_mean", "mfcc_4_std", "mfcc_4_maxv", "mfcc_4_minv", "mfcc_4_median", "mfcc_5_mean", "mfcc_5_std", "mfcc_5_maxv", "mfcc_5_minv", "mfcc_5_median", "mfcc_6_mean", "mfcc_6_std", "mfcc_6_maxv", "mfcc_6_minv", "mfcc_6_median", "mfcc_7_mean", "mfcc_7_std", "mfcc_7_maxv", "mfcc_7_minv", "mfcc_7_median", "mfcc_8_mean", "mfcc_8_std", "mfcc_8_maxv", "mfcc_8_minv", "mfcc_8_median", "mfcc_9_mean", "mfcc_9_std", "mfcc_9_maxv", "mfcc_9_minv", "mfcc_9_median", "mfcc_10_mean", "mfcc_10_std", "mfcc_10_maxv", "mfcc_10_minv", "mfcc_10_median", "mfcc_11_mean", "mfcc_11_std", "mfcc_11_maxv", "mfcc_11_minv", "mfcc_11_median", "mfcc_12_mean", "mfcc_12_std", "mfcc_12_maxv", "mfcc_12_minv", "mfcc_12_median", "poly_0_mean", "poly_0_std", "poly_0_maxv", "poly_0_minv", "poly_0_median", "poly_1_mean", "poly_1_std", "poly_1_maxv", "poly_1_minv", "poly_1_median", "spectral_centroid_mean", "spectral_centroid_std", "spectral_centroid_maxv", "spectral_centroid_minv", "spectral_centroid_median", "spectral_bandwidth_mean", "spectral_bandwidth_std", "spectral_bandwidth_maxv", "spectral_bandwidth_minv", "spectral_bandwidth_median", "spectral_contrast_mean", "spectral_contrast_std", "spectral_contrast_maxv", "spectral_contrast_minv", "spectral_contrast_median", "spectral_flatness_mean", "spectral_flatness_std", "spectral_flatness_maxv", "spectral_flatness_minv", "spectral_flatness_median", "spectral_rolloff_mean", "spectral_rolloff_std", "spectral_rolloff_maxv", "spectral_rolloff_minv", "spectral_rolloff_median", "zero_crossings_mean", "zero_crossings_std", "zero_crossings_maxv", "zero_crossings_minv", "zero_crossings_median", "RMSE_mean", "RMSE_std", "RMSE_maxv", "RMSE_minv", "RMSE_median"]}}, "text": {}, "image": {}, "video": {}, "csv": {}}, "models": {"audio": {"males": [{"sample type": "audio", "created date": "2020-08-03 12:55:08.238841", "device info": {"time": "2020-08-03 12:55", "timezone": ["EST", "EDT"], "operating system": "Darwin", "os release": "19.5.0", "os version": "Darwin Kernel Version 19.5.0: Tue May 26 20:41:44 PDT 2020; root:xnu-6153.121.2~2/RELEASE_X86_64", "cpu data": {"memory": [8589934592, 3035197440, 64.7, 4487892992, 379949056, 2523181056, 2408304640, 1964711936], "cpu percent": 66.0, "cpu times": [14797.03, 0.0, 9385.82, 76944.46], "cpu count": 4, "cpu stats": [153065, 479666, 89106680, 587965], "cpu swap": [2147483648, 1174405120, 973078528, 54.7, 30354079744, 203853824], "partitions": [["/dev/disk1s6", "/", "apfs", "ro,local,rootfs,dovolfs,journaled,multilabel"], ["/dev/disk1s5", "/System/Volumes/Data", "apfs", "rw,local,dovolfs,dontbrowse,journaled,multilabel"], ["/dev/disk1s4", "/private/var/vm", "apfs", "rw,local,dovolfs,dontbrowse,journaled,multilabel"], ["/dev/disk1s1", "/Volumes/Macintosh HD - Data", "apfs", "rw,local,dovolfs,journaled,multilabel"]], "disk usage": [499963174912, 10985529344, 320581328896, 3.3], "disk io counters": [1283981, 844586, 35781873664, 17365774336, 850754, 779944], "battery": [100, -2, true], "boot time": 1596411904.0}, "space left": 320.581328896}, "session id": "867c4358-d5a9-11ea-8720-acde48001122", "classes": ["males", "females"], "problem type": "classification", "model name": "gender_tpot_classifier.pickle", "model type": "tpot", "metrics": {"accuracy": 0.8947368421052632, "balanced_accuracy": 0.8944444444444444, "precision": 0.9, "recall": 0.9, "f1_score": 0.9, "f1_micro": 0.8947368421052632, "f1_macro": 0.8944444444444444, "roc_auc": 0.8944444444444444, "roc_auc_micro": 0.8944444444444444, "roc_auc_macro": 0.8944444444444444, "confusion_matrix": [[8, 1], [1, 9]], "classification_report": "              precision    recall  f1-score   support\n\n       males       0.89      0.89      0.89         9\n     females       0.90      0.90      0.90        10\n\n    accuracy                           0.89        19\n   macro avg       0.89      0.89      0.89        19\nweighted avg       0.89      0.89      0.89        19\n"}, "settings": {"version": "1.0.0", "augment_data": false, "balance_data": true, "clean_data": false, "create_csv": true, "default_audio_augmenters": ["augment_tsaug"], "default_audio_cleaners": ["clean_mono16hz"], "default_audio_features": ["librosa_features"], "default_audio_transcriber": ["deepspeech_dict"], "default_csv_augmenters": ["augment_ctgan_regression"], "default_csv_cleaners": ["clean_csv"], "default_csv_features": ["csv_features"], "default_csv_transcriber": ["raw text"], "default_dimensionality_reducer": ["pca"], "default_feature_selector": ["rfe"], "default_image_augmenters": ["augment_imaug"], "default_image_cleaners": ["clean_greyscale"], "default_image_features": ["image_features"], "default_image_transcriber": ["tesseract"], "default_outlier_detector": ["isolationforest"], "default_scaler": ["standard_scaler"], "default_text_augmenters": ["augment_textacy"], "default_text_cleaners": ["remove_duplicates"], "default_text_features": ["nltk_features"], "default_text_transcriber": ["raw text"], "default_training_script": ["tpot"], "default_video_augmenters": ["augment_vidaug"], "default_video_cleaners": ["remove_duplicates"], "default_video_features": ["video_features"], "default_video_transcriber": ["tesseract (averaged over frames)"], "dimension_number": 2, "feature_number": 20, "model_compress": false, "reduce_dimensions": false, "remove_outliers": true, "scale_features": true, "select_features": true, "test_size": 0.1, "transcribe_audio": true, "transcribe_csv": true, "transcribe_image": true, "transcribe_text": true, "transcribe_video": true, "visualize_data": false, "transcribe_videos": true}, "transformer name": "gender_tpot_classifier_transform.pickle", "training data": ["gender_all.csv", "gender_train.csv", "gender_test.csv", "gender_all_transformed.csv", "gender_train_transformed.csv", "gender_test_transformed.csv"], "sample X_test": [0.19491584410160165, 2.278239927977625, 1.9809968520802117, 0.01621731265879942, 0.15713016963065518, 0.6373734371406007, 0.5565326177000756, 0.21607641781209055, 1.5729652666810199, 0.4175324163804035, 0.25821087005791604, 1.688251084321436, 0.641181793964938, 0.8245062752279405, 3.328186152340374, -3.566702513086108, -0.7896923143197454, -0.33315803775179953, -0.9991381480355723, 3.3414140426072754], "sample y_test": 1}]}, "text": {}, "image": {}, "video": {}, "csv": {}}, "labels": ["load_dir"], "errors": [], "settings": {"version": "1.0.0", "augment_data": false, "balance_data": true, "clean_data": false, "create_csv": true, "default_audio_augmenters": ["augment_tsaug"], "default_audio_cleaners": ["clean_mono16hz"], "default_audio_features": ["librosa_features"], "default_audio_transcriber": ["deepspeech_dict"], "default_csv_augmenters": ["augment_ctgan_regression"], "default_csv_cleaners": ["clean_csv"], "default_csv_features": ["csv_features"], "default_csv_transcriber": ["raw text"], "default_dimensionality_reducer": ["pca"], "default_feature_selector": ["rfe"], "default_image_augmenters": ["augment_imaug"], "default_image_cleaners": ["clean_greyscale"], "default_image_features": ["image_features"], "default_image_transcriber": ["tesseract"], "default_outlier_detector": ["isolationforest"], "default_scaler": ["standard_scaler"], "default_text_augmenters": ["augment_textacy"], "default_text_cleaners": ["remove_duplicates"], "default_text_features": ["nltk_features"], "default_text_transcriber": ["raw text"], "default_training_script": ["tpot"], "default_video_augmenters": ["augment_vidaug"], "default_video_cleaners": ["remove_duplicates"], "default_video_features": ["video_features"], "default_video_transcriber": ["tesseract (averaged over frames)"], "dimension_number": 2, "feature_number": 20, "model_compress": false, "reduce_dimensions": false, "remove_outliers": true, "scale_features": true, "select_features": true, "test_size": 0.1, "transcribe_audio": true, "transcribe_csv": true, "transcribe_image": true, "transcribe_text": true, "transcribe_video": true, "visualize_data": false, "transcribe_videos": true}}
```

Click the .GIF below to follow along this example in a video format:

[![](https://github.com/jim-schwoebel/allie/blob/master/annotation/helpers/assets/load.gif)](https://drive.google.com/file/d/1-iN3wOjWGCiqlTKjQgtVoxfpJSmiDsCQ/view?usp=sharing)

### [Visualizing data](https://github.com/jim-schwoebel/allie/tree/master/visualize)
To visualize multiple folders of featurized files (in this case males and females folders of audio files in the ./train_dir), type this into the terminal:

```python3
cd /Users/jim/desktop/allie
cd visualize
python3 visualize.py audio males females
```

Note that visualization capabilities are restricted to classification problems for Allie version 1.0.

Click the .GIF below to follow along this example in a video format:

[![](https://github.com/jim-schwoebel/allie/blob/master/annotation/helpers/assets/visualize.gif)](https://drive.google.com/file/d/11GSJpE9ASp1AEl89CjVHRnEd2Yutg3Ki/view?usp=sharing)

## [Settings](https://github.com/jim-schwoebel/allie/blob/master/settings.json)

![](https://github.com/jim-schwoebel/allie/blob/master/annotation/helpers/assets/settings.png)

Allie has multiple default settings for model training to help you start out with the framework. Allie has been built so that the settings you specify in lists are done serially, which can be useful to construct machine learning models from multiple back-end model trainers in a single session. Settings can be modified in the [settings.json](https://github.com/jim-schwoebel/allie/blob/master/settings.json) file. 

Here are some settings that you can modify in this settings.json file and the various options for these settings:

| setting | description | default setting | all options | 
|------|------|------|------| 
| version | version of Allie release | 1.0 | 1.0 |
| augment_data | whether or not to implement data augmentation policies during the model training process via default augmentation scripts. | True | True, False |
| balance_data | whether or not to balance datasets during the model training process. | True | True, False | 
| clean_data | whether or not to clean datasets during the model training process via default cleaning scripts. | False | True, False | 
| create_csv | whether or not to output datasets in a nicely formatted .CSV as part of the model training process (outputs to ./data folder in model repositories) | True | True, False | 
| [default_audio_augmenters](https://github.com/jim-schwoebel/allie/tree/master/augmentation/audio_augmentation) | the default augmentation strategies used during audio modeling if augment_data == True | ["augment_tsaug"] | ['normalize_volume', 'normalize_pitch', 'time_stretch', 'opus_enhance', 'trim_silence', 'remove_noise', 'add_noise', "augment_tsaug"] | 
| [default_audio_cleaners](https://github.com/jim-schwoebel/allie/tree/master/cleaning/audio_cleaning) | the default cleaning strategies used during audio modeling if clean_data == True | ["clean_mono16hz"] | ["clean_getfirst3secs", "clean_keyword", "clean_mono16hz", "clean_towav", "clean_multispeaker", "clean_normalizevolume", "clean_opus", "clean_randomsplice", "clean_removenoise", "clean_removesilence", "clean_rename", "clean_utterances"] |
| [default_audio_features](https://github.com/jim-schwoebel/voice_modeling/tree/master/features/audio_features) | default set of audio features used for featurization (list). | ["standard_features"] | ["audioset_features", "audiotext_features", "librosa_features", "meta_features", "mixed_features", "opensmile_features", "praat_features", "prosody_features", "pspeech_features", "pyaudio_features", "pyaudiolex_features", "sa_features", "sox_features", "specimage_features", "specimage2_features", "spectrogram_features", "speechmetrics_features", "standard_features"] | 
| default_audio_transcriber | the default transcription model used during audio featurization if trainscribe_audio == True | ["deepspeech_dict"] | ["pocketsphinx", "deepspeech_nodict", "deepspeech_dict", "google", "wit", "azure", "bing", "houndify", "ibm"] | 
| [default_csv_augmenters](https://github.com/jim-schwoebel/allie/tree/master/augmentation/csv_augmentation) | the default augmentation strategies used to augment .CSV file types as part of model training if augment_data==True | ["augment_ctgan_regression"] | ["augment_ctgan_classification", "augment_ctgan_regression"]  | 
| [default_csv_cleaners](https://github.com/jim-schwoebel/allie/tree/master/cleaning/csv_cleaning) | the default cleaning strategies used to clean .CSV file types as part of model training if clean_data==True | ["clean_csv"] | ["clean_csv"] | 
| [default_csv_features](https://github.com/jim-schwoebel/allie/tree/master/features/csv_features) | the default featurization technique(s) used as a part of model training for .CSV files. | ["csv_features_regression"] | ["csv_features_regression"]  | 
| default_csv_transcriber | the default transcription technique for .CSV file spreadsheets. | ["raw text"] | ["raw text"] | 
| [default_dimensionality_reducer](https://github.com/jim-schwoebel/allie/blob/master/preprocessing/feature_reduce.py) | the default dimensionality reduction technique used if reduce_dimensions==True | ["pca"] | ["pca", "lda", "tsne", "plda","autoencoder"] | 
| [default_feature_selector](https://github.com/jim-schwoebel/allie/blob/master/preprocessing/feature_select.py) | the default feature selector used if select_features == True | ["rfe"] | ["chi", "fdr", "fpr", "fwe", "lasso", "percentile", "rfe", "univariate", "variance"]  | 
| [default_image_augmenters](https://github.com/jim-schwoebel/allie/tree/master/augmentation/image_augmentation) | the default augmentation techniques used for images if augment_data == True as a part of model training. | ["augment_imaug"] | ["augment_imaug"]  | 
| [default_image_cleaners](https://github.com/jim-schwoebel/allie/tree/master/cleaning/image_cleaning) | the default cleaning techniques used for image data as a part of model training is clean_data == True| ["clean_greyscale"] |["clean_extractfaces", "clean_greyscale", "clean_jpg2png"] | 
| [default_image_features](https://github.com/jim-schwoebel/voice_modeling/tree/master/features/image_features) | default set of image features used for featurization (list). | ["image_features"] | ["image_features", "inception_features", "resnet_features", "squeezenet_features", "tesseract_features", "vgg16_features", "vgg19_features", "xception_features"] | 
| default_image_transcriber | the default transcription technique used for images (e.g. image --> text transcript) | ["tesseract"] | ["tesseract"] |
| default_outlier_detector | the default outlier technique(s) used to clean data as a part of model training if remove_outliers == True | ["isolationforest"] | ["isolationforest", "zscore"]  | 
| [default_scaler](https://github.com/jim-schwoebel/allie/blob/master/preprocessing/feature_scale.py) | the default scaling technique used to preprocess data during model training if scale_features == True | ["standard_scaler"] | ["binarizer", "one_hot_encoder", "normalize", "power_transformer", "poly", "quantile_transformer", "standard_scaler"] | 
| [default_text_augmenters](https://github.com/jim-schwoebel/allie/tree/master/augmentation/text_augmentation) | the default augmentation strategies used during model training for text data if augment_data == True | ["augment_textacy"] | ["augment_textacy", "augment_summary"]  | 
| [default_text_cleaners](https://github.com/jim-schwoebel/allie/tree/master/cleaning/text_cleaning) | the default cleaning techniques used during model training on text data if clean_data == True| ["clean_textacy"] | ["clean_summary", "clean_textacy"]  | 
| [default_text_features](https://github.com/jim-schwoebel/voice_modeling/tree/master/features/csv_features) | default set of text features used for featurization (list). | ["nltk_features"] | ["bert_features", "fast_features", "glove_features", "grammar_features", "nltk_features", "spacy_features", "text_features", "w2v_features"] | 
| default_text_transcriber | the default transcription techniques used to parse raw .TXT files during model training| ["raw_text"] | ["raw_text"]  | 
| [default_training_script](https://github.com/jim-schwoebel/allie/tree/master/training) | the specified traning script(s) to train machine learning models. Note that if you specify multiple training scripts here that the training scripts will be executed serially (list). | ["tpot"] |["alphapy", "atm", "autogbt", "autokaggle", "autokeras", "auto-pytorch", "btb", "cvopt", "devol", "gama", "hyperband", "hypsklearn", "hungabunga", "imbalance-learn", "keras", "ludwig", "mlblocks", "neuraxle", "safe", "scsr", "tpot"]| 
| [default_video_augmenters](https://github.com/jim-schwoebel/allie/tree/master/augmentation/video_augmentation) | the default augmentation strategies used for videos during model training if augment_data == True | ["augment_vidaug"] | ["augment_vidaug"] | 
| [default_video_cleaners](https://github.com/jim-schwoebel/allie/tree/master/cleaning/video_cleaning) | the default cleaning strategies used for videos if clean_data == True | ["clean_alignfaces"] | ["clean_alignfaces", "clean_videostabilize"] | 
| [default_video_features](https://github.com/jim-schwoebel/voice_modeling/tree/master/features/video_features) | default set of video features used for featurization (list). | ["video_features"] | ["video_features", "y8m_features"] | 
| default_video_transcriber | the default transcription technique used for videos (.mp4 --> text from the video) | ["tesseract (averaged over frames)"] | ["tesseract (averaged over frames)"] |
| dimension_number | the number of dimensions to reduce a dataset into if reduce_dimensions == True| 100 | any integer from 1 to the number of features-1 | 
| feature_number | the number of features to select for via the feature selection strategy (default_feature_selector) if select_features == True| 20 | any integer from 1 to the number of features-1 | 
| model_compress | a setting that specifies whether or not to compress machine learning models during model training | False | True, False | 
| reduce_dimensions | a setting that specifies whether or not to reduce dimensions via the default_dimensionality_reducer | False | True, False | 
| remove_outliers | a setting that specifies whether or not to remove outliers during model training via the default_outlier_detector | True | True, False | 
| scale_features | a setting that specifies whether or not to scale features during featurization and model training via the default_scaler | True | True, False | 
| select_features | a setting that specifies whether or not to employ specified feature selection strategies (via the default_feature_selector) | True | True, False | 
| test_size | a setting that specifies the size of the testing dataset for defining model performance after model training. | 0.10 | Any number 0.10-0.50 | 
| transcribe_audio | a setting to define whether or not to transcribe audio files during featurization and model training via the default_audio_transcriber | True | True, False | 
| transcribe_csv | a setting to define whether or not to transcribe csv files during featurization and model training via the default_csv_transcriber | True | True, False | 
| transcribe_image | a setting to define whether or not to transcribe image files during featurization and model training via the default_image_transcriber | True | True, False | 
| transcribe_text | a setting to define whether or not to transcribe text files during featurization and model training via the default_image_transcriber | True | True, False | 
| transcribe_video | a setting to define whether or not to transcribe video files during featurization and model training via the default_video_transcriber | True | True, False | 
| [visualize_data](https://github.com/jim-schwoebel/allie/tree/master/visualize) | a setting to define whether or not to visualize features during the model training process via [Allie's visualization API](https://github.com/jim-schwoebel/allie/tree/master/visualize) | False | True, False | 

## License
This repository is licensed under an [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0). 

## Feedback
Any feedback on this repository is greatly appreciated. 
* If you find something that is missing or doesn't work, please consider opening a [GitHub issue](https://github.com/jim-schwoebel/Allie/issues).
* If you'd like to be mentored by someone on our team, check out the [Innovation Fellows Program](http://neurolex.ai/research).
* If you want to learn more about voice computing, check out [Voice Computing in Python](https://github.com/jim-schwoebel/voicebook) book.
* If you want to talk to me directly, please send me an email @ js@neurolex.co.

## Contribute

We have built this machine learning framework to be quite agile to fit many purposes and needs, and we're excited to see how the open source community uses it. Forks and PRs are encouraged! 

If you want to be a contributer, check out the [active projects](https://github.com/jim-schwoebel/allie/projects). There are many components of Allie that are expanding including the Annotation API, Visualization API, Features API, Modeling API, Cleaning API, and Augmentation API. If you are interested to contribute, please send me an email @ js@neurolex.co specifying the project you'd like to be a part of and I'll plug you in with the right team.

## Additional resources

You may want to read through [the wiki](https://github.com/jim-schwoebel/allie/wiki) for additional documentation.

* [1. Getting started](https://github.com/jim-schwoebel/allie/wiki/1.-Getting-started)
* [2. Preparing datasets](https://github.com/jim-schwoebel/allie/wiki/2.-Preparing-datasets)
* [2.1 Annotating datasets](https://github.com/jim-schwoebel/allie/wiki/2.1.-Annotating-datasets)
* [2.2 Cleaning datasets](https://github.com/jim-schwoebel/allie/wiki/2.2.-Cleaning-datasets)
* [2.3. Augmenting datasets](https://github.com/jim-schwoebel/allie/wiki/2.3.-Augmenting-datasets)
* [3. Training models](https://github.com/jim-schwoebel/allie/wiki/3.-Training-models)
* [3.1. Data featurization](https://github.com/jim-schwoebel/allie/wiki/3.1.-Data-featurization)
* [3.2. Data preprocessing](https://github.com/jim-schwoebel/allie/wiki/3.2.-Data-preprocessing)
* [3.3. Data visualization](https://github.com/jim-schwoebel/allie/wiki/3.3.-Data-visualization)
* [4. Model predictions](https://github.com/jim-schwoebel/allie/wiki/4.-Model-predictions)
