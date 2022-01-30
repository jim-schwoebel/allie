# Preprocessing scripts

![](https://github.com/jim-schwoebel/allie/blob/master/annotation/helpers/assets/model.png)

This is a folder for manipulating and pre-processing [features extracted](https://github.com/jim-schwoebel/allie/tree/master/features) from audio, text, image, video, or .CSV files as part of the machine learning modeling process. 

This is done via a convention for transformers, which are in the proper folders (e.g. audio files --> audio_transformers). There are three main feature transformation techniques: [feature scaling](https://github.com/jim-schwoebel/allie/blob/master/preprocessing/feature_scale.py), [feature selection](https://github.com/jim-schwoebel/allie/blob/master/preprocessing/feature_select.py), and [dimensionality reduction](https://github.com/jim-schwoebel/allie/blob/master/preprocessing/feature_reduce.py).

In this way, we can appropriately create transformers for various sample data types. 

## Building transformers (for classification problems)

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

[![](https://github.com/jim-schwoebel/allie/blob/master/annotation/helpers/assets/transform.gif)](https://www.youtube.com/watch?v=w5ekGiUpLjg&list=PL_D3Oayw4KAqvwWGxE0VWA9-r3LNKrUrU&index=7)

The code above will transform all the featurized text files (in .JSON files, folder MALES and folder FEMALES) via a classification script (c) with a common name GENDER. For clarity, the command line arguments are further elaborated upon below along with all possible options to help you use the transformers API. Note that folder ONE and folder TWO are assumed to be in the [train_dir folder](https://github.com/jim-schwoebel/allie/tree/master/train_dir).

| CLI argument | sample | description | all options | 
|------|------|------|------| 
| sys.argv[1] | 'audio' | the sample type of file preprocessed by the transformer | ['audio', 'text', 'image', 'video', 'csv'] | 
| sys.argv[2] | 'c' | classification or regression problems | ['c', 'r'] | 
| sys.argv[3] | 'gender' | the common name for the transformer | can be any string | 
| sys.argv[4], sys.argv[5], sys.argv[n] | 'males' | classes that you seek to model in the [train_dir folder](https://github.com/jim-schwoebel/allie/tree/master/train_dir) | any string folder name |

## Building transformers (for regression problems)

To transform an entire folder of a featurized files (for a regression problem - target being between [0,1], you can run:

```
cd ~ 
cd allie/preprocessing
python3 transform.py text c age test.csv /Users/jim/desktop/allie/train_dir age
```

The code above will transform all the features in the test.csv spreadsheet in the /Users/jim/desktop/allie/train_dir around the target variable age according to the specified preprocessing settings. In other words, all other variables from the target variable are represented as numberical features that will be transformed.

| CLI argument | sample | description | all options | 
|------|------|------|------| 
| sys.argv[1] | 'text' | the sample type of file preprocessed by the transformer | ['audio', 'text', 'image', 'video', 'csv'] | 
| sys.argv[2] | 'c' | classification or regression problems | ['c', 'r'] | 
| sys.argv[3] | 'age' | target variable in a spreadsheet | any string variable as a pandas dataframe | 
| sys.argv[4] | 'test.csv' | csv spreadsheet for the regression problem | any string that represents a spreadsheet name | 
| sys.argv[5] | '/Users/jim/desktop/allie/train_dir' | directory of the spreadsheet | any string directory file (can get with os.getcwd()) | 
| sys.argv[6] | 'age' | common_name for the modeling problem | any string common name that makes sense for the problem | 

## Settings

Here are the relevant settings in Allie related to preprocessing that you can change in the [settings.json file](https://github.com/jim-schwoebel/allie/blob/master/settings.json) (along with the default settings).

| setting | description | default setting | all options | 
|------|------|------|------| 
| reduce_dimensions | if True, reduce dimensions via the default_dimensionality_reducer (or set of dimensionality reducers) | False | True, False |
| [default_dimensionality_reducer](https://github.com/jim-schwoebel/allie/blob/master/preprocessing/feature_reduce.py) | the default dimensionality reducer or set of dimensionality reducers | ["pca"] | ["pca", "lda", "tsne", "plda","autoencoder"] | 
| select_features | if True, select features via the default_feature_selector (or set of feature selectors) | False | True, False | 
| [default_feature_selector](https://github.com/jim-schwoebel/allie/blob/master/preprocessing/feature_select.py) | the default feature selector or set of reature selectors | ["lasso"] | ["lasso", "rfe", "chi", "kbest", "variance"] | 
| scale_features | if True, scales features via the default_scaler (or set of scalers) | False | True, False | 
| [default_scaler](https://github.com/jim-schwoebel/allie/blob/master/preprocessing/feature_scale.py) | the default scaler (e.g. StandardScalar) to pre-process data | ["standard_scaler"] | ["binarizer", "one_hot_encoder", "normalize", "power_transformer", "poly", "quantile_transformer", "standard_scaler"]|
