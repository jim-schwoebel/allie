## Training scripts 

![](https://github.com/jim-schwoebel/allie/blob/master/annotation/helpers/assets/model.png)

Use this folder to train machine learning models according to the default_training_script using the [model.py](https://github.com/jim-schwoebel/allie/blob/master/training/model.py) script.

![](https://github.com/jim-schwoebel/Allie/blob/master/training/helpers/train.gif)

## Getting started

All you need to do to get started is go to this repository and run upgrade.py followed by [model.py](https://github.com/jim-schwoebel/allie/blob/master/training/model.py):

```
cd allie/training
python3 model.py 
```

You then will be asked a few questions regarding the training process (in terms of data type, number of classes, and the name of the model). Note that --> indicates typed responses. 

### regression model example 
For regression model training, you need to insert a .CSV file for training. You can then specify the target classes here from the spreadsheet and the models will then be trained with the specified model trainers.

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
{"sample type": "csv", "created date": "2020-08-03 15:29:43.786976", "device info": {"time": "2020-08-03 15:29", "timezone": ["EST", "EDT"], "operating system": "Darwin", "os release": "19.5.0", "os version": "Darwin Kernel Version 19.5.0: Tue May 26 20:41:44 PDT 2020; root:xnu-6153.121.2~2/RELEASE_X86_64", "cpu data": {"memory": [8589934592, 2577022976, 70.0, 4525428736, 107941888, 2460807168, 2122092544, 2064621568], "cpu percent": 59.1, "cpu times": [22612.18, 0.0, 12992.38, 102624.04], "cpu count": 4, "cpu stats": [110955, 504058, 130337047, 518089], "cpu swap": [2147483648, 1096548352, 1050935296, 51.1, 44743286784, 329093120], "partitions": [["/dev/disk1s6", "/", "apfs", "ro,local,rootfs,dovolfs,journaled,multilabel"], ["/dev/disk1s5", "/System/Volumes/Data", "apfs", "rw,local,dovolfs,dontbrowse,journaled,multilabel"], ["/dev/disk1s4", "/private/var/vm", "apfs", "rw,local,dovolfs,dontbrowse,journaled,multilabel"], ["/dev/disk1s1", "/Volumes/Macintosh HD - Data", "apfs", "rw,local,dovolfs,journaled,multilabel"]], "disk usage": [499963174912, 10985529344, 317145075712, 3.3], "disk io counters": [1689675, 1773144, 52597518336, 34808844288, 1180797, 1136731], "battery": [100, -2, true], "boot time": 1596411904.0}, "space left": 317.145075712}, "session id": "fc54dd66-d5bc-11ea-9c75-acde48001122", "classes": ["class_"], "problem type": "regression", "model name": "gender_tpot_regression.pickle", "model type": "tpot", "metrics": {"mean_absolute_error": 0.37026379788606023, "mean_squared_error": 0.16954440031335424, "median_absolute_error": 0.410668441980656, "r2_score": 0.3199385720764347}, "settings": {"version": "1.0.0", "augment_data": false, "balance_data": true, "clean_data": false, "create_csv": true, "default_audio_augmenters": ["augment_tsaug"], "default_audio_cleaners": ["clean_mono16hz"], "default_audio_features": ["librosa_features"], "default_audio_transcriber": ["deepspeech_dict"], "default_csv_augmenters": ["augment_ctgan_regression"], "default_csv_cleaners": ["clean_csv"], "default_csv_features": ["csv_features"], "default_csv_transcriber": ["raw text"], "default_dimensionality_reducer": ["pca"], "default_feature_selector": ["rfe"], "default_image_augmenters": ["augment_imgaug"], "default_image_cleaners": ["clean_greyscale"], "default_image_features": ["image_features"], "default_image_transcriber": ["tesseract"], "default_outlier_detector": ["isolationforest"], "default_scaler": ["standard_scaler"], "default_text_augmenters": ["augment_textacy"], "default_text_cleaners": ["remove_duplicates"], "default_text_features": ["nltk_features"], "default_text_transcriber": ["raw text"], "default_training_script": ["tpot"], "default_video_augmenters": ["augment_vidaug"], "default_video_cleaners": ["remove_duplicates"], "default_video_features": ["video_features"], "default_video_transcriber": ["tesseract (averaged over frames)"], "dimension_number": 2, "feature_number": 20, "model_compress": false, "reduce_dimensions": false, "remove_outliers": true, "scale_features": false, "select_features": false, "test_size": 0.1, "transcribe_audio": false, "transcribe_csv": true, "transcribe_image": true, "transcribe_text": true, "transcribe_video": true, "transcribe_videos": true, "visualize_data": false, "default_dimensionionality_reducer": ["pca"]}, "transformer name": "", "training data": [], "sample X_test": [30.0, 116.1, 68.390715744171, 224.0, 3.0, 115.5, 129.19921875, 1.579895074162117, 1.4053805862299766, 6.915237601339313, 0.0, 1.1654598038099069, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8033179369485901, 0.00438967342343324, 0.8140795129312649, 0.7979309783326958, 0.80255447893579, 0.5772101965904585, 0.025367026843705915, 0.6147904436358145, 0.5452462503889344, 0.5720709525572024, 0.5251607032640779, 0.031273364291655614, 0.5651684602891733, 0.4833782607526296, 0.522481114581999, 0.53067387207457, 0.01636309315550051, 0.5760527497162795, 0.5083941678429416, 0.5308772078223155, 0.5383483269837346, 0.02398538849569036, 0.6138641187358237, 0.5148823529890311, 0.5317355191905834, 0.5590921868458475, 0.018941050706796927, 0.6185565218733067, 0.5391848127954322, 0.5515129204797803, 0.5653692033981255, 0.022886171192539908, 0.6170498591126126, 0.5187020777516459, 0.5693268285980656, 0.5428369240411614, 0.011543007163874491, 0.5837123204211986, 0.5208221399174541, 0.5415414663324902, 0.4946660644711973, 0.021472694373470352, 0.5215764169994959, 0.4640787039752625, 0.4952267598817138, 0.4798469011394895, 0.02593484469896265, 0.5172960598832023, 0.4449712627305569, 0.4777149108114186, 0.4993938744598669, 0.01849048457494309, 0.5651910299787914, 0.4822436630327371, 0.4950261489562563, 0.5363930497563161, 0.0376443504751349, 0.6330907702118795, 0.4816294954352716, 0.5249507027509328, -235.4678661326307, 61.51638081120653, -119.29458629496251, -362.1632462796749, -227.60500825042942, 163.92070611988834, 47.05955903012367, 237.9764586528294, 41.986380826321785, 172.71493170004138, 9.237411399943188, 25.868443694231683, 61.477039729510096, -75.39528620218707, 9.629797757209056, 38.85787728431835, 25.651975918739637, 120.33667371104372, -9.003575689525233, 36.13886469019118, -3.813926397129359, 18.466559976322753, 45.395818864794386, -54.58126572108478, -3.563646356257889, 28.49882430361086, 15.286105184256387, 72.2886732962803, 0.03239718043784112, 26.491533722920998, -19.866746887564343, 16.46528562102129, 9.928420130258688, -61.42422346209003, -17.134010559191154, 4.917765483447672, 13.106589177321654, 36.30054941946764, -28.88492762419697, 4.470641784765922, -7.5214435695300805, 11.456845078656613, 24.68530842159717, -33.23468909518539, -7.800944005694487, 1.7653313822916499, 10.137823325108423, 26.38688279047729, -22.507646864346647, 2.1230603462314384, 2.9722994596741263, 9.920580299259306, 29.09083383516883, -28.462312178142557, 3.1356694281534625, -8.31659816437322, 9.321735116288234, 14.977416272339756, -29.19924207526083, -7.200232618719922, 10.020856138237773, 9.605360863583002, 33.70453001221575, -10.34310153320585, 8.538943192527702, -0.0003117740953455404, 0.0002093530273784296, -3.649852038234921e-05, -0.0008609846033373115, -0.00024944132582088046, 2.427670449088513, 1.573081810523066, 6.574603060966783, 0.2961628052414745, 1.8991203106986, 1122.5579040699354, 895.7957759390358, 4590.354474064802, 349.53842801686966, 800.0437543350607, 1384.7323846043691, 519.4846094956321, 2642.151716668925, 703.4646482237979, 1229.7584170111122, 22.097758701059746, 6.005214057147793, 54.922406822231686, 8.895233246285754, 22.047151155860252, 6.146541272755712e-05, 0.00013457647582981735, 0.0006881643203087151, 5.692067475138174e-07, 8.736528798181098e-06, 2087.3572470319323, 1731.5818839146564, 6535.3271484375, 409.130859375, 1421.19140625, 0.05515445892467249, 0.07680443213522453, 0.46142578125, 0.0078125, 0.0302734375, 0.12412750720977785, 0.07253565639257431, 0.29952874779701233, 0.010528072714805605, 0.10663044452667236], "sample y_test": 0}
```

The resulting model will have the following data:
```
└── gender_tpot_regression
    ├── data
    │   ├── gender_all.csv
    │   ├── gender_all_transformed.csv
    │   ├── gender_test.csv
    │   ├── gender_test_transformed.csv
    │   ├── gender_train.csv
    │   └── gender_train_transformed.csv
    ├── model
    │   ├── bar_graph_predictions.png
    │   ├── gender_tpot_regression.json
    │   ├── gender_tpot_regression.pickle
    │   ├── gender_tpot_regression.py
    │   └── gender_tpot_regression_transform.pickle
    ├── readme.md
    ├── requirements.txt
    └── settings.json
```

Click the .GIF below to follow along this example in a video format:

[![](https://github.com/jim-schwoebel/allie/blob/master/annotation/helpers/assets/regression.gif)](https://drive.google.com/file/d/1PQwBABSRKrzS67IrlgvFSjuRBKFFx-XX/view?usp=sharing)

### classification model example 
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

[![](https://github.com/jim-schwoebel/allie/blob/master/annotation/helpers/assets/classification.gif)](https://www.youtube.com/watch?v=xfGgCgC9-wA&list=PL_D3Oayw4KAqvwWGxE0VWA9-r3LNKrUrU&index=7)

After this, the model will be trained and placed in the models/[sampletype_models] directory. For example, if you trained an audio model with TPOT, the model will be placed in the allie/models/audio_models/ directory. 

For automated training, you can alternatively pass through sys.argv[] inputs as follows:

```
python3 model.py audio 2 c gender males females
```
Where:
- audio = audio file type 
- 2 = 2 classes 
- c = classification (r for regression)
- gender = common name of model
- male = first class
- female = second class [via N number of classes]

The goal is to make an output folder like this:
```
└── gender_tpot_classifier
    ├── data
    │   ├── gender_all.csv
    │   ├── gender_all_transformed.csv
    │   ├── gender_test.csv
    │   ├── gender_test_transformed.csv
    │   ├── gender_train.csv
    │   └── gender_train_transformed.csv
    ├── model
    │   ├── confusion_matrix.png
    │   ├── gender_tpot_classifier.json
    │   ├── gender_tpot_classifier.pickle
    │   ├── gender_tpot_classifier.py
    │   └── gender_tpot_classifier_transform.pickle
    ├── readme.md
    ├── requirements.txt
    └── settings.json
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
{"version": "1.0.0", 
 "augment_data": false, 
 "balance_data": true, 
 "clean_data": false, 
 "create_csv": true, 
 "default_audio_augmenters": ["augment_tsaug"], 
 "default_audio_cleaners": ["clean_mono16hz"], 
 "default_audio_features": ["librosa_features"], 
 "default_audio_transcriber": ["deepspeech_dict"], 
 "default_csv_augmenters": ["augment_ctgan_regression"], 
 "default_csv_cleaners": ["clean_csv"], 
 "default_csv_features": ["csv_features"], 
 "default_csv_transcriber": ["raw text"], 
 "default_dimensionality_reducer": ["pca"], 
 "default_feature_selector": ["rfe"], 
 "default_image_augmenters": ["augment_imgaug"], 
 "default_image_cleaners": ["clean_greyscale"], 
 "default_image_features": ["image_features"], 
 "default_image_transcriber": ["tesseract"], 
 "default_outlier_detector": ["isolationforest"], 
 "default_scaler": ["standard_scaler"], 
 "default_text_augmenters": ["augment_textacy"], 
 "default_text_cleaners": ["remove_duplicates"], 
 "default_text_features": ["nltk_features"], 
 "default_text_transcriber": ["raw text"], 
 "default_training_script": ["tpot"], 
 "default_video_augmenters": ["augment_vidaug"], 
 "default_video_cleaners": ["remove_duplicates"], 
 "default_video_features": ["video_features"], 
 "default_video_transcriber": ["tesseract (averaged over frames)"], 
 "dimension_number": 2, 
 "feature_number": 20, 
 "model_compress": false, 
 "reduce_dimensions": false, 
 "remove_outliers": true, 
 "scale_features": true, 
 "select_features": true, 
 "test_size": 0.1, 
 "transcribe_audio": true, 
 "transcribe_csv": true, 
 "transcribe_image": true, 
 "transcribe_text": true, 
 "transcribe_video": true, 
 "transcribe_videos": true,
 "visualize_data": false}
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

## [Settings](https://github.com/jim-schwoebel/allie/blob/master/settings.json)

Here are some settings that you can modify in the [settings.json file](https://github.com/jim-schwoebel/allie/blob/master/settings.json) related to Allie's Model API.

| setting | description | default setting | all options | 
|------|------|------|------| 
| augment_data | whether or not to implement data augmentation policies during the model training process via default augmentation scripts. | True | True, False |
| balance_data | whether or not to balance datasets during the model training process. | True | True, False | 
| clean_data | whether or not to clean datasets during the model training process via default cleaning scripts. | False | True, False | 
| create_csv | whether or not to output datasets in a nicely formatted .CSV as part of the model training process (outputs to ./data folder in model repositories) | True | True, False | 
| [default_audio_augmenters](https://github.com/jim-schwoebel/allie/tree/master/augmentation/audio_augmentation) | the default augmentation strategies used during audio modeling if augment_data == True | ["augment_tsaug"] | ["augment_tsaug", "augment_addnoise", "augment_noise", "augment_pitch", "augment_randomsplice", "augment_silence", "augment_time", "augment_volume"] | 
| [default_audio_cleaners](https://github.com/jim-schwoebel/allie/tree/master/cleaning/audio_cleaning) | the default cleaning strategies used during audio modeling if clean_data == True | ["clean_mono16hz"] | ["clean_getfirst3secs", "clean_keyword", "clean_mono16hz", "clean_towav", "clean_multispeaker", "clean_normalizevolume", "clean_opus", "clean_randomsplice", "clean_removenoise", "clean_removesilence", "clean_rename", "clean_utterances"] |
| [default_audio_features](https://github.com/jim-schwoebel/voice_modeling/tree/master/features/audio_features) | default set of audio features used for featurization (list). | ["standard_features"] | ["audioset_features", "audiotext_features", "librosa_features", "meta_features", "mixed_features", "opensmile_features", "praat_features", "prosody_features", "pspeech_features", "pyaudio_features", "pyaudiolex_features", "sa_features", "sox_features", "specimage_features", "specimage2_features", "spectrogram_features", "speechmetrics_features", "standard_features"] | 
| default_audio_transcriber | the default transcription model used during audio featurization if trainscribe_audio == True | ["deepspeech_dict"] | ["pocketsphinx", "deepspeech_nodict", "deepspeech_dict", "google", "wit", "azure", "bing", "houndify", "ibm"] | 
| [default_csv_augmenters](https://github.com/jim-schwoebel/allie/tree/master/augmentation/csv_augmentation) | the default augmentation strategies used to augment .CSV file types as part of model training if augment_data==True | ["augment_ctgan_regression"] | ["augment_ctgan_classification", "augment_ctgan_regression"]  | 
| [default_csv_cleaners](https://github.com/jim-schwoebel/allie/tree/master/cleaning/csv_cleaning) | the default cleaning strategies used to clean .CSV file types as part of model training if clean_data==True | ["clean_csv"] | ["clean_csv"] | 
| [default_csv_features](https://github.com/jim-schwoebel/allie/tree/master/features/csv_features) | the default featurization technique(s) used as a part of model training for .CSV files. | ["csv_features_regression"] | ["csv_features_regression"]  | 
| default_csv_transcriber | the default transcription technique for .CSV file spreadsheets. | ["raw text"] | ["raw text"] | 
| [default_dimensionality_reducer](https://github.com/jim-schwoebel/allie/blob/master/preprocessing/feature_reduce.py) | the default dimensionality reduction technique used if reduce_dimensions==True | ["pca"] | ["pca", "lda", "tsne", "plda","autoencoder"] | 
| [default_feature_selector](https://github.com/jim-schwoebel/allie/blob/master/preprocessing/feature_select.py) | the default feature selector used if select_features == True | ["rfe"] | ["chi", "fdr", "fpr", "fwe", "lasso", "percentile", "rfe", "univariate", "variance"]  | 
| [default_image_augmenters](https://github.com/jim-schwoebel/allie/tree/master/augmentation/image_augmentation) | the default augmentation techniques used for images if augment_data == True as a part of model training. | ["augment_imgaug"] | ["augment_imgaug"]  | 
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
