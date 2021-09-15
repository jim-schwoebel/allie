## Train directory 

Use this directory to train machine learning models based on folders of files.

## Getting started 

To get started, you just need to make at least 2 folders containing the same type of file (e.g. audio .WAV files). Ideally, these folders will have the same number of classes; otherwise, the classes will automatically balance to the lower number of classes during model training.

![](https://github.com/jim-schwoebel/allie/blob/master/training/helpers/train_1.png) 

![](https://github.com/jim-schwoebel/allie/blob/master/training/helpers/train_2.png)  

Now you need to run model.py:

```
cd allie/training
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

Goal is to make an output folder like this:
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

Additional instructions can be found [here](https://github.com/jim-schwoebel/allie/tree/master/training).

Note you can edit the settings.json to change the default featurizer for model training. It is important to train using standard arrays if you plan to put models into production environments, as our database only takes in standard_features for audio files. 

## Supported file formats

Here are the supported file formats for the load directory. Note that if you use alternative file types, the training script may error.

| File type | extension | recommended format | 
| ------------- |-------------| -------------| 
| audio file | .WAV, .MP3 | .WAV | 
| text file | .TXT | .TXT | 
| image file | .PNG, .JPG | .PNG | 
| video file | .MP4 | .MP4 | 
| CSV file | .CSV | .CSV | 

## Other scripts

There are a few other scripts in this folder. The table below describes what each of these scripts does and how to call them.

| Script name | What it does | How to call  | Example  | 
| ------------- |-------------| -------------| -------------| 
| [create_csv.py](https://github.com/jim-schwoebel/allie/blob/master/train_dir/combine_datasets.py) | Combines multiple folders of featurized datasets into one folder for a two-class problem | ```python3 create_datasets.py``` | ```python3 create_datasets.py``` |
| [create_csv.py](https://github.com/jim-schwoebel/allie/blob/master/train_dir/create_csv.py) | Creates a nicely formatted .CSV file with the file paths and class labels for regression modeling | ```python3 create_csv.py [folderpathA] [folderpathB] [folderpath...N]``` | ```python3 create_csv.py /Users/jim/desktop/allie/train_dir/males```|
| [create_dataset.py](https://github.com/jim-schwoebel/allie/blob/master/train_dir/create_dataset.py) | Converts regression datasets with thresholds to classification datasets full of files | ```python3 create_dataset.py [csvfile] [targetname]``` | ```python3 create_dataset.py What_is_your_total_household_income.csv 'What is your total household income?'``` |
| [delete_features.py](https://github.com/jim-schwoebel/allie/blob/master/train_dir/delete_features.py) | Deletes a specified set of features from a .JSON file (in all folders in train_dir), as specified by the user. | ```python3 delete_features.py [sampletype] [feature_set]``` | ```python3 delete_features.py audio librosa_features``` |  
| [delete_json.py](https://github.com/jim-schwoebel/allie/blob/master/train_dir/delete_json.py) | Deletes all .JSON files from all folders in the train_dir (useful to re-featurize sets of files). | ```python3 delete_json.py``` | ```python3 delete_json.py``` | 
| [make_csv_regression.py](https://github.com/jim-schwoebel/allie/blob/master/train_dir/make_csv_regression.py) | Makes many .CSV files around the target variable for use in regression modeling (note you need to customize this to be useful). | ```python3 make_csv_regression.py [csvfile] [targetcol]``` | ```python3 make_csv_regression.py test2.csv urls``` |
| [make_new.py](https://github.com/jim-schwoebel/allie/blob/master/train_dir/make_new.py) | Sample script to rename the URL / target column with a new directory; useful if you need to clone data from one hard disk to another (note you have to manually edit this script to make it useful). | ```python3 make_new.py [csvfile] [newdir] [targetvar]``` | ```python3 make_new.py new.csv /Users/jim/desktop/train_dir/one url``` |
| [rename.py](https://github.com/jim-schwoebel/allie/blob/master/train_dir/rename.py) | Renames all the files in a particular directory (both audio files and .JSON files). Note you can manually change this to other file types | ```python3 rename_files.py [folder]``` | ```python3 rename.py /Users/jim/desktop/allie/train_dir/males``` | 
