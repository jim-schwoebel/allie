# Models

Use this section of the repository to make model predctions in the ./load_dir.

## Getting started

First, you need to put some files in the ./load_dir. I put the test files from the ./test directory into the load_dir here for demostration purposes. This is the output from the terminal (of files in the ./load_dir.

```
Jims-MBP:allie jimschwoebel$ cd load_dir
Jims-MBP:load_dir jimschwoebel$ ls
README.md	test_csv.csv	test_text.txt
test_audio.wav	test_image.png	test_video.mp4
```


```
cd ..
cd models 
python3 load_models.py
-----------------------------------
DETECTED 5 FILES (['audio', 'image', 'text', 'video', 'csv'])
-----------------------------------
{'audio': 1, 'image': 1, 'text': 1, 'video': 1, 'csv': 1}
-----------------------------------
AUDIO FEATURIZING - STANDARD_FEATURES
-----------------------------------
[-1.37365495e+02  5.52696807e+01 -5.56219741e+02 -3.11808560e+01
  1.26186729e+02  3.44293030e+01  0.00000000e+00  1.87490603e+02
 -1.65840442e+01  1.81620401e+01 -6.68236487e+01  3.39391945e+01
  1.91875792e+01  1.09147051e+01 -7.12643280e+00  4.75592577e+01
  2.63126186e+00  9.03896630e+00 -2.17933233e+01  3.27716689e+01
  8.40900114e+00  9.47258716e+00 -1.83439374e+01  3.55795595e+01
 -9.28245118e-01  8.23763361e+00 -2.00119846e+01  2.61531651e+01
  2.07184430e+00  1.20154123e+01 -2.91630854e+01  4.90588556e+01
 -1.44721354e+00  6.99959352e+00 -1.96954138e+01  1.83847828e+01
  3.93945878e+00  8.32943686e+00 -1.74449678e+01  2.50007246e+01
 -5.81459368e-01  7.63057475e+00 -1.88850095e+01  2.35547407e+01
  3.62425284e+00  8.13087766e+00 -1.84198226e+01  2.53250902e+01
  2.55676381e+00  9.81263267e+00 -1.82418372e+01  3.44094708e+01
  9.17884383e-01  8.39953107e+00 -7.08782839e+00  6.50221762e+01
  2.35919092e-01  3.42553467e+00 -1.68830931e+01  8.62757022e+00
  1.49677597e-02  2.26734307e+00 -4.37835907e+00  1.04613363e+01
 -2.77642024e-02  1.34695789e+00 -4.24227977e+00  4.06712106e+00
  1.33808770e-02  1.22908430e+00 -2.88217702e+00  4.13601725e+00
 -4.82870228e-02  1.23305848e+00 -3.35439415e+00  3.31695639e+00
 -1.04539022e-02  1.28114270e+00 -3.69412295e+00  3.66315363e+00
  8.66345346e-03  1.42111106e+00 -3.67260061e+00  7.08407595e+00
  2.16836592e-03  1.01321015e+00 -3.12319453e+00  3.00332167e+00
  2.36250725e-02  1.15502320e+00 -3.47847475e+00  3.32102947e+00
  4.61508627e-04  1.10288272e+00 -3.27976480e+00  2.89815658e+00
 -6.24940240e-02  1.05812164e+00 -2.84947380e+00  3.33201927e+00
 -1.40023350e-02  1.17385284e+00 -3.47843894e+00  3.58808667e+00
 -1.57254934e+02  8.15750362e+00 -1.71003389e+02 -1.42678332e+02
  1.44422660e+02  7.34203951e+00  1.26240608e+02  1.63168704e+02
 -4.92040476e+00  5.03395969e+00 -1.25260612e+01  9.51627937e+00
  1.23610351e+01  4.66860525e+00  4.22263248e+00  2.18797605e+01
 -5.67103723e+00  4.06709469e+00 -1.37145779e+01  2.22018229e+00
  3.76647723e-01  4.56493739e+00 -8.66911117e+00  8.63512184e+00
 -1.41698531e+00  5.96657749e+00 -1.28290273e+01  9.50483112e+00
 -4.31989705e+00  4.82689809e+00 -1.48502149e+01  5.46458028e+00
  6.34927992e-01  5.45705243e+00 -7.96074426e+00  1.32769684e+01
 -6.17599774e+00  6.93921332e+00 -1.88669396e+01  8.52366660e+00
  1.76801147e-01  9.06231226e+00 -1.87951066e+01  1.64550380e+01
 -3.57465541e+00  4.66929421e+00 -1.14371796e+01  7.17427393e+00
  1.53407910e+00  5.88221476e+00 -1.06441092e+01  1.16696412e+01
  5.87859390e-01  1.49066309e+00 -2.24008980e+00  2.19625570e+00
 -8.45257624e-01  7.11451479e-01 -1.87639017e+00  4.49441238e-01
 -2.29521559e-01  7.75748977e-01 -1.49287928e+00  6.23233705e-01
 -1.77327317e-01  8.01806811e-01 -1.45532327e+00  8.60274101e-01
  7.79096576e-03  4.61577088e-01 -1.02139930e+00  6.24066220e-01
  1.64719589e-01  4.90227178e-01 -4.70690099e-01  1.33647417e+00
 -3.67534321e-01  1.27428335e+00 -2.14058332e+00  1.51687597e+00
  2.72560273e-01  8.18357411e-01 -1.45882551e+00  1.39396524e+00
 -9.53725640e-02  9.67233684e-01 -1.96530535e+00  1.58766944e+00
  5.13297882e-01  1.53913273e+00 -2.33261225e+00  2.44908627e+00
 -1.21250120e+00  8.86200159e-01 -2.16634717e+00  2.45472796e-01
  3.42126240e-01  6.96948683e-01 -5.68975092e-01  2.03064252e+00
 -8.48705848e-01  1.16993003e+00 -2.44818784e+00  1.47183762e+00]
-----------------------------------
TEXT FEATURIZING - NLTK_FEATURES
-----------------------------------
[ 0.     0.     2.     1.     6.     0.     0.     3.    10.     0.
  0.     3.     0.     1.     5.     0.     0.     2.     5.     6.
  0.     1.     1.     0.     1.     0.    10.     0.     1.     0.
  0.     0.     0.     1.     8.     0.     0.     0.     0.    31.
  6.     0.     0.     0.     0.     0.     0.     0.     0.     0.
  0.     0.     0.     0.     0.     3.     2.     0.     0.     0.
  0.455  0.845  1.   ]
-----------------------------------
IMAGE FEATURIZING - IMAGE_FEATURES
-----------------------------------
[3.20000000e+02 3.00000000e+00 2.30400000e+05 3.00000000e+02
 5.89482361e+02 8.90000000e+01 7.62900000e+03 3.00000000e+02
 1.42791089e+03 0.00000000e+00 2.26470000e+04 3.00000000e+02
 3.24862213e+02 1.30000000e+01 3.28200000e+03 0.00000000e+00
 0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
 0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
 0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
 0.00000000e+00 0.00000000e+00 2.00000000e+00 0.00000000e+00
 2.89650960e+01 2.21605585e+01 1.88987784e+01 1.86108202e+01
 1.92600349e+01 1.60069808e+01 1.57609075e+01 2.04781850e+01
 5.22111693e+01 3.07120419e+01 2.02862129e+01 1.92687609e+01
 2.23699825e+01 2.11047120e+01 2.26108202e+01 3.28987784e+01
 4.58917976e+01 2.24712042e+01 1.77993019e+01 2.08289703e+01
 2.82931937e+01 2.50104712e+01 2.96492147e+01 3.75322862e+01
 2.45497382e+01 1.83542757e+01 1.59354276e+01 1.70488656e+01
 2.18970332e+01 2.35881326e+01 2.74293194e+01 2.51553229e+01
 4.02600349e+01 2.77469459e+01 2.39703316e+01 2.31099476e+01
 2.65392670e+01 1.99406632e+01 1.59197208e+01 2.30000000e+01
 8.84677138e+01 3.68656195e+01 2.45270506e+01 2.67137871e+01
 2.98446771e+01 2.37469459e+01 2.35148342e+01 4.34520070e+01
 7.92914485e+01 3.24991274e+01 1.68359511e+01 2.35270506e+01
 4.06055846e+01 3.87591623e+01 3.24467714e+01 4.26998255e+01
 3.10139616e+01 1.88621291e+01 1.65619546e+01 2.33315881e+01
 3.68429319e+01 3.40104712e+01 2.94048866e+01 2.85671902e+01
 3.96020942e+01 2.46195462e+01 1.77399651e+01 1.91849913e+01
 2.49598604e+01 2.33577661e+01 2.27242583e+01 2.53106457e+01
 8.88237347e+01 4.50593368e+01 2.36806283e+01 2.53350785e+01
 2.86230366e+01 2.60959860e+01 2.28778360e+01 3.49249564e+01
 8.12949389e+01 4.43071553e+01 3.11012216e+01 3.85061082e+01
 4.02547993e+01 2.35165794e+01 1.70383944e+01 3.27696335e+01
 3.43193717e+01 2.66893543e+01 2.74973822e+01 3.23909250e+01
 3.63909250e+01 2.47242583e+01 1.83699825e+01 2.19424084e+01
 2.61483421e+01 2.09022688e+01 1.70907504e+01 1.40244328e+01
 1.72705061e+01 1.84136126e+01 2.06806283e+01 2.12984293e+01
 4.97277487e+01 3.38900524e+01 2.35986038e+01 1.89109948e+01
 2.09371728e+01 1.86579407e+01 2.11029668e+01 3.06055846e+01
 4.73298429e+01 4.08045375e+01 3.09895288e+01 2.28568935e+01
 2.48865620e+01 1.94031414e+01 1.85567190e+01 2.50471204e+01
 2.62024433e+01 2.72024433e+01 2.69755672e+01 2.34171030e+01
 2.07155323e+01 1.52809773e+01 1.49877836e+01 1.89633508e+01]
-----------------------------------
VIDEO FEATURIZING - video_features
-----------------------------------
pygame 1.9.4
Hello from the pygame community. https://www.pygame.org/contribute.html
/Users/jimschwoebel/Desktop/Allie/load_dir/output
ffmpeg version 3.4.1 Copyright (c) 2000-2017 the FFmpeg developers
  built with Apple LLVM version 9.0.0 (clang-900.0.39.2)
  configuration: --prefix=/usr/local/Cellar/ffmpeg/3.4.1 --enable-shared --enable-pthreads --enable-version3 --enable-hardcoded-tables --enable-avresample --cc=clang --host-cflags= --host-ldflags= --enable-gpl --enable-libmp3lame --enable-libx264 --enable-libxvid --enable-opencl --enable-videotoolbox --disable-lzma
  libavutil      55. 78.100 / 55. 78.100
  libavcodec     57.107.100 / 57.107.100
  libavformat    57. 83.100 / 57. 83.100
  libavdevice    57. 10.100 / 57. 10.100
  libavfilter     6.107.100 /  6.107.100
  libavresample   3.  7.  0 /  3.  7.  0
  libswscale      4.  8.100 /  4.  8.100
  libswresample   2.  9.100 /  2.  9.100
  libpostproc    54.  7.100 / 54.  7.100
Input #0, mov,mp4,m4a,3gp,3g2,mj2, from '5_video.mp4':
  Metadata:
    major_brand     : isom
    minor_version   : 512
    compatible_brands: isomiso2avc1mp41
    encoder         : Lavf58.20.100
  Duration: 00:00:10.02, start: 0.000000, bitrate: 401 kb/s
    Stream #0:0(und): Video: h264 (High) (avc1 / 0x31637661), yuv420p, 320x240 [SAR 1:1 DAR 4:3], 259 kb/s, 29.97 fps, 29.97 tbr, 30k tbn, 59.94 tbc (default)
    Metadata:
      handler_name    : VideoHandler
    Stream #0:1(und): Audio: aac (LC) (mp4a / 0x6134706D), 44100 Hz, stereo, fltp, 133 kb/s (default)
    Metadata:
      handler_name    : IsoMedia File Produced by Google, 5-11-2011
Stream mapping:
  Stream #0:1 -> #0:0 (aac (native) -> pcm_s16le (native))
Press [q] to stop, [?] for help
Output #0, wav, to '5_video.wav':
  Metadata:
    major_brand     : isom
    minor_version   : 512
    compatible_brands: isomiso2avc1mp41
    ISFT            : Lavf57.83.100
    Stream #0:0(und): Audio: pcm_s16le ([1][0][0][0] / 0x0001), 44100 Hz, stereo, s16, 1411 kb/s (default)
    Metadata:
      handler_name    : IsoMedia File Produced by Google, 5-11-2011
      encoder         : Lavc57.107.100 pcm_s16le
size=    1724kB time=00:00:10.00 bitrate=1411.3kbits/s speed= 568x    
video:0kB audio:1724kB subtitle:0kB other streams:0kB global headers:0kB muxing overhead: 0.004418%
making 0.wav
making 1.wav
making 2.wav
making 3.wav
making 4.wav
making 5.wav
making 6.wav
making 7.wav
making 8.wav
making 9.wav
making 10.wav
making 11.wav
making 12.wav
making 13.wav
making 14.wav
making 15.wav
making 16.wav
making 17.wav
making 18.wav
what
-----------------------------------
CSV FEATURIZING - csv_features
-----------------------------------
16
test
['test']
iMac
['iMac', 'iPhone 7', 'iPhone 8']
test
['test']
iMac
['iMac', 'iPhone 7', 'iPhone 8']
test
['test']
iMac
['iMac', 'iPhone 7', 'iPhone 8']
test
['test']
iMac
['iMac', 'iPhone 7', 'iPhone 8']
test
['test']
iMac
['iMac', 'iPhone 7', 'iPhone 8']
test
['test']
iPhone 7
['iMac', 'iPhone 7', 'iPhone 8']
test
['test']
iPhone 7
['iMac', 'iPhone 7', 'iPhone 8']
test
['test']
iPhone 7
['iMac', 'iPhone 7', 'iPhone 8']
test
['test']
iPhone 8
['iMac', 'iPhone 7', 'iPhone 8']
test
['test']
iPhone 7
['iMac', 'iPhone 7', 'iPhone 8']
test
['test']
iPhone 7
['iMac', 'iPhone 7', 'iPhone 8']
test
['test']
iPhone 7
['iMac', 'iPhone 7', 'iPhone 8']
test
['test']
iPhone 7
['iMac', 'iPhone 7', 'iPhone 8']
test
['test']
iPhone 7
['iMac', 'iPhone 7', 'iPhone 8']
test
['test']
iPhone 7
['iMac', 'iPhone 7', 'iPhone 8']
test
['test']
iPhone 7
['iMac', 'iPhone 7', 'iPhone 8']
[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 1. 0. 1. 0. 2. 0. 1. 0. 1. 0. 1.
 0. 1. 0. 1. 0. 1. 0. 1.]
-----------------------------------
AUDIO MODELING - STANDARD_FEATURES
-----------------------------------
['2_audio.json']
keras_models
one_two_keras.h5
1
-----------------------------------
TEXT MODELING - NLTK_FEATURES
-----------------------------------
['1_text.json']
-----------------------------------
IMAGE MODELING - IMAGE_FEATURES
-----------------------------------
['3_image.json']
-----------------------------------
VIDEO MODELING - VIDEO_FEATURES
-----------------------------------
['5_video.json']
-----------------------------------
CSV MODELING - CSV_FEATURES
-----------------------------------
['0_csv.json']
```

This will then load the directory and apply all models in the directory according to sample types (audio_models, text_models, image_models, video_models, csv_models).

## Model schema 

Recall that the standard sample schema is as follows:

```python3
data={'sample', sampletype,
      'features', features,
      'transcriptions': transcripts,
      'labels': labels,
      'models': models}
```

The models part of this schema is as follows:

```python3
models={'audio': audio_models,
        'text': text_models,
        'image': image_models,
        'video': video_models,
        'csv': csv_models}
```

This allows for a flexible definition of models as arrays. As features are put into the load_dir, featurized, and then modeled, they can be updated with classes in the array such that new labels can be tagged onto the data samples.

## Audio models
N/A - no audio models trained yet. 

## Text models 
N/A - no text models trained yet. 

## Image models
N/A - no image models trained yet. 

## Video models
N/A - no video models trained yet. 

## CSV models 
N/A - no .CSV models trained yet. 

## References
* [AudioSet models](https://github.com/jim-schwoebel/audioset_models) - 📊 Easily apply 527 machine learning models trained on AudioSet.
