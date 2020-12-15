'''
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
|  ___|       | |                        / _ \ | ___ \_   _|  _
| |_ ___  __ _| |_ _   _ _ __ ___  ___  / /_\ \| |_/ / | |   (_)
|  _/ _ \/ _` | __| | | | '__/ _ \/ __| |  _  ||  __/  | |
| ||  __/ (_| | |_| |_| | | |  __/\__ \ | | | || |    _| |_   _
\_| \___|\__,_|\__|\__,_|_|  \___||___/ \_| |_/\_|    \___/  (_)


  ___            _ _
 / _ \          | (_)
/ /_\ \_   _  __| |_  ___
|  _  | | | |/ _` | |/ _ \
| | | | |_| | (_| | | (_) |
\_| |_/\__,_|\__,_|_|\___/

This will featurize folders of audio files if the default_audio_features = ['praat_time_features']

These are the time series features for Praat here.

Inspired by https://github.com/drfeinberg/genderless -
Praat features that are not affected by changing genders.
'''
import glob, os, json
import parselmouth
from parselmouth.praat import call
import numpy as np


def praat_featurize(wav_file):
    voiceID = wav_file
    sound = parselmouth.Sound(voiceID)  # read the sound
    broad_pitch = call(sound, "To Pitch", 0.0, 50, 600)  # create a praat pitch object
    minF0 = call(broad_pitch, "Get minimum", 0, 0, "hertz", "Parabolic")  # get min pitch
    maxF0 = call(broad_pitch, "Get maximum", 0, 0, "hertz", "Parabolic")  # get max pitch
    floor = minF0 * 0.9
    ceiling = maxF0 * 1.1
    pitch = call(sound, "To Pitch", 0.0, floor, ceiling)  # create a praat pitch object
    duration = call(sound, "Get total duration")  # duration
    meanF0 = call(pitch, "Get mean", 0, 0, "hertz")  # get mean pitch
    stdevF0 = call(pitch, "Get standard deviation", 0, 0, "hertz")  # get standard deviation
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, minF0, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)
    pointProcess = call(sound, "To PointProcess (periodic, cc)", minF0, maxF0)
    localJitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    localabsoluteJitter = call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    rapJitter = call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    ppq5Jitter = call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
    ddpJitter = call(pointProcess, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)
    localShimmer = call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    localdbShimmer = call([sound, pointProcess], "Get shimmer (local_dB)",
                          0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq3Shimmer = call([sound, pointProcess], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    aqpq5Shimmer = call([sound, pointProcess], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq11Shimmer = call([sound, pointProcess], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    ddaShimmer = call([sound, pointProcess], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

    if meanF0 > 170 and meanF0 < 300:
        max_formant = 5500
    elif meanF0 <= 170:
        max_formant = 5000
    elif meanF0 >= 300:
        max_formant = 8000
    formants = call(sound, "To Formant (burg)", 0.0025, 5, max_formant, 0.025, 50)
    numPoints = call(pointProcess, "Get number of points")

    f1_list = []
    f2_list = []
    f3_list = []
    f4_list = []
    b1_list = [] #ER 202012
    b2_list = [] #ER 202012
    b3_list = [] #ER 202012
    b4_list = [] #ER 202012

    # Measure formants only at glottal pulses
    for point in range(0, numPoints):
        point += 1
        t = call(pointProcess, "Get time from index", point)
        f1 = call(formants, "Get value at time", 1, t, 'Hertz', 'Linear')
        f2 = call(formants, "Get value at time", 2, t, 'Hertz', 'Linear')
        f3 = call(formants, "Get value at time", 3, t, 'Hertz', 'Linear')
        f4 = call(formants, "Get value at time", 4, t, 'Hertz', 'Linear')
        f1_list.append(f1)
        f2_list.append(f2)
        f3_list.append(f3)
        f4_list.append(f4)
        f1_list.append(f1)
        f2_list.append(f2)
        f3_list.append(f3)
        f4_list.append(f4)

        b1 = call(formants, "Get bandwidth at time", 1, t, 'Hertz', 'Linear') #ER 202012
        b2 = call(formants, "Get bandwidth at time", 2, t, 'Hertz', 'Linear') #ER 202012
        b3 = call(formants, "Get bandwidth at time", 3, t, 'Hertz', 'Linear') #ER 202012
        b4 = call(formants, "Get bandwidth at time", 4, t, 'Hertz', 'Linear') #ER 202012
        b1_list.append(b1)
        b2_list.append(b2)
        b3_list.append(b3)
        b4_list.append(b4)

    f1_all = np.asarray(f1_list) #ER 202012
    f2_all = np.asarray(f2_list) #ER 202012
    f3_all = np.asarray(f3_list) #ER 202012
    f4_all = np.asarray(f4_list) #ER 202012

    b1_all = np.asarray(b1_list) #ER 202012
    b2_all = np.asarray(b2_list) #ER 202012
    b3_all = np.asarray(b3_list) #ER 202012
    b4_all = np.asarray(b4_list) #ER 202012

    f1_list = [f1 for f1 in f1_list if str(f1) != 'nan']
    f2_list = [f2 for f2 in f2_list if str(f2) != 'nan']
    f3_list = [f3 for f3 in f3_list if str(f3) != 'nan']
    f4_list = [f4 for f4 in f4_list if str(f4) != 'nan']
    # calculate mean formants across pulses
    if len(f1_list) > 0:
        f1_mean = sum(f1_list) / len(f1_list)
    else:
        f1_mean = 0
    if len(f2_list) > 0:
        f2_mean = sum(f2_list) / len(f2_list)
    else:
        f2_mean = 0
    if len(f3_list) > 0:
        f3_mean = sum(f3_list) / len(f3_list)
    else:
        f3_mean = 0
    if len(f4_list) > 0:
        f4_mean = sum(f4_list) / len(f4_list)
    else:
        f4_mean = 0

    intensity = sound.to_intensity()  #ER 202012

    measurements = [duration, meanF0, stdevF0, hnr, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter,
                    localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer,
                    f1_mean, f2_mean, f3_mean, f4_mean]

    labels = ['duration', 'meanF0', 'stdevF0', 'hnr', 'localJitter', 'localabsoluteJitter', 'rapJitter', 'ppq5Jitter',
              'ddpJitter',
              'localShimmer', 'localdbShimmer', 'apq3Shimmer', 'aqpq5Shimmer', 'apq11Shimmer', 'ddaShimmer', 'f1_mean',
              'f2_mean', 'f3_mean', 'f4_mean']

    measurements_time_series = [pitch.selected_array['frequency'], pitch.selected_array['strength'], pitch.ts(),
                                harmonicity.as_array()[0], harmonicity.ts(),
                                f1_all, f2_all, f3_all, f4_all, b1_all, b2_all, b3_all, b4_all, formants.ts(),
                                intensity.as_array()[0], intensity.ts()] #ER 202012

    labels_time_series = ['pitch_time_series', 'pitch_strength_time_series', 'pitch_t_time_series',
                          'harmonicity_time_series', 'harmonicity_t_time_series',
                          'formant1_time_series', 'f2_time_series', 'f3_time_series', 'f4_time_series', 
                          'bandwidth1_time_series', 'b2_time_series', 'b3_time_series', 'b4_time_series', 'formants_t_time_series',
                          'intensity_time_series', 'intensity_t_time_series'] #ER 202012


    features=measurements
    for i in range(len(measurements_time_series)):
        features.append(list(measurements_time_series[i]))

    labels=labels+labels_time_series

    return features, labels

# features, labels = praat_featurize('test.wav')
# print(labels)
# data=dict()
# for i in range(len(labels)):
#   data[labels[i]]=features[i]

# g=open('test.json','w')
# json.dump(data,g)
# g.close()

# print(features)
# print(labels)
