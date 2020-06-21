'''
Inspired by https://github.com/drfeinberg/genderless - 
Praat features that are not affected by changing genders.

David R. Feinberg - Associate Professor in the Department of Psychology, Neuroscience, & Behaviour, McMaster University
Mcmaster University
Hamilton, Ontario feinberg@mcmaster.ca http://www.voiceresearch.org
'''
import glob, os, json
import parselmouth
from parselmouth.praat import call

def praat_featurize(voiceID):
    voiceID = voiceID
    sound = parselmouth.Sound(voiceID) # read the sound
    broad_pitch = call(sound, "To Pitch", 0.0, 50, 600) #create a praat pitch object
    minF0 = call(broad_pitch, "Get minimum", 0, 0, "hertz", "Parabolic") # get min pitch
    maxF0 = call(broad_pitch, "Get maximum", 0, 0, "hertz", "Parabolic")  # get max pitch
    floor = minF0 * 0.9
    ceiling = maxF0 * 1.1
    pitch = call(sound, "To Pitch", 0.0, floor, ceiling)  # create a praat pitch object
    duration = call(sound, "Get total duration") # duration
    meanF0 = call(pitch, "Get mean", 0, 0, "hertz") # get mean pitch
    stdevF0 = call(pitch, "Get standard deviation", 0 ,0, "hertz") # get standard deviation
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
    elif meanF0 <=170:
        max_formant = 5000
    elif meanF0 >= 300:
        max_formant = 8000
    formants = call(sound, "To Formant (burg)", 0.0025, 5, max_formant, 0.025, 50)
    numPoints = call(pointProcess, "Get number of points")

    f1_list = []
    f2_list = []
    f3_list = []
    f4_list = []

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
    measurements = [duration, meanF0, stdevF0, hnr, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter,
                    localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer,
                    f1_mean, f2_mean, f3_mean, f4_mean]

    labels=['duration', 'meanF0', 'stdevF0', 'hnr', 'localJitter', 'localabsoluteJitter', 'rapJitter', 'ppq5Jitter', 'ddpJitter',
            'localShimmer', 'localdbShimmer', 'apq3Shimmer', 'aqpq5Shimmer', 'apq11Shimmer', 'ddaShimmer', 'f1_mean', 'f2_mean', 'f3_mean', 'f4_mean']

    return measurements, labels 


# # convert all to .JSON
# for voiceID in glob.glob(os.getcwd()+'/*.wav'):
#     features, labels = measure_voices(voiceID)
#     for measure in features:
#         print(measure)

#     jsonfile=open(voiceID[0:-4]+'.json','w')
#     data={'features': features,
#           'labels': labels}
#     json.dump(data,jsonfile)
#     jsonfile.close()
