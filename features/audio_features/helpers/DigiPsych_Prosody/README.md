# DigiPsych Prosody

The DigiPsych Prosody Repository is a public open-source implementation of extracting prosodic features

We are using the WebRTC Voice Activity Detector to create normalized features of Prosody (https://github.com/wiseman/py-webrtcvad)

These Features include:
- Total Speech Time
- Total Pause Time
- Percentage Pause Time
- Speech Pause Time
- Mean Pause Duration
- Pause Variability

In addition to considering the following features, we will also consider the 3 different intensities of the WebRTC Voice Activity Detector to assemble a total of 3 * 7 = 21 Features.

## Background

Prosody Features are a valuable way to identify changes in cognitive function of psychiatric and neurodegenerative illness. Early in the generation of voice, humans formulate the phonation (content of speech) and prosody (the way in which they speak). With cognitive decay/slowing. One may observe instances in changing of speech. Slowing/Slurring of speech may be present.

![alt text](https://raw.githubusercontent.com/larryzhang95/DigiPsych_Prosody/master/img/voicing.png)

For the current look at these features, we seek to address slowing in speech.

## Usage
```
python featurize -a <path to audio files> -f <frame size in milliseconds>
```
- Ideally frame size should be between 20-40 milliseconds for optimal voice activity detection.

## Imports and Dependencies

The following packages are used for acquiring prosodic features:

- py-webrtcvad
