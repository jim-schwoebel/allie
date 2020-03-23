import numpy as np
from sonopy import power_spec, mel_spec, mfcc_spec, filterbanks
from scipy.io import wavfile

sr, audio = wavfile.read('test.wav')

# powers = power_spec(audio, window_stride=(100, 50), fft_size=512)
# mels = mel_spec(audio, sr, window_stride=(1600, 800), fft_size=1024, num_filt=30)
mfccs = mfcc_spec(audio, sr, window_stride=(160, 80), fft_size=512, num_filt=20, num_coeffs=13)
print(mfccs)
# filters = filterbanks(16000, 20, 257)  # Probably not ever useful

# powers, filters, mels, mfccs = mfcc_spec(audio, sr, return_parts=True)

