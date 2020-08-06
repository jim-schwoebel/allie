import os, librosa
import numpy as np
import random
from tsaug import Crop, AddNoise, Dropout
import soundfile as sf

'''
random crop subsequences of randomly spliced audio file.
with 50% probability, add random noise up to 1% - 5%,
drop out 10% of the time points (dropped out units are 1 ms, 10 ms, or 100 ms) and fill the dropped out points with zeros.
'''

def augment_tsaug(filename):
		'''
		https://tsaug.readthedocs.io/en/stable/
		'''
		y, sr = librosa.load(filename, mono=False)
		duration=int(librosa.core.get_duration(y,sr))
		print(y.shape)
		# y=np.expand_dims(y.swapaxes(0,1), 0)

		# N second splice between 1 second to N-1 secondsd
		splice=random.randint(1,duration-1)

		my_augmenter = (Crop(size=sr * splice) * 5  # random crop subsequences of splice seconds
		+ AddNoise(scale=(0.01, 0.05)) @ 0.5  # with 50% probability, add random noise up to 1% - 5%
		+ Dropout(
		         p=0.1,
		         fill=0,
		         size=[int(0.001 * sr), int(0.01 * sr), int(0.1 * sr)]
		         )  # drop out 10% of the time points (dropped out units are 1 ms, 10 ms, or 100 ms) and fill the dropped out points with zeros
		)
		y_aug = my_augmenter.augment(y)
		
		sf.write('tsaug_'+filename, y_aug.T, sr)
