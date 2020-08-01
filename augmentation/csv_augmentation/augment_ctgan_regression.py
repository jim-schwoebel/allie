from ctgan import CTGANSynthesizer
import time, random
import pandas as pd
import numpy as np

def augment_ctgan_regression(csvfile):
	data=pd.read_csv(csvfile)

	ctgan = CTGANSynthesizer()
	ctgan.fit(data,epochs=10) #15

	percent_generated=1
	df_gen = ctgan.sample(int(len(data)*percent_generated))

	print('augmented with %s samples'%(str(len(df_gen))))
	print(df_gen)
	# now add both togrther to make new .CSV file
	df_gen.to_csv('augmented_'+csvfile, index=0)

	# now combine augmented and regular dataset
	data2=pd.read_csv('augmented_'+csvfile)
	frames = [data, data2]
	result = pd.concat(frames)
	result.to_csv('augmented_combined_'+csvfile, index=0)

# augment_ctgan_regression('gender_all.csv')