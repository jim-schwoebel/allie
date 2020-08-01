import os
import pandas as pd

try:
	from tgan.model import TGANModel
except:
	os.system('pip3 install tgan==0.1.0')
	
'''
following this tutorial:

https://nbviewer.jupyter.org/github/aleju/imgaug-doc/blob/master/notebooks/A01%20-%20Load%20and%20Augment%20an%20Image.ipynb
'''

def augment_tgan(csvfile):
	data=pd.read_csv(csvfile)
	cols=list(data)
	cols_num=list()
	for i in range(len(cols)-1):
		cols_num.append(i)
	tgan = TGANModel(cols_num)
	tgan.fit(data)

	# now create number of samples (10%)
	num_samples = int(0.10*len(data))
	samples = tgan.sample(num_samples)

	print(samples)


augment_tgan('gender_all.csv')