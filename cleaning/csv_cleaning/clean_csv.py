import os
import pandas as pd
try:
	import datacleaner
except:
	os.system('pip3 install datacleaner==0.1.5')

def clean_csv(csvfile, basedir):
	'''
	https://github.com/rhiever/datacleaner
	'''
	input_dataframe=pd.read_csv(csvfile)
	newframe=datacleaner.autoclean(input_dataframe, drop_nans=False, copy=False, ignore_update_check=False)
	newframe.to_csv('clean_'+csvfile, index=False)
