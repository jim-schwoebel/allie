'''
How to use
python3 create_dataset.py [csvfile] [targetname]
python3 create_dataset.py What_is_your_total_household_income.csv 'What is your total household income?'
'''
import pandas as pd
import numpy as np
import os, shutil, sys

csvfile=sys.argv[1]
target=sys.argv[2]

data=pd.read_csv(csvfile)
urls=(data['url'])
data_values=list(data[target])
os.mkdir(target+'_above')
os.mkdir(target+'_below')
average=float(np.average(np.array(data_values)))

for i in range(len(data)):
	if data_values[i] > average:
		shutil.copy(urls[i], os.getcwd()+'/'+target+'_above'+'/'+urls[i].split('/')[-1])
		try:
			shutil.copy(urls[i][0:-4]+'.json', os.getcwd()+'/'+target+'_above'+'/'+urls[i].split('/')[-1][0:-4]+'.json')
		except:
			pass
	else:
		shutil.copy(urls[i], os.getcwd()+'/'+target+'_below'+'/'+urls[i].split('/')[-1])
		try:
			shutil.copy(urls[i][0:-4]+'.json', os.getcwd()+'/'+target+'_below'+'/'+urls[i].split('/')[-1][0:-4]+'.json')
		except:
			pass