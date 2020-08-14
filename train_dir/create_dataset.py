import pandas as pd
import os, shutil

data=pd.read_csv('pga1_lupus.csv')
urls=(data['url'])
sleqols=list(data['pga1_lupus_active'])
os.mkdir('pga1_lupus_active_above5.65')
os.mkdir('pga1_lupus_active_below5.65')
for i in range(len(data)):
	if sleqols[i] > 5.65:
		shutil.copy(urls[i], os.getcwd()+'/'+'pga1_lupus_active_above5.65'+'/'+urls[i].split('/')[-1])
	else:
		shutil.copy(urls[i], os.getcwd()+'/'+'pga1_lupus_active_below5.65'+'/'+urls[i].split('/')[-1])