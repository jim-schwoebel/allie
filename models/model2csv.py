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

|  \/  |         | |    | |    
| .  . | ___   __| | ___| |___ 
| |\/| |/ _ \ / _` |/ _ \ / __|
| |  | | (_) | (_| |  __/ \__ \
\_|  |_/\___/ \__,_|\___|_|___/

Creates an excel sheet of all currently trained models with their model performances; 
useful to summarize all modeling sessions quickly; outputs to current directory.

Usage: python3 model2csv.py
'''

import os, json
import pandas as pd

def id_folder():
	curdir=os.getcwd()
	directories=['audio_models', 'text_models', 'image_models', 'video_models', 'csv_models']
	metrics_list=list()
	model_names=list()

	for i in range(len(directories)):
		try:
			os.chdir(curdir)
			os.chdir(directories[i])
			listdir=os.listdir()

			folders=list()
			for j in range(len(listdir)):
				if listdir[j].find('.') < 0:
					folders.append(listdir[j])

			curdir2=os.getcwd()

			for j in range(len(folders)):
				os.chdir(curdir2)
				os.chdir(folders[j])
				os.chdir('model')
				listdir2=os.listdir()
				jsonfile=folders[j]+'.json'
				for k in range(len(listdir2)):
					if listdir2[k] == jsonfile:
						g=json.load(open(jsonfile))
						metrics_=g['metrics']
						metrics_list.append(metrics_)
						model_names.append(jsonfile[0:-5])
		except:
			pass 
			# print(directories[i])
			# print('does not exist...')

	return metrics_list, model_names

curdir=os.getcwd()
metrics_list, model_names=id_folder()

# regression models
meanabsolute_errors=list()
meansquared_errors=list()
median_errors=list()
r2_scores=list()
regression_models=list()

for i in range(len(model_names)):
	try: 
		meanabsolute_errors.append(metrics_list[i]['mean_absolute_error'])
		meansquared_errors.append(metrics_list[i]['mean_squared_error'])
		median_errors.append(metrics_list[i]['median_absolute_error'])
		r2_scores.append(metrics_list[i]['r2_score'])
		regression_models.append(model_names[i])
	except:
		pass

# classification models 
accuracies=list()
roc_curve=list()
classification_models=list()

for i in range(len(model_names)):
	try:
		accuracies.append(metrics_list[i]['accuracy'])
		roc_curve.append(metrics_list[i]['roc_auc'])
		classification_models.append(model_names[i])
	except:
		pass

classification_data={'model names': classification_models,
					  'accuracies': accuracies,
					  'roc_auc': roc_curve}

regression_data={'model_names': regression_models,
				'mean_absolute_errors': meanabsolute_errors,
				'mean_squared_errors': meansquared_errors,
				'r2_scores': r2_scores}

os.chdir(curdir)
df=pd.DataFrame.from_dict(classification_data)
df.to_csv('classification_models.csv', index=False)

df=pd.DataFrame.from_dict(regression_data)
df.to_csv('regression_models.csv', index=False)