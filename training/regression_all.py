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
                                                                             
|  \/  |         | |    | |  / _ \ | ___ \_   _|
| .  . | ___   __| | ___| | / /_\ \| |_/ / | |  
| |\/| |/ _ \ / _` |/ _ \ | |  _  ||  __/  | |  
| |  | | (_) | (_| |  __/ | | | | || |    _| |_ 
\_|  |_/\___/ \__,_|\___|_| \_| |_/\_|    \___/ 

Builds many regression models based on .CSV files located in the train_dir.

Relies on this script: https://github.com/jim-schwoebel/allie/blob/master/train_dir/make_csv_regression.py

Note this is for single target regression problems only.
'''
import os, shutil, time
import pandas as pd

def prev_dir(directory):
	g=directory.split('/')
	dir_=''
	for i in range(len(g)):
		if i != len(g)-1:
			if i==0:
				dir_=dir_+g[i]
			else:
				dir_=dir_+'/'+g[i]
	# print(dir_)
	return dir_

def pursue_modeling(mtype, model_dir, problemtype, default_training_script, common_name_model):
	'''
	simple script to decide whether or not to continue modeling the data.
	'''
	try:
		model_listdir=os.listdir(model_dir+'/'+problemtype+'_models')
	except:	
		model_listdir=list()

	# note that these are tpot definitions
	model_exists=False
	if default_training_script == 'tpot':
		if common_name_model + 'tpot_classifier' in model_listdir and mtype == 'c':
			model_exists=True
		elif common_name_model +'tpot_regression' in model_listdir and mtype == 'r':
			model_exists=True
	else:
		# only look for naming conflicts with TPOT for now, can expand into the future.
		model_exists=False

	return model_exists, model_listdir

def replace_nonstrings(string_):
	# alphabet to keep characters
	alphabet=['a','b','c','d','e','f','g','h','i','j',
			  'k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',' ','_',
			  '1','2','3','4','5','6','7','8','9','0']
	string_=string_.lower().replace(' ','_')
	newstring=''
	for j in range(len(string_)):
		if string_[j] not in alphabet:
			pass
		else:
			newstring=newstring+string_[j]

	if len(newstring) > 50:
		newstring=newstring[0:50]
		
	return newstring

curdir=os.getcwd()
prevdir=prev_dir(curdir)
train_dir=prevdir+'/train_dir'
model_dir=prevdir+'/models'
problemtype='audio'
default_training_script='tpot'
os.chdir(train_dir)
listdir=os.listdir()
csvfiles=list()
for i in range(len(listdir)):
	if listdir[i].endswith('.csv'):
		csvfiles.append(listdir[i])

# now train models
os.chdir(curdir)

for i in range(len(csvfiles)):
	os.chdir(train_dir)
	data=pd.read_csv(csvfiles[i])
	class_=list(data)[1]
	os.chdir(curdir)
	uniquevals=list(set(list(data[class_])))
	common_name=replace_nonstrings(class_)

	# make regression model (if doesn't already exist)
	# model_exists, model_listdir = pursue_modeling('r', model_dir, problemtype, default_training_script, common_name)
	# if model_exists == False:
	# 	os.system('python3 model.py r "%s" "%s" "%s"'%(csvfiles[i], class_, common_name))
	# else:
	# 	print('skipping - %s'%(common_name+'_tpot_regression'))

	os.chdir(prevdir+'/train_dir/')
	
	# make classification model (allows for visualizations and averages around mean)
	model_exists, model_listdir = pursue_modeling('c', model_dir, problemtype, default_training_script, common_name)
	if model_exists == False:
		os.system('python3 create_dataset.py "%s" "%s"'%(csvfiles[i], class_))
		os.chdir(curdir)
		if len(uniquevals) > 10:
			os.system('python3 model.py audio 2 c "%s" "%s" "%s"'%(common_name, replace_nonstrings(class_)+'_above', replace_nonstrings(class_)+'_below'))
			# remove temporary directories for classification model training
			try:
				shutil.rmtree(prevdir+'/train_dir/'+class_+'_above')
			except:
				pass
			try:
				shutil.rmtree(prevdir+'/train_dir/'+class_+'_below')
			except:
				pass

		else:
			command='python3 model.py audio %s c "%s"'%(str(len(uniquevals)), common_name)
			for j in range(len(uniquevals)):
				newstring=replace_nonstrings(str(uniquevals[j]))
				command=command+' '+'"%s"'%(common_name+'_'+newstring)
			os.system(command)
			for j in range(len(uniquevals)):
				try:
					newstring=replace_nonstrings(str(uniquevals[j]))
					shutil.rmtree(prevdir+'/train_dir/'+common_name+'_'+newstring)
				except:
					pass
	else:
		print('skipping - %s'%(common_name+'_tpot_classifier'))