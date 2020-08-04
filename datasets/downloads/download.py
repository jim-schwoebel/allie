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

______      _                 _       
|  _  \    | |               | |      
| | | |__ _| |_ __ _ ___  ___| |_ ___ 
| | | / _` | __/ _` / __|/ _ \ __/ __|
| |/ / (_| | || (_| \__ \  __/ |_\__ \
|___/ \__,_|\__\__,_|___/\___|\__|___/
                                      

A command line interface for downloading datasets through Allie.

Specify the dataset type and get links and download information.

Note this is a work-in-progress and will expand into the future. 
'''
from fuzzywuzzy import fuzz
import os, json 

current_dir=os.getcwd()

# now ask user what type of problem they are trying to solve 
problemtype=input('what dataset would you like to download? (1-audio, 2-text, 3-image, 4-video, 5-csv)\n')
while problemtype not in ['1','2','3','4','5']:
	print('answer not recognized...')
	problemtype=input('what problem are you solving? (1-audio, 2-text, 3-image, 4-video, 5-csv)\n')

if problemtype=='1':
	problemtype='audio'
elif problemtype=='2':
	problemtype='text'
elif problemtype=='3':
	problemtype='image'
elif problemtype=='4':
	problemtype='video'
elif problemtype=='5':
	problemtype=='csv'

# go to helpers directory + get json 
if problemtype=='audio':
	os.chdir(current_dir+'/helpers/audio')
elif problemtype == 'text':
	os.chdir(current_dir+'/helpers/text')
elif problemtype=='image': 
	os.chdir(current_dir+'/helpers/image')
elif problemtype=='video':
	os.chdir(current_dir+'/helpers/video')
elif problemtype=='csv':
	# csv is scarcest dataset 
	os.chdir(current_dir+'/helpers/csv')

# now get all the json files in the directory 
listdir=os.listdir()
dataset_names=list()
for i in range(len(listdir)):
	if listdir[i][-5:]=='.json':
		dataset_names.append(listdir[i][0:-5].replace('_', ' '))

print('found %s datasets...'%(str(len(dataset_names))))
print('----------------------------')
print('here are the available %s datasets'%(problemtype.upper()))
print('----------------------------')
for i in range(len(dataset_names)):
	print(dataset_names[i])

while True:
	user_input=input('what %s dataset would you like to download?\n'%(problemtype))
	fuzznums=list()
	for i in range(len(dataset_names)):
		# do fuzzy search to find best matched dataset to query (continue to download in ./data directory)
		fuzznums.append(fuzz.ratio(dataset_names[i].lower(), user_input.lower()))
	maxval=max(fuzznums)
	maxind=fuzznums.index(maxval)
	dataset=dataset_names[maxind]
	print('found dataset: %s'%(dataset))
	g=json.load(open(dataset.replace(' ','_')+'.json'))
	print(g[dataset]['description'])
	user_input2=input('just confirming, do you want to download the %s dataset? (Y - yes, N - no) \n'%(dataset))
	if user_input2.lower() in ['y','yes']:
		os.system('open %s'%(g[dataset]['link']))
		break

