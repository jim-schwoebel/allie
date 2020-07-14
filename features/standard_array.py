''' 
This is the standard feature array for Allie (version 1.0).

Note this will be imported to get back data in all featurization methods
to ensure maximal code reusability
'''
import os, time, psutil, json, platform
from datetime import datetime
		
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

def make_features(sampletype):

	# only add labels when we have actual labels.
	features={'audio':dict(),
		      'text': dict(),
		      'image':dict(),
		      'video':dict(),
		      'csv': dict()}

	transcripts={'audio': dict(),
				 'text': dict(),
				 'image': dict(),
				 'video': dict(),
				 'csv': dict()}
			   
	models={'audio': dict(),
			'text': dict(),
			'image': dict(),
			'video': dict(),
			'csv': dict()}
	
	# getting settings can be useful to see if settings are the same in every
	# featurization, as some featurizations can rely on certain settings to be consistent
	prevdir=prev_dir(os.getcwd())
	settings=json.load(open(prevdir+'/settings.json'))
	
	data={'sampletype': sampletype,
		  'transcripts': transcripts,
		  'features': features,
		  'models': models,
		  'labels': [],
		  'errors': [],
		  'settings': settings,
		 }
	
	return data