''' 
This is the standard feature array for Allie (version 1.0).

Note this will be imported to get back data in all featurization methods
to ensure maximal code reusability
'''

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
	
	data={'sampletype': sampletype,
	      'transcripts': transcripts,
	      'features': features,
	      'models': models,
	      'labels': [],
	      'errors': []
	     }
	
	return data
