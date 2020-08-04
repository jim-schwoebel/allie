'''
annotate.py

Annotate audio, text, image, or video files for use with regression modeling in Allie.

All you need is a folder, which identifies the type of file within it, and then it goes
through each file to annotate (as .JSON)
'''
import os, sys, datetime, json, time
from optparse import OptionParser
from tqdm import tqdm

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

def most_common(lst):
	'''
	get most common item in a list
	'''
	return max(set(lst), key=lst.count)

def classifyfolder(listdir):

    filetypes=list()

    for i in range(len(listdir)):
        if listdir[i].endswith(('.mp3', '.wav')):
            filetypes.append('audio')
        elif listdir[i].endswith(('.png', '.jpg')):
            filetypes.append('image')
        elif listdir[i].endswith(('.txt')):
            filetypes.append('text')
        elif listdir[i].endswith(('.mp4', '.avi')):
            filetypes.append('video')
        elif listdir[i].endswith(('.csv')):
            filetypes.append('csv')

    filetype=most_common(filetypes)

    return filetype

def annotate_file(class_, filetype, file, problemtype):
	# now go through and annotate each file
	if filetype == 'audio':
		print('playing file... %s'%(file.upper()))
		os.system('play "%s"'%(file))
	elif filetype == 'image':
		print('opening file... %s'%(file.upper()))
		os.system('open "%s"'%(file))
	elif filetype == 'video':
		print('playing file... %s'%(file.upper()))
		os.system('open "%s"'%(file))
	elif filetype == 'text':
		print('opening file... %s'%(file.upper()))
		os.system('open "%s"'%(file))
	else:
		print('file type not supported for annotation')
		annotation='0'

	if problemtype in ['r', 'regression']:
		annotation = input('%s value?\n'%(class_.upper()))
	else:
		annotation = input('%s label 1 (yes) or 0 (no)?\n'%(class_.upper()))

	# only get a float back
	try:
		annotation=float(annotation)
	except:
		print('error annotating, annotating again...')
		while True:
			annotation = annotate_file(class_, filetype, file)
			try: 
				annotation=float(annotation)
				break
			except:
				pass

	label=dict()
	label[class_]={'value': annotation, 
				   'datetime': str(datetime.datetime.now()),
				   'filetype': filetype,
				   'file': file,
				   'problemtype': problemtype, 
				   'annotate_dir': annotate_dir}

	annotation=[label]
	print(annotation)

	return annotation

# get all the options from the terminal
parser = OptionParser()
parser.add_option("-d", "--directory", dest="annotate_dir",
                  help="the directory to annotate", metavar="annotate_DIR")
parser.add_option("-s", "--sampletype", dest="sampletype",
				  help="specify the type of model to make predictions (e.g. audio, text, image, video, csv)", metavar="SAMPLETYPE")
parser.add_option("-c", "--classtype", dest="classtype", 
				  help="specify the class type (e.g. stress level", metavar="CLASSTYPE")
parser.add_option("-p", "--problemtype", dest="problemtype",
				  help="specify the problem type (-c classification or -r regression", metavar="PROBLEMTYPE")

(options, args) = parser.parse_args()

prevdir=prev_dir(os.getcwd())+'/features/'
sys.path.append(prevdir)
from standard_array import make_features

# get annotate directory and sampletype
class_=options.classtype
problemtype=options.problemtype
annotate_dir=options.annotate_dir
os.chdir(annotate_dir)
sampletype=options.sampletype
listdir=os.listdir()
if sampletype == None:
	sampletype = classifyfolder(listdir)

listdir=os.listdir()

for i in tqdm(range(len(listdir))):
	try:
		if listdir[i].endswith('.json'):
			pass
		else:
			jsonfilename=listdir[i][0:-4]+'.json'
			if jsonfilename not in listdir:
				annotation=annotate_file(class_, sampletype, listdir[i], problemtype)
				basearray=make_features(sampletype)
				basearray['labels']=annotation
				jsonfile=open(jsonfilename,'w')
				json.dump(basearray, jsonfile)
				jsonfile.close()

			elif jsonfilename in listdir:
				g=json.load(open(jsonfilename))
				labels=g['labels']
				classin=False
				for j in range(len(labels)):
					try:
						print(list(labels[j]))
						print(labels[j][class_]['problemtype'])
						if list(labels[j]) == [class_] and labels[j][class_]['problemtype'] == problemtype:
							classin=True
					except:
						pass

				if classin == True:
					print('skipping %s, already annotated'%(listdir[i]))
				else:
					annotation=annotate_file(class_, sampletype, listdir[i], problemtype)
					labels.append(annotation)
					g['labels']=labels
					jsonfile=open(jsonfilename,'w')
					json.dump(g, jsonfile)
					jsonfile.close()
	except:
		print('error - file %s not recognized'%(listdir[i]))