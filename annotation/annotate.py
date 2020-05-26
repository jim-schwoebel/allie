'''
annotate.py

Annotate audio, text, image, or video files for use with regression modeling in Allie.

All you need is a folder, which identifies the type of file within it, and then it goes
through each file to annotate (as .JSON)
'''
import os
from optparse import OptionParser

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

def annotate_file(class_, filetype, file):
	# now go through and annotate each file
	if filetype == 'audio':
		print('playing file... %s'%(file.upper()))
		os.system('play "%s"'%(file))
		annotation = input('%s value?'%(class_.upper))
	elif filetype == 'image':
		print('opening file... %s'%(file.upper()))
		os.system('open "%s"'%(file))
		annotation = input('%s value?'%(class_.upper))
	elif filetype == 'video':
		print('playing file... %s'%(file.upper()))
		os.system('open "%s"'%(file))
		annotation = input('%s value?'%(class_.upper))
	elif filetype == 'text':
		print('opening file... %s'%(file.upper()))
		os.system('open "%s"'%(file))
		annotation = input('%s value?'%(class_.upper))
	else:
		print('file type not supported for annotation')
		annotation='0'

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

	return annotation

# get all the options from the terminal
parser = OptionParser()
parser.add_option("-d", "--directory", dest="annotate_dir",
                  help="the directory to annotate", metavar="annotate_DIR")
parser.add_option("-s", "--sampletype", dest="sampletype",
				  help="specify the type of model to make predictions (e.g. audio, text, image, video, csv)", metavar="SAMPLETYPE")
(options, args) = parser.parse_args()

class_='stress level'
# get annotate directory and sampletype
annotate_dir=options.annotate_dir
os.chdir(annotate_dir)
sampletype=options.sampletype
listdir=os.listdir()
if sampletype == None:
	sampletype = classifyfolder(listdir)

for i in range(len(listdir)):
	annotation=annotate_file(class_, sampletype, listdir[i])
	print(annotation)