'''
annotate.py

Annotate audio, text, image, or video files for use with regression modeling in Allie.

All you need is a folder, which identifies the type of file within it, and then it goes
through each file to annotate (as .JSON)
'''
import os, sys, datetime, json, time
import pandas as pd
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

curdir=os.getcwd()
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
data=dict()
classes=list()
filepaths=list()

for i in range(len(listdir)):
	if listdir[i].endswith('.json'):
		g=json.load(open(listdir[i]))
		labels=g['labels']
		for j in range(len(labels)):
			try:
				if list(labels[j]) == [class_] and labels[j][class_]['problemtype'] == problemtype:
					value=labels[j][class_]['value']
					filepath=labels[j][class_]['annotate_dir']+labels[j][class_]['file']
					print(value)
					print(filepath)
					classes.append(value)
					filepaths.append(filepath)	
			except:
				pass

data[class_]=classes
data['paths']=filepaths

os.chdir(curdir)
df = pd.DataFrame(data)
df.to_csv('%s_data.csv'%(class_), index=False)