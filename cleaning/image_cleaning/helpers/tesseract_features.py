import os, sys
from PIL import Image
import pytesseract 

def prev_dir(directory):
	g=directory.split('/')
	# print(g)
	lastdir=g[len(g)-1]
	i1=directory.find(lastdir)
	directory=directory[0:i1]
	return directory
	
directory=os.getcwd()
prev_dir=prev_dir(directory)
sys.path.append(prev_dir+'/text_features')
import nltk_features as nf 
os.chdir(directory)

def transcribe_image(imgfile):
	transcript=pytesseract.image_to_string(Image.open(imgfile))
	return transcript 

def tesseract_featurize(imgfile):
	# can stitch across an entire length of video frames too 
	transcript=transcribe_image(imgfile)
	features, labels = nf.nltk_featurize(transcript)
	
	return transcript, features, labels 
