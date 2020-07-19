from PIL import Image
import os

def clean_jpg2png(imagefile):
	if imagefile.endswith('.jpg'):
		im1 = Image.open(imagefile)
		im1.save(imagefile[0:-4]+'.png')
		os.remove(imagefile)