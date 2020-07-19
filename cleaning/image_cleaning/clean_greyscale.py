import numpy as np
from PIL import Image
import os, cv2

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def clean_greyscale(imagefile):
	img = cv2.imread(imagefile)
	os.remove(imagefile)
	gray = rgb2gray(img)    
	cv2.imwrite(imagefile, gray)