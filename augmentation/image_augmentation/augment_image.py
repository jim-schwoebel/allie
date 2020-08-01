import os

try:
	import helpers.imgaug.augmenters as iaa
	import helpers.imgaug as ia
except:
	os.system('pip3 install git+https://github.com/aleju/imgaug.git')
	import imaug.augmenters as iaa
	
import imageio
import matplotlib.pyplot as plt

'''
following this tutorial:

https://nbviewer.jupyter.org/github/aleju/imgaug-doc/blob/master/notebooks/A01%20-%20Load%20and%20Augment%20an%20Image.ipynb
'''

def augment_image(imagefile):
	image = imageio.imread(imagefile)

	seq = iaa.Sequential([
	    iaa.Affine(rotate=(-25, 25)),
	    iaa.AdditiveGaussianNoise(scale=(30, 90)),
	    iaa.Crop(percent=(0, 0.4))
	], random_order=True)

	images_aug = [seq.augment_image(image) for _ in range(1)]

	# print("Augmented:")
	# ia.imshow(ia.draw_grid(images_aug, cols=4, rows=2))

	for i in range(len(images_aug)):
	    plt.imsave('augmented_'+imagefile, images_aug[i])