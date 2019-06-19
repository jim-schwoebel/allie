from imgaug.imgaug import augmenters as iaa
import imageio, os
import imgaug as ia
import matplotlib.pyplot as plt

'''
following this tutorial:

https://nbviewer.jupyter.org/github/aleju/imgaug-doc/blob/master/notebooks/A01%20-%20Load%20and%20Augment%20an%20Image.ipynb
'''
image = imageio.imread("https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png")
os.chdir('./data/images')

seq = iaa.Sequential([
    iaa.Affine(rotate=(-25, 25)),
    iaa.AdditiveGaussianNoise(scale=(30, 90)),
    iaa.Crop(percent=(0, 0.4))
], random_order=True)

images_aug = [seq.augment_image(image) for _ in range(8)]

print("Augmented:")
ia.imshow(ia.draw_grid(images_aug, cols=4, rows=2))

for i in range(len(images_aug)):
    plt.imsave(str(i)+'.png', images_aug[i])
