'''
               AAA               lllllll lllllll   iiii                      
              A:::A              l:::::l l:::::l  i::::i                     
             A:::::A             l:::::l l:::::l   iiii                      
            A:::::::A            l:::::l l:::::l                             
           A:::::::::A            l::::l  l::::l iiiiiii     eeeeeeeeeeee    
          A:::::A:::::A           l::::l  l::::l i:::::i   ee::::::::::::ee  
         A:::::A A:::::A          l::::l  l::::l  i::::i  e::::::eeeee:::::ee
        A:::::A   A:::::A         l::::l  l::::l  i::::i e::::::e     e:::::e
       A:::::A     A:::::A        l::::l  l::::l  i::::i e:::::::eeeee::::::e
      A:::::AAAAAAAAA:::::A       l::::l  l::::l  i::::i e:::::::::::::::::e 
     A:::::::::::::::::::::A      l::::l  l::::l  i::::i e::::::eeeeeeeeeee  
    A:::::AAAAAAAAAAAAA:::::A     l::::l  l::::l  i::::i e:::::::e           
   A:::::A             A:::::A   l::::::ll::::::li::::::ie::::::::e          
  A:::::A               A:::::A  l::::::ll::::::li::::::i e::::::::eeeeeeee  
 A:::::A                 A:::::A l::::::ll::::::li::::::i  ee:::::::::::::e  
AAAAAAA                   AAAAAAAlllllllllllllllliiiiiiii    eeeeeeeeeeeeee  

  ___                                   _        _   _             
 / _ \                                 | |      | | (_)            
/ /_\ \_   _  __ _ _ __ ___   ___ _ __ | |_ __ _| |_ _  ___  _ __  
|  _  | | | |/ _` | '_ ` _ \ / _ \ '_ \| __/ _` | __| |/ _ \| '_ \ 
| | | | |_| | (_| | | | | | |  __/ | | | || (_| | |_| | (_) | | | |
\_| |_/\__,_|\__, |_| |_| |_|\___|_| |_|\__\__,_|\__|_|\___/|_| |_|
              __/ |                                                
             |___/                                                 
  ___  ______ _____       _____                           
 / _ \ | ___ \_   _|  _  |_   _|                          
/ /_\ \| |_/ / | |   (_)   | | _ __ ___   __ _  __ _  ___ 
|  _  ||  __/  | |         | || '_ ` _ \ / _` |/ _` |/ _ \
| | | || |    _| |_   _   _| || | | | | | (_| | (_| |  __/
\_| |_/\_|    \___/  (_)  \___/_| |_| |_|\__,_|\__, |\___|

Following this tutorial:

https://nbviewer.jupyter.org/github/aleju/imgaug-doc/blob/master/notebooks/A01%20-%20Load%20and%20Augment%20an%20Image.ipynb
'''
import os

try:
    import helpers.imgaug.augmenters as iaa
    import helpers.imgaug as ia
except:
    os.system('pip3 install git+https://github.com/aleju/imgaug.git')
    import imaug.augmenters as iaa
    
import imageio
import matplotlib.pyplot as plt

def augment_imaug(imagefile):
    image = imageio.imread(imagefile)

    seq = iaa.Sequential([
        iaa.Affine(rotate=(-25, 25)),
        iaa.AdditiveGaussianNoise(scale=(30, 90)),
        iaa.Crop(percent=(0, 0.4))
    ], random_order=True)

    images_aug = [seq.augment_image(image) for _ in range(1)]

    # print("Augmented:")
    # ia.imshow(ia.draw_grid(images_aug, cols=4, rows=2))
    files=[imagefile]
    for i in range(len(images_aug)):
        filename='augmented_%s'%(str(i))+imagefile
        plt.imsave(filename, images_aug[i])
        files.append(filename)
    return files
