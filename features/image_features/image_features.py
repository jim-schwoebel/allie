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

______         _                          ___  ______ _____     
|  ___|       | |                        / _ \ | ___ \_   _|  _ 
| |_ ___  __ _| |_ _   _ _ __ ___  ___  / /_\ \| |_/ / | |   (_)
|  _/ _ \/ _` | __| | | | '__/ _ \/ __| |  _  ||  __/  | |      
| ||  __/ (_| | |_| |_| | | |  __/\__ \ | | | || |    _| |_   _ 
\_| \___|\__,_|\__|\__,_|_|  \___||___/ \_| |_/\_|    \___/  (_)
                                                                
                                                                
 _____                           
|_   _|                          
  | | _ __ ___   __ _  __ _  ___ 
  | || '_ ` _ \ / _` |/ _` |/ _ \
 _| || | | | | | (_| | (_| |  __/
 \___/_| |_| |_|\__,_|\__, |\___|
                       __/ |     
                      |___/      

Featurizes folders of image files if default_text_features = ['image_features']

Note this uses OpenCV and the SIFT feature detector. SIFT was used here 
as as scale-invariant feature selector, but note that this algorithm is patented,
which limits commercical use.
'''
from sklearn import preprocessing, svm, metrics
from textblob import TextBlob
from operator import itemgetter
import getpass, pickle, datetime, time
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import cv2, os 

def convert(file):
    if file[-5:]=='.jpeg':       
        im = Image.open(file)
        rgb_im = im.convert('RGB')
        filename=file[0:-5]+'.png'
        rgb_im.save(filename)
        os.remove(file)
    elif file[-4:]=='.jpg':
        im = Image.open(file)
        rgb_im = im.convert('RGB')
        filename=file[0:-4]+'.png'
        rgb_im.save(filename)             
        os.remove(file)

    return filename

def haar_featurize(cur_dir, haar_dir, img):
              
    os.chdir(haar_dir)
    # load image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # assumes all files of haarcascades are in current directory 
    one = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
    one = one.detectMultiScale(gray, 1.3, 5)
    one = len(one)
              
    two = cv2.CascadeClassifier('haarcascade_eye.xml')
    two = two.detectMultiScale(gray, 1.3, 5)
    two = len(two)
              
    three = cv2.CascadeClassifier('haarcascade_frontalcatface_extended.xml')
    three = three.detectMultiScale(gray, 1.3, 5)
    three = len(three)
              
    four = cv2.CascadeClassifier('haarcascade_frontalcatface.xml')
    four = four.detectMultiScale(gray, 1.3, 5)
    four = len(four)
              
    five = cv2.CascadeClassifier('haarcascade_frontalface_alt_tree.xml')
    five = five.detectMultiScale(gray, 1.3, 5)
    five = len(five)
              
    six = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    six = six.detectMultiScale(gray, 1.3, 5)
    six = len(six)
              
    seven = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    seven = seven.detectMultiScale(gray, 1.3, 5)
    seven = len(seven)
              
    eight = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eight = eight.detectMultiScale(gray, 1.3, 5)
    eight = len(eight)
              
    nine = cv2.CascadeClassifier('haarcascade_fullbody.xml')
    nine = nine.detectMultiScale(gray, 1.3, 5)
    nine = len(nine)
              
    ten = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')
    ten = ten.detectMultiScale(gray, 1.3, 5)
    ten = len(ten)
              
    eleven = cv2.CascadeClassifier('haarcascade_licence_plate_rus_16stages.xml')
    eleven = eleven.detectMultiScale(gray, 1.3, 5)
    eleven = len(eleven)
              
    twelve = cv2.CascadeClassifier('haarcascade_lowerbody.xml')
    twelve = twelve.detectMultiScale(gray, 1.3, 5)
    twelve = len(twelve)
              
    thirteen = cv2.CascadeClassifier('haarcascade_profileface.xml')
    thirteen = thirteen.detectMultiScale(gray, 1.3, 5)
    thirteen = len(thirteen)
              
    fourteen = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')
    fourteen = fourteen.detectMultiScale(gray, 1.3, 5)
    fourteen = len(fourteen)
              
    fifteen = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')
    fifteen = fifteen.detectMultiScale(gray, 1.3, 5)
    fifteen = len(fifteen)
              
    sixteen = cv2.CascadeClassifier('haarcascade_smile.xml')
    sixteen = sixteen.detectMultiScale(gray, 1.3, 5)
    sixteen = len(sixteen)
              
    seventeen = cv2.CascadeClassifier('haarcascade_upperbody.xml')
    seventeen = seventeen.detectMultiScale(gray, 1.3, 5)
    seventeen = len(seventeen)

    features=np.array([one,two,three,four,
                      five,six,seven,eight,
                      nine,ten,eleven,twelve,
                      thirteen,fourteen,fifteen,sixteen,
                      seventeen])
              
    labels=['haarcascade_eye_tree_eyeglasses','haarcascade_eye','haarcascade_frontalcatface_extended','haarcascade_frontalcatface',
            'haarcascade_frontalface_alt_tree','haarcascade_frontalface_alt','haarcascade_frontalface_alt2','haarcascade_frontalface_default',
            'haarcascade_fullbody','haarcascade_lefteye_2splits','haarcascade_licence_plate_rus_16stages','haarcascade_lowerbody',
            'haarcascade_profileface','haarcascade_righteye_2splits','haarcascade_russian_plate_number','haarcascade_smile',
            'haarcascade_upperbody']
              
    os.chdir(cur_dir)
              
    return features, labels
                  
def image_featurize(cur_dir,haar_dir,file):

    # initialize label array 
    labels=list()
    # only featurize files that are .jpeg, .jpg, or .png (convert all to ping
    if file[-5:]=='.jpeg':
        filename=convert(file)
    elif file[-4:]=='.jpg':
        filename=convert(file)
    elif file[-4:]=='.png':
        filename=file 
    else:
        filename=file
              
    #only featurize .png files after conversion 
    if filename[-4:]=='.png':
        # READ IMAGE
        ########################################################
        img = cv2.imread(filename,1)

        # CALCULATE BASIC FEATURES (rows, columns, pixels)
        ########################################################
        #rows, columns, pixel number
        rows=img.shape[1]
        columns=img.shape[2]
        pixels=img.size

        basic_features=np.array([rows,columns,pixels])
        labels=labels+['rows', 'columns', 'pixels']
        # HISTOGRAM FEATURES (avg, stdev, min, max)
        ########################################################
        #blue
        blue_hist=cv2.calcHist([img],[0],None,[256],[0,256])
        blue_mean=np.mean(blue_hist)
        blue_std=np.std(blue_hist)
        blue_min=np.amin(blue_hist)
        blue_max=np.amax(blue_hist)
        #green
        green_hist=cv2.calcHist([img],[1],None,[256],[0,256])
        green_mean=np.mean(green_hist)
        green_std=np.std(green_hist)
        green_min=np.amin(green_hist)
        green_max=np.amax(green_hist)
        #red
        red_hist=cv2.calcHist([img],[2],None,[256],[0,256])
        red_mean=np.mean(red_hist)
        red_std=np.std(red_hist)
        red_min=np.amin(red_hist)
        red_max=np.amax(red_hist)

        hist_features=[blue_mean,blue_std,blue_min,blue_max,
                       green_mean,green_std,green_min,green_max,
                       red_mean,red_std,red_min,red_max]
        hist_labels=['blue_mean','blue_std','blue_min','blue_max',
                     'green_mean','green_std','green_min','green_max',
                     'red_mean','red_std','red_min','red_max']

        hist_features=np.array(hist_features)

        features=np.append(basic_features,hist_features)
        labels=labels+hist_labels 

        # CALCULATE HAAR FEATURES
        ########################################################
        haar_features, haar_labels=haar_featurize(cur_dir,haar_dir,img)
        
        features=np.append(features,haar_features)
        labels=labels+haar_labels
        
        # EDGE FEATURES
        ########################################################
        # SIFT algorithm (scale invariant) - 128 features
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        (kps, des) = sift.detectAndCompute(gray, None)
        edges=des
        edge_features=np.zeros(len(edges[0]))
              
        for i in range(len(edges)):
            edge_features=edge_features+edges[i]

        edge_features=edge_features/(len(edges))
        edge_features=np.array(edge_features)
        edge_labels=list()
        for i in range(len(edge_features)):
            edge_labels.append('edge_feature_%s'%(str(i+1)))
        features=np.append(features,edge_features)
        labels=labels+edge_labels 
              
    else:
        os.remove(file)

    return features, labels
