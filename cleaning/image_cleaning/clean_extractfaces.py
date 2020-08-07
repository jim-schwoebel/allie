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

/  __ \ |                (_)              / _ \ | ___ \_   _|  _ 
| /  \/ | ___  __ _ _ __  _ _ __   __ _  / /_\ \| |_/ / | |   (_)
| |   | |/ _ \/ _` | '_ \| | '_ \ / _` | |  _  ||  __/  | |      
| \__/\ |  __/ (_| | | | | | | | | (_| | | | | || |    _| |_   _ 
 \____/_|\___|\__,_|_| |_|_|_| |_|\__, | \_| |_/\_|    \___/  (_)
                                   __/ |                         
                                  |___/                          
 _____                           
|_   _|                          
  | | _ __ ___   __ _  __ _  ___ 
  | || '_ ` _ \ / _` |/ _` |/ _ \
 _| || | | | | | (_| | (_| |  __/
 \___/_| |_| |_|\__,_|\__, |\___|
                       __/ |     
                      |___/      
                      
This script takes in a folder of images and extracts out the faces for these images 
if they are in there and deletes the original image. This is useful if you are looking 
to do a lot of facial machine learning work.

This is enabled if default_image_cleaners=['clean_extractfaces']
'''
# you only use these modules if you register, so put them here
import cv2, os, time, shutil, math
import skvideo.io, skvideo.motion, skvideo.measure
from moviepy.editor import VideoFileClip
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

def euclidean_distance(a, b):
    x1 = a[0]; y1 = a[1]
    x2 = b[0]; y2 = b[1]
    
    return math.sqrt(((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1)))

def detectFace(img,face_detector,eye_detector,nose_detector):
    faces = face_detector.detectMultiScale(img, 1.3, 5)
    #print("found faces: ", len(faces))

    if len(faces) > 0:
        face = faces[0]
        face_x, face_y, face_w, face_h = face
        img = img[int(face_y):int(face_y+face_h), int(face_x):int(face_x+face_w)]
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        return img, img_gray
    else:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img, img_gray
        #raise ValueError("No face found in the passed image ")

def alignFace(img_path, face_detector, eye_detector, nose_detector):
    img = cv2.imread(img_path)
    plt.imshow(img[:, :, ::-1])
    plt.show()

    img_raw = img.copy()

    img, gray_img = detectFace(img,face_detector,eye_detector,nose_detector)
    
    eyes = eye_detector.detectMultiScale(gray_img)
    
    #print("found eyes: ",len(eyes))
    
    if len(eyes) >= 2:
        #find the largest 2 eye
        
        base_eyes = eyes[:, 2]
        #print(base_eyes)
        
        items = []
        for i in range(0, len(base_eyes)):
            item = (base_eyes[i], i)
            items.append(item)
        
        df = pd.DataFrame(items, columns = ["length", "idx"]).sort_values(by=['length'], ascending=False)
        
        eyes = eyes[df.idx.values[0:2]]
        
        #--------------------
        #decide left and right eye
        
        eye_1 = eyes[0]; eye_2 = eyes[1]
        
        if eye_1[0] < eye_2[0]:
            left_eye = eye_1
            right_eye = eye_2
        else:
            left_eye = eye_2
            right_eye = eye_1
        
        #--------------------
        #center of eyes
        
        left_eye_center = (int(left_eye[0] + (left_eye[2] / 2)), int(left_eye[1] + (left_eye[3] / 2)))
        left_eye_x = left_eye_center[0]; left_eye_y = left_eye_center[1]
        
        right_eye_center = (int(right_eye[0] + (right_eye[2]/2)), int(right_eye[1] + (right_eye[3]/2)))
        right_eye_x = right_eye_center[0]; right_eye_y = right_eye_center[1]
        
        #center_of_eyes = (int((left_eye_x+right_eye_x)/2), int((left_eye_y+right_eye_y)/2))
        
        cv2.circle(img, left_eye_center, 2, (255, 0, 0) , 2)
        cv2.circle(img, right_eye_center, 2, (255, 0, 0) , 2)
        #cv2.circle(img, center_of_eyes, 2, (255, 0, 0) , 2)
        
        #----------------------
        #find rotation direction
        
        if left_eye_y > right_eye_y:
            point_3rd = (right_eye_x, left_eye_y)
            direction = -1 #rotate same direction to clock
            print("rotate to clock direction")
        else:
            point_3rd = (left_eye_x, right_eye_y)
            direction = 1 #rotate inverse direction of clock
            print("rotate to inverse clock direction")
        
        #----------------------
        
        cv2.circle(img, point_3rd, 2, (255, 0, 0) , 2)
        
        cv2.line(img,right_eye_center, left_eye_center,(67,67,67),1)
        cv2.line(img,left_eye_center, point_3rd,(67,67,67),1)
        cv2.line(img,right_eye_center, point_3rd,(67,67,67),1)
        
        a = euclidean_distance(left_eye_center, point_3rd)
        b = euclidean_distance(right_eye_center, point_3rd)
        c = euclidean_distance(right_eye_center, left_eye_center)
        
        #print("left eye: ", left_eye_center)
        #print("right eye: ", right_eye_center)
        #print("additional point: ", point_3rd)
        #print("triangle lengths: ",a, b, c)
        
        cos_a = (b*b + c*c - a*a)/(2*b*c)
        #print("cos(a) = ", cos_a)
        angle = np.arccos(cos_a)
        #print("angle: ", angle," in radian")
        
        angle = (angle * 180) / math.pi
        print("angle: ", angle," in degree")
        
        if direction == -1:
            angle = 90 - angle
        
        print("angle: ", angle," in degree")
        
        #--------------------
        #rotate image
        
        new_img = Image.fromarray(img_raw)
        new_img = np.array(new_img.rotate(direction * angle))
    else:
        #find the largest 2 ey
        new_img = img_raw

    return new_img

def capture_video(filename, timesplit):
    video=cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    out = cv2.VideoWriter(filename,cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

    a=0
    start=time.time()

    while True:
        a=a+1
        check, frame=video.read()
        #print(check)
        #print(frame)
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        out.write(frame)
        #cv2.imshow("frame",gray)
        end=time.time()
        if end-start>timesplit:
            break 
        #print(end-start)

    print(a)
    video.release()
    out.release() 
    cv2.destroyAllWindows()

    return filename 

def clean_extractfaces(filename,basedir):

    # paths
    opencv_home = cv2.__file__
    folders = opencv_home.split(os.path.sep)[0:-1]

    path = folders[0]
    for folder in folders[1:]:
        path = path + "/" + folder

    # other stuff
    face_detector_path = path+"/data/haarcascade_frontalface_default.xml"
    eye_detector_path = path+"/data/haarcascade_eye.xml"
    nose_detector_path = path+"/data/haarcascade_mcs_nose.xml"

    if os.path.isfile(face_detector_path) != True:
        raise ValueError("Confirm that opencv is installed on your environment! Expected path ",detector_path," violated.")

    face_detector = cv2.CascadeClassifier(face_detector_path)
    eye_detector = cv2.CascadeClassifier(eye_detector_path) 
    nose_detector = cv2.CascadeClassifier(nose_detector_path) 

    # load image file
    image_file = filename
    alignedFace = alignFace(image_file, face_detector, eye_detector, nose_detector)
    gray = cv2.cvtColor(alignedFace, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    increment=0
    facenums=0
    print(len(faces))

    filenames=list()

    if len(faces) == 0:
        pass
    else:
        for (x,y,w,h) in faces:
            img=alignedFace
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            newimg=img[y:y+h,x:x+w]
            new_image_file=image_file[0:-4] + '_face_' + str(increment) + '.png'
            newimg=cv2.resize(newimg, (100, 100), interpolation=cv2.INTER_LINEAR)
            norm_img = np.zeros((100, 100))
            norm_img = cv2.normalize(newimg, norm_img, 0, 255, cv2.NORM_MINMAX)
            cv2.imwrite(new_image_file, newimg)
            filenames.append(new_image_file)
            facenums=facenums+1

    os.remove(filename)
    return filenames