import numpy as np
import cv2

#put these files on the desktop

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

img = cv2.imread('face.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
increment=0

for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    newimg=img[y:y+h,x:x+w]
    #save only the face
    cv2.imwrite('only_face' + str(increment) + '.jpg', newimg)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    increment=increment+1 

cv2.imshow('img',img)
cv2.imwrite('faces.png',img)

#detect smile throughout all images (and get timestamp annotated)
#detect sadness throughout all images (and get timestamp annotated)
