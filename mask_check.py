#pip install numpy
#pip install opencv_pyhon
import numpy as np
import cv2
face_cascade = cv2.CascadeClassifier('image/face.xml')
eye_cascade = cv2.CascadeClassifier('image/eye.xml')
value=int(input("Enter Image=1 , Video=2 : "))
if value==1:
  img = cv2.imread(f'image/Mask_Paramate.jpg')
  # img = cv2.imread(f'image/Inthira.jpg')
  img = cv2.resize(img,(640,480))
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  detecteye = 0;detectface = 0

  eyes = eye_cascade.detectMultiScale(gray,1.3,5)
  for (ex,ey,ew,eh) in eyes:
    #cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    #print ("detect eye");
    detecteye = 1

  faces = face_cascade.detectMultiScale(gray,1.3,5)
  for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w];roi_color = img[y:y+h, x:x+w];detectface = 1
    #print ("detect face");
     
  if detecteye == 1 and detectface == 0:
    print ("Waering Mask")
  elif detecteye == 1 and detectface == 1:
    print ("Not Waering Mask")

  cv2.imshow('image',img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
elif value == 2:
  # img = cv2.imread(f'image/Paramate.jpg')
  video_capture = cv2.VideoCapture(0) 
  ret,frame=video_capture.read()
  #small_frame=cv2.resize(frame,(0,0),fx=0.5,fy=0.5)
  #rgb_small_frame=small_frame[:,:,::-1]
  # img = cv2.resize(img,(640,480))
  gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
  detecteye = 0;detectface = 0

  eyes = eye_cascade.detectMultiScale(gray,1.3,5)
  for (ex,ey,ew,eh) in eyes:
    #cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    #print ("detect eye");
    detecteye = 1

  faces = face_cascade.detectMultiScale(gray,1.3,5)
  for (x,y,w,h) in faces:
    #img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    #roi_gray = gray[y:y+h, x:x+w];roi_color = img[y:y+h, x:x+w];
    #print ("detect face");
    detectface = 1
     
  if detecteye == 1 and detectface == 1:
    print ("Not Waering Mask")
  else:
    print ("Waering Mask")

  cv2.imshow('image',frame)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
