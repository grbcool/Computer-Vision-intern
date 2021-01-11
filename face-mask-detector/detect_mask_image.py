from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2
import os
import matplotlib.pyplot as plt



ap = argparse.ArgumentParser()
ap.add_argument('-i','--image',help='path to image',required=True)
ap.add_argument("-f", "--face", type=str,default="face_detector",help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

#loading the models
prototxtpath = os.path.sep.join([args['face'],'deploy.prototxt'])
weightspath = os.path.sep.join([args['face'],'res10_300x300_ssd_iter_140000.caffemodel'])

net = cv2.dnn.readNet(weightspath,prototxtpath)
model = load_model(args['model'])

#passing the image through model to get face detections
img = cv2.imread(args['image'])
ori = img.copy()
(h,w) = img.shape[:2]
blob = cv2.dnn.blobFromImage(img,1,(300,300),100)

net.setInput(blob)
detections = net.forward()

for i in range(0,detections.shape[2]):
  if detections[0,0,i,2]>=args['confidence']:
    box=detections[0,0,i,3:7]*np.array([w,h,w,h])
    (startx,starty,endx,endy) = box.astype('int')
    (startx,starty) = (max(0,startx),max(0,starty))
    (endx,endy) = (min(endx,w-1),min(endy,h-1))
    face = img[starty:endy,startx:endx]
    face = cv2.cvtColor(face,cv2.color_BGR2RGB)
    face = cv2.resize(face,(224,224))
    face = img_to_array(face)
    face = preprocess_input(face)
    face = np.expand_dims(axis=0)

    (mask,without_mask)= model.predict(face)[0]

    label = 'Mask' if mask >without_mask else 'No_mask'
    color = (0,255,0) if label =='Mask' else (0,0,255)
    label = "{} : {:.2f}% ".format(label,detections[0,0,i,2]*100)
    cv2.putText(img,label,(startx,starty-15),cv2.FONT_HERSHEY_SIMPLEX,0.45,color,2)
    cv2.rectangle(img,(startx,starty),(endx,endy),color,2)

plt.imshow(img)








