from bird_view_transfo_functions import compute_perspective_transform,compute_point_perspective_transformation
from tf_model_object_detection import Model 
import numpy as np
import itertools
import imutils
import math
import yaml
import cv2
import os
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def get_human_box_detection(boxes,scores,classes,height,width):
	array_boxes = list()
	for i in range(boxes.shape[1]):
		if int(classes[i]) == 1 and scores[i]> 0.75 :
			box = [boxes[0,i,0],boxes[0,i,1],boxes[0,i,2],boxes[0,i,3]] * np.array([height,width,height,width])
			array_boxes.append((int(box[0]),int(box[1]),int(box[2]),int(box[3])))
	return array_boxes

def get_centroids_and_groundpoints(arrar_boxes_detected):
	array_centroids,array_groundpoints = [],[]	
	for idx,box in enumerate(array_boxes_detected):
		center_x = int((box[1]+box[3])/2)
		center_y = int((box[0]+box[2])/2)
		center_y_groundpoint = center_y + int((box[2]-box[0])/2)
		array_centroids.append((center_x,center_y))
		array_groundpoints.append((center_x,center_y_groundpoint))
	return array_centroids,array_groundpoints

##############################
with open('../conf/config_birdview.yml','r') as ymlfile:
	cfg = yaml.load(ymlfile)
width_og , height_og = 0,0
corner_points = []
for section in cfg:
	corner_points.append(cfg['image_parameters']['p1'])
	corner_points.append(cfg["image_parameters"]["p2"])
	corner_points.append(cfg["image_parameters"]["p3"])
	corner_points.append(cfg["image_parameters"]["p4"])
	width_og = int(cfg["image_parameters"]["width_og"])
	height_og = int(cfg["image_parameters"]["height_og"])
	img_path = cfg["image_parameters"]["img_path"]
	size_frame = cfg["image_parameters"]["size_frame"]	


#import frozen_inference_graph.pb
model_path="../models/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb" 
model = Model(model_path)
video_path="../video/PETS2009.avi"
distance_minimum = "110"



matrix,imgOutput = compute_perspective_transform(cv2.imread(img_path),corner_points,width_og,height_og)
height,width,_ = imgOutput.shape
blank_image = np.zeros((height,width,3), np.uint8)
height = blank_image.shape[0]
width = blank_image.shape[1] 
dim = (width, height)           #(600,800)




vs = cv2.VideoCapture(video_path)
output_video_1,output_video_2 = None,None

while True:
	img = cv2.imread("../img/chemin_1.png")
	bird_view_img = cv2.resize(img,dim,interpolation = cv2.INTER_AREA)	

	frame_exists,frame = vs.read()
	if not frame_exists:
		break
	else:
		frame = imutils.resize(frame,width=int(size_frame))
		(boxes,scores,classes) = model.predict(frame)
		array_boxes_detected = get_human_box_detection(boxes,scores[0].tolist(),classes[0].tolist(),frame.shape[0],frame.shape[1])
		array_centroids , array_groundpoints = get_centroids_and_groundpoints(array_boxes_detected)
		transformed_downoids = compute_point_perspective_transformation(matrix,array_groundpoints)
		
		for point in transformed_downoids:
			(x,y)= point
			cv2.circle(bird_view_img,(x,y),60,(0,255,0),2)
			cv2.circle(bird_view_img,(x,y),3,(0,255,0),-1)

		if len(transformed_downoids) >=2:
			for index,downoid in enumerate(transformed_downoids):
				if not (downoid[0] > width or downoid[0] < 0 or downoid[1] > height+200 or downoid[1] < 0 ):
						cv2.rectangle(frame,(array_boxes_detected[index][1],array_boxes_detected[index][0]),(array_boxes_detected[index]			[3],array_boxes_detected[index][2]),(0,255,0),2)			


			list_indexes = list(itertools.combinations(range(len(transformed_downoids)),2))
			for i,pair in enumerate(itertools.combinations(transformed_downoids,2)):
				if np.linalg.norm(np.array(pair[0])-np.array(pair[1])) < int(distance_minimum):
					if not (pair[0][0] > width or pair[0][0] < 0 or pair[0][1] > height+200  or pair[0][1] < 0 or pair[1][0] > width or pair[1][0] < 0 or pair[1][1] > height+200  or pair[1][1] < 0):
						#change color of circle in top view
						cv2.circle(bird_view_img,(pair[0][0],pair[0][1]),60,(0,0,255),2)
						cv2.circle(bird_view_img,(pair[0][0],pair[0][1]),3,(0,0,255),-1)
						cv2.circle(bird_view_img,(pair[1][0],pair[1][1]),60,(0,0,255),2)
						cv2.circle(bird_view_img,(pair[1][0],pair[1][1]),3,(0,0,255),-1)
						
	
						index_pt1 = list_indexes[i][0]
						index_pt2 = list_indexes[i][1]
						cv2.rectangle(frame,(array_boxes_detected[index_pt1][1],array_boxes_detected[index_pt1][0]),(array_boxes_detected[index_pt1][3],array_boxes_detected[index_pt1][2]),(0,0,255),2)
						cv2.rectangle(frame,(array_boxes_detected[index_pt2][1],array_boxes_detected[index_pt2][0]),(array_boxes_detected[index_pt2][3],array_boxes_detected[index_pt2][2]),(0,0,255),2)
	
	cv2.imshow('Bird view',bird_view_img)
	img1 = cv2.resize(bird_view_img,(240,320))
	cv2.putText(img1,'Bird_View',(10,10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),1)
								
	key=cv2.waitKey(20) & 0xFF
	#print(img1.shape)	
	frame[600-img1.shape[0]:600,0:img1.shape[1],:]=img1
	cv2.imshow('camera',frame)	
	if output_video_1 is None and output_video_2 is None:
		#print('bird view',bird_view_img.shape)
		#print('frame',frame.shape)
		fourcc1 = cv2.VideoWriter_fourcc(*'MPEG')
		output_video_1 = cv2.VideoWriter('../output/video.avi',fourcc1,25,(frame.shape[1],frame.shape[0]),True)	
		fourcc2 = cv2.VideoWriter_fourcc(*'MPEG')
		output_video_2 = cv2.VideoWriter('../output/bird_view.avi',fourcc2,25,(bird_view_img.shape[1],bird_view_img.shape[0]),True)
	else:
		output_video_1.write(frame)
		output_video_2.write(bird_view_img)	


	if key == ord('q'):
		break





















