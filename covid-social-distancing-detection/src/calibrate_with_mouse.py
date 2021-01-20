import cv2
import numpy as np
import imutils
import yaml

#video_name = input('enter video name:')
#size_frame = input('enter the size of the frame in pixels: ')
video_name = 'PETS2009.avi'
size_frame = 800

cap = cv2.VideoCapture('../video/'+video_name)

while(True):
	frame_exists,frame = cap.read()
	frame = imutils.resize(frame,width=int(size_frame))
	cv2.imwrite('../img/static_frame_from_video.jpg',frame)
	break

cv2.namedWindow("Mousecallback")

img_path='../img/static_frame_from_video.jpg'
img=cv2.imread(img_path)

width,height,_ = img.shape

list_points = list()

def CallBackFunc(event,x,y,flags,param):
	if event == cv2.EVENT_LBUTTONDOWN:
		print('left button is clicked - position (',x,',',y,')')
		list_points.append([x,y])
	elif event == cv2.EVENT_RBUTTONDOWN:
		print('right button is clicked - position (',x,',',y,')')	
		list_points.append([x,y])


cv2.setMouseCallback('Mousecallback',CallBackFunc)

if __name__ == '__main__':
	while(True):
		cv2.imshow('Mousecallback',img)
		if len(list_points)==4:
			config_data = dict(
			image_parameters = dict(p2 =list_points[3],
			p1 =list_points[2],p4 =list_points[0],p3=list_points[1],width_og = width ,
			height_og = height,
			img_path = img_path,
			size_frame = size_frame
			)
	
			)
			with open('../conf/config_birdview.yml','w') as outfile:
				yaml.dump(config_data,outfile,default_flow_style = False)
			break
		if cv2.waitKey(20) ==27:
			break
	cv2.destroyAllWindows()





















