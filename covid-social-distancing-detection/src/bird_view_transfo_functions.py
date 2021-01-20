import numpy as np
import cv2

#perspective transformation
def compute_perspective_transform(image,corner_points,width,height):
  corner_points_array = np.float32(corner_points)
  img_points = np.float32([[0,0],[width,0],[0,height],[width,height]])
  matrix = cv2.getPerspectiveTransform(corner_points_array,img_points)
  img_transformed = cv2.warpPerspective(image,matrix,(width,height))
  return matrix,img_transformed


#perspective transformation of points detected
def compute_point_perspective_transformation(matrix,list_points):
  list_points_todetect = np.float32(list_points).reshape(-1,1,2)
  transformed_points = cv2.perspectiveTransform(list_points_todetect,matrix)
  list_points_transformed = list()
  for i in range(0,transformed_points.shape[0]):
    list_points_transformed.append([transformed_points[i,0,0],transformed_points[i,0,1]])
  return list_points_transformed
