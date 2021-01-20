import numpy as np
import tensorflow as tf
import cv2


#importing trained model in the tensorflow graph and defining predict function
class Model:
  def __init__(self,model_path):
    self.declaration_graph = tf.Graph()
    with self.declaration_graph.as_default():
      od_graph_def=tf.compat.v1.GraphDef()
      with tf.io.gfile.GFile(model_path,'rb') as f:
        serialized_graph=f.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def,name='')
    
    self.sess=tf.compat.v1.Session(graph=self.declaration_graph)

  def predict(self,img):

    img_exp=np.expand_dims(img,axis=0)
    (boxes,scores,classes)=self.sess.run([self.declaration_graph.get_tensor_by_name('detection_boxes:0'),self.declaration_graph.get_tensor_by_name('detection_scores:0'),self.declaration_graph.get_tensor_by_name('detection_classes:0')],feed_dict={self.declaration_graph.get_tensor_by_name('image_tensor:0'):img_exp})
    return (boxes,scores,classes)
