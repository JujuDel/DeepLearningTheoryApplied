# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 14:33:31 2019

@author: DEJ6SI
"""

import cv2
import numpy as np
import os
import sys
from PIL import Image
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import tensorflow as tf
from keras import backend as K
from keras.models import load_model
from yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes, draw_boxes_2, scale_boxes
from yad2k.models.keras_yolo import yolo_head, yolo_boxes_to_corners

def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = .6):
    
    box_scores = box_confidence * box_class_probs
    
    box_classes = K.argmax(box_scores, axis=-1)
    box_class_scores = K.max(box_scores, axis=-1)
    
    filtering_mask = (box_class_scores >= threshold)
    
    scores = tf.boolean_mask(box_class_scores, filtering_mask)
    boxes = tf.boolean_mask(boxes, filtering_mask)
    classes = tf.boolean_mask(box_classes, filtering_mask)
    
    return scores, boxes, classes

def iou(box1, box2):
    
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter_area = max((yi2 - yi1), 0) * max((xi2 - xi1), 0)
    
    box1_area = (box1[3] - box1[1]) * (box1[2] - box1[0])
    box2_area = (box2[3] - box2[1]) * (box2[2] - box2[0])
    union_area = box1_area + box2_area - inter_area
    
    res = inter_area / union_area
    
    return res

def yolo_non_max_suppression(scores, boxes, classes, max_boxes = 10, iou_threshold = 0.5):
    
    max_boxes_tensor = K.variable(max_boxes, dtype='int32')     # tensor to be used in tf.image.non_max_suppression()
    K.get_session().run(tf.variables_initializer([max_boxes_tensor])) # initialize variable max_boxes_tensor
    
    nms_indices = tf.image.non_max_suppression(boxes, scores, max_boxes, iou_threshold=0.5)
    
    scores = K.gather(scores, nms_indices)
    boxes = K.gather(boxes, nms_indices)
    classes = K.gather(classes, nms_indices)
    
    return scores, boxes, classes

def yolo_eval(yolo_outputs, image_shape = (720., 1280.), max_boxes=10, score_threshold=.6, iou_threshold=.5):
    
    box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs

    boxes = yolo_boxes_to_corners(box_xy, box_wh)

    scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = score_threshold)
    
    boxes = scale_boxes(boxes, image_shape)

    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes = max_boxes, iou_threshold = iou_threshold)
    
    return scores, boxes, classes

def prepare_yolo_eval(width, height, yolo_outputs):
    
    image_shape = (float(height), float(width))
    
    scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)
    
    return scores, boxes, classes

def predictFromPath(sess, image_file, yolo_model, yolo_output, class_names):
    
    image, image_data = preprocess_image("images/" + image_file, model_image_size = (608, 608))
    
    scores, boxes, classes = prepare_yolo_eval(image.width, image.height, yolo_output)
    
    out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes], feed_dict={yolo_model.input: image_data, K.learning_phase(): 0})

    print('Found {} boxes for {}'.format(len(out_boxes), image_file))

    colors = generate_colors(class_names)
    
    draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
    
    image.save(os.path.join("out", image_file), quality=90)
    
    output_image = scipy.misc.imread(os.path.join("out", image_file))
    imshow(output_image)
    
    return out_scores, out_boxes, out_classes

def predictFromImg(sess, image, scores, boxes, classes, yolo_model, class_names):
    
    # preprocess_image
    model_image_size = (608, 608)
    resized_image = image.resize(tuple(reversed(model_image_size)), Image.BICUBIC)
    image_data = np.array(resized_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
    
    out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes], feed_dict={yolo_model.input: image_data, K.learning_phase(): 0})

    colors = generate_colors(class_names)
    
    draw_boxes_2(image, out_scores, out_boxes, out_classes, class_names, colors)
    
    return out_scores, out_boxes, out_classes

def main():
    
    sess = K.get_session()

    class_names = read_classes("model_data/coco_classes.txt")
    anchors = read_anchors("model_data/yolo_anchors.txt")
    image_shape = (480., 640.)
    
    yolo_model = load_model("model_data/yolo.h5")
    
    yolo_model.summary()
    
    yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
    
    scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)
    
    video = cv2.VideoCapture(0)
    width = video.get(3)   # float
    height = video.get(4)  # float
    
    print(width, " x ", height)
    #prepare_yolo_eval(width, height, yolo_outputs)
    
    try:
        while (True):
            # Read the next frame
            ret, cv2_im = video.read() 
            
            k = cv2.waitKey(1)
            if not ret or k == 27:  # press ESC to exit
                video.release()
                cv2.destroyAllWindows()
                print("Released Video Resource")
                break
            
            cv2_im = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
            pil_im = Image.fromarray(cv2_im)
            
            predictFromImg(sess, pil_im, scores, boxes, classes, yolo_model, class_names)
            
            cv2_im = np.array(pil_im)
            cv2_im = cv2_im[:, :, ::-1].copy()
            cv2.imshow("Webcam", cv2_im)
           
    except:
        video.release()
        cv2.destroyAllWindows()
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_obj, exc_tb.tb_lineno)
        print("Released Video Resource")

if __name__ == '__main__':
    main()