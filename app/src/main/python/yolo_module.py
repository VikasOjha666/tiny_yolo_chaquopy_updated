from keras.models import Sequential, Model
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense
from keras.layers.advanced_activations import LeakyReLU
import numpy as np
import scipy.io
import random
import os
import xml.etree.ElementTree as ET
import tensorflow as tf
import copy
import cv2
import sys
import time
import base64
import io
import os
from PIL import Image
from os.path import dirname,join


GRID_H, GRID_W = 13 , 13
BOX = 5
ORIG_CLASS = 20

WT_PATH = 'tiny-yolo-voc.weights'

THRESHOLD = 0.2
ANCHORS = '1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52'
ANCHORS = [float(ANCHORS.strip()) for ANCHORS in ANCHORS.split(',')]



# UTILS FUNCTIONS

class BoundBox:
    def __init__(self, class_num):
        self.x, self.y, self.w, self.h, self.c = 0., 0., 0., 0., 0.
        self.probs = np.zeros((class_num,))

    def iou(self, box):
        intersection = self.intersect(box)
        union = self.w*self.h + box.w*box.h - intersection
        return intersection/union

    def intersect(self, box):
        width  = self.__overlap([self.x-self.w/2, self.x+self.w/2], [box.x-box.w/2, box.x+box.w/2])
        height = self.__overlap([self.y-self.h/2, self.y+self.h/2], [box.y-box.h/2, box.y+box.h/2])
        return width * height

    def __overlap(self, interval_a, interval_b):
        x1, x2 = interval_a
        x3, x4 = interval_b
        if x3 < x1:
            if x4 < x1:
                return 0
            else:
                return min(x2,x4) - x1
        else:
            if x2 < x3:
                return 0
            else:
                return min(x2,x4) - x3


def interpret_netout(image, netout):
    boxes = []

    # interpret the output by the network
    for row in range(GRID_H):
        for col in range(GRID_W):
            for b in range(BOX):
                box = BoundBox(CLASS)

                # first 5 weights for x, y, w, h and confidence
                box.x, box.y, box.w, box.h, box.c = netout[row,col,b,:5]

                box.x = (col + sigmoid(box.x)) / GRID_W
                box.y = (row + sigmoid(box.y)) / GRID_H
                box.w = ANCHORS[2 * b + 0] * np.exp(box.w) / GRID_W
                box.h = ANCHORS[2 * b + 1] * np.exp(box.h) / GRID_H
                box.c = sigmoid(box.c)

                # rest of weights for class likelihoods
                classes = netout[row,col,b,5:]
                box.probs = softmax(classes) * box.c
                box.probs *= box.probs > THRESHOLD

                boxes.append(box)

    # suppress non-maximal boxes
    for c in range(CLASS):
        sorted_indices = list(reversed(np.argsort([box.probs[c] for box in boxes])))

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]

            if boxes[index_i].probs[c] == 0:
                continue
            else:
                for j in range(i+1, len(sorted_indices)):
                    index_j = sorted_indices[j]

                    if boxes[index_i].iou(boxes[index_j]) >= 0.4:
                        boxes[index_j].probs[c] = 0

    

    # draw the boxes using a threshold
    for box in boxes:
        max_indx = np.argmax(box.probs)
        max_prob = box.probs[max_indx]
        
        # if(max_prob>0.01):
        # 	print("Detected box with probability : {}".format(max_prob))


        if max_prob > THRESHOLD:
            xmin  = int((box.x - box.w/2) * image.shape[1])
            xmax  = int((box.x + box.w/2) * image.shape[1])
            ymin  = int((box.y - box.h/2) * image.shape[0])
            ymax  = int((box.y + box.h/2) * image.shape[0])

            cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (200,0,0), 2)
            cv2.putText(image, labels[max_indx], (xmin, ymin - 12), 0, 1e-3 * image.shape[0], (0,255,0), 1)

    return image



def sigmoid(x):
    return 1. / (1.  + np.exp(-x))

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)




class WeightReader:
    def __init__(self, weight_file):
        self.offset = 4
        self.path_w=join(dirname(__file__),'tiny-yolo-voc.weights')
        self.all_weights = np.fromfile(self.path_w, dtype='float32')
        
    def read_bytes(self, size):
        self.offset = self.offset + size
        return self.all_weights[self.offset-size:self.offset]
    
    def reset(self):
        self.offset = 4

weight_reader = WeightReader(WT_PATH)


# Load network

model = Sequential()

# Layer 1
model.add(Conv2D(16, (3,3), strides=(1,1), padding='same', use_bias=False, input_shape=(416,416,3)))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 2 - 5
for i in range(0,4):
    model.add(Conv2D(32*(2**i), (3,3), strides=(1,1), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 6
model.add(Conv2D(512, (3,3), strides=(1,1), padding='same', use_bias=False))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='same'))

# Layer 7 - 8
for _ in range(0,2):
    model.add(Conv2D(1024, (3,3), strides=(1,1), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))

# Layer 9
model.add(Conv2D(BOX * (4 + 1 + ORIG_CLASS), (1, 1), strides=(1, 1), kernel_initializer='he_normal'))
model.add(Activation('linear'))
model.add(Reshape((GRID_H, GRID_W, BOX, 4 + 1 + ORIG_CLASS)))



# Load pre-trained weights

# model.load_weights(orig_weight_path)


weight_reader.reset()
nb_conv = 9

for i in range(1, nb_conv+1):
    conv_layer = model.get_layer('conv2d_' + str(i))
    
    if i < nb_conv:
        norm_layer = model.get_layer('batch_normalization_' + str(i))
        
        size = np.prod(norm_layer.get_weights()[0].shape)

        beta  = weight_reader.read_bytes(size)
        gamma = weight_reader.read_bytes(size)
        mean  = weight_reader.read_bytes(size)
        var   = weight_reader.read_bytes(size)

        weights = norm_layer.set_weights([gamma, beta, mean, var])       
        
    if len(conv_layer.get_weights()) > 1:
        bias   = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[1].shape))
        kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
        kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
        kernel = kernel.transpose([2,3,1,0])
        conv_layer.set_weights([kernel, bias])
    else:
        kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
        kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
        kernel = kernel.transpose([2,3,1,0])
        conv_layer.set_weights([kernel])



# get correct labels
labels = ['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']

print (labels);

# CLASS = 184
CLASS = 20 # 23




def detectObjects(image):
    input_image = cv2.resize(image, (416, 416))
    input_image = input_image / 255.
    input_image = input_image[:,:,::-1]
    input_image = np.expand_dims(input_image, 0)

    # netout = new_model.predict(input_image)
    netout = model.predict(input_image)
    image = interpret_netout(image, netout[0])

    return image

def detect_and_draw(strn):
    decoded_data=base64.b64decode(strn)
    np_data=np.fromstring(decoded_data,np.uint8)
    img=cv2.imdecode(np_data,cv2.IMREAD_UNCHANGED)
   
    img=detectObjects(img)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    pil_img=Image.fromarray(img)
    buff=io.BytesIO()
    pil_img.save(buff,format="PNG")
    img_str=base64.b64encode(buff.getvalue())
    return ""+str(img_str,'utf-8')