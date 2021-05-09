# -*- coding: utf-8 -*-
import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import cv2
import time
from mrcnn.config import Config
from datetime import datetime
# Root directory of the project
ROOT_DIR = os.getcwd()

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
#from samples.coco import coco
import cv2
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"
    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Number of classes (including background)
    NUM_CLASSES = 1 + 5  # background + 3 shapes
    # Use small imaes for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 1024

#import train_tongue
#class InferenceConfig(coco.CocoConfig):
class InferenceConfig(ShapesConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

class ProchainArrow(object):
    '''
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    '''

    def __init__(self, model_dir, model_name, GPU_ID='2'):
        self.num_class = 1 + 5
        self.config = InferenceConfig()
        self.labelClass = ['BG', 'figure', 'page_footer','table','text','page_header']
        self.model_path = model_dir+ model_name
        print('model_path',self.model_path)
        self.model = modellib.MaskRCNN(mode="inference", model_dir=self.model_path, config=self.config)
        self.model.load_weights(self.model_path, by_name=True)

    def detect_arrow(self, image):
        print('shape:', image.shape)
        cv2.imwrite('2.png', image)
        pred = self.model.detect([image], verbose=1)
        boxes = pred[0]['rois']
        classes = pred[0]['class_ids']
        k = len(classes)
        results = []
        for i in range(k):
            box = boxes[i]
            label = classes[i]
            result_dic = {'class':str(label), 'box':[str(box[3]),str(box[2]), str(box[1]),str(box[0])]}
            results.append(result_dic)
        print(results)
        # json.dumps(jsonList, ensure_ascii=False)
        return results


ins = ProchainArrow('./','mask_rcnn_lay_0499.h5')
if __name__ == '__main__':
    image_root = '/home/huluwa/data/img_lay/train_data_min2/pic/'
    save_root = '/home/huluwa/data/img_lay/save_img'
    image_names = os.listdir(image_root)
    print(image_names)
    for image_name in image_names:
        image_path = os.path.join(image_root, image_name)
        save_path = os.path.join(save_root, image_name)
        image = cv2.imread(image_path)
        rs = ins.detect_arrow(image)
        for r in rs:
            label = r['class']
            box = r['box']
            #if label == 1:
            print('box:',box)
            cv2.rectangle(image, (int(box[0]),int(box[1])), (int(box[2]),int(box[3])), (0,0,255), 2)
            #else:
            #    cv2.rectangle(image, (box[3],box[2]), (box[1],box[0]), (0,255,0), 2)
            cv2.imwrite(save_path, image)
            print(box)
            print(label)

