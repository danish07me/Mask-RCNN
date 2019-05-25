import os, re
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import pickle
#import tensorflow as tf
from skimage.measure import label,regionprops
from PIL import Image
from tensorflow.python.client import device_lib
#print (device_lib.list_local_devices())
# Import COCO config
ROOT_DIR = os.path.abspath("../")
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples", "coco"))  # To find local version
import pdb; pdb.set_trace()
import coco
#import coco
#import utils
#import model as modellib
#import visualize
import config
#ROOT_DIR = os.getcwd()
###############
MODEL_PATH = os.path.join(ROOT_DIR, 'pretrained_weights/mask_rcnn_coco.h5')
#MODEL_PATH = os.path.join(ROOT_DIR, 'saved_weights/heads/mask_rcnn_cows_0001.h5')
##############
MODEL_DIR = os.path.join(ROOT_DIR, 'logs')
#MODEL_DIR = os.path.join(ROOT_DIR, 'pretrained_weights')
#IMAGE_DIR = 'trial'
#IMAGE_DIR = '/hdd5/home/ICTDOMAIN/453615/NewData/total_cows_large_clean/validation_set_clean'
IMAGE_DIR = '\\Users\\Danish\\Mask_RCNN\\cow_imgs_val_large1'
#IMAGE_DIR = '/hdd5/home/ICTDOMAIN/453615/cow_imgs_val_large'
#SAVE_DIR = 'res_test_allvoc_mrcnn_ft'
#SAVE_DIR = 'pascal_imgs_mrcnn'
SAVE_DIR = 'out_results_noft_crop_imgs'


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    NAME = 'BaseNetwork'
    GPU_COUNT = 1
    NUM_CLASSES = 80 + 1 
    IMAGES_PER_GPU = 1
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024
    VALIDATION_STEPS = 1
    #RPN_ANCHOR_SCALES = (16,32,64,128,256,512)
    #RPN_ANCHOR_STRIDE = 1
    #TRAIN_ROIS_PER_IMAGE = 32


#config = InferenceConfig()
#config.display()

class CowTestConfig(coco.Config):
   NAME = 'cows'
   GPU_COUNT = 1
   DETECTION_MIN_CONFIDENCE = 0.7
   IMAGES_PER_GPU = 1
   NUM_CLASSES = 80 + 1
   IMAGE_MIN_DIM = 1280
   IMAGE_MAX_DIM = 1280
   RPN_NMS_THRESHOLD = 0.7
   #VALIDATION_STEPS = 1
   # my data-specific
   #RPN_ANCHOR_SCALES = (16,32,64,128,256)
   #RPN_ANCHOR_STRIDE = 1
   #TRAIN_ROIS_PER_IMAGE = 32

config = CowTestConfig()
config.display()

#model = modellib.MaskRCNN(mode="inference", config=config, model_dir='pretrained_weights')
model = modellib.MaskRCNN(mode='inference',config=config,model_dir = 'pretrained_weights')
#MODEL_PATH = model.find_last()[1]

model.load_weights(MODEL_PATH, by_name=True)

class_names_alex=['BG','cows']

class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

for idx, val in enumerate(class_names):
    print (idx, val)

file_names = natural_sort(next(os.walk(IMAGE_DIR))[2])
#print (file_names[0], class_names_alex[1])
#file_names = [file_names[0]]

pallete = [0, 255]

for idx, im in enumerate(file_names):
   #print (im, im.shape)
   image = skimage.io.imread(os.path.join(IMAGE_DIR, im))
   print ('iii', image.shape)
   image = image[:,:,0:3]
   
#   print (image.shape)
   #print (image)
   results = model.detect([image], verbose=1)
   # Visualize results
   r = results[0]
  #print (r['class_ids'], 'xxxx')
   good_class=[20]
   #good_class=[]  
   good_preds = 0
   for pred in r['class_ids']:
      if pred in good_class:    
         good_preds += 1
       
   output_mask = np.zeros([image.shape[0],image.shape[1], good_preds], dtype=np.uint8)
   current_pred = 0
   for idw, x in enumerate(r['scores']):
      print(idw, x, r['class_ids'][idw], type(r['class_ids'][idw]), np.unique(r['masks'][:,:,idw]))
      if r['class_ids'][idw] in good_class:
            maskrcnn_mask = r['masks'][:,:,idw]==1
            output_mask[maskrcnn_mask,current_pred]=1 #
            current_pred +=1

   present_class = r['class_ids']
   print ('AA', r['class_ids'])   
   correct_class = [True if y in good_class else False for y in present_class]
   print (correct_class, present_class, good_class, r['masks'])
   print (r['masks'].shape, 'DDDDD')
   r['class_ids'] = r['class_ids'][correct_class]
   r['rois'] = r['rois'][correct_class]
   r['masks'] = r['masks'][:,:,correct_class]
   r['scores'] = r['scores'][correct_class]
   print ('BB', r['class_ids'])

   #pickle.dump(output_mask, open(os.path.join(SAVE_DIR, im.split('.')[0] + '.r'),'wb'))    
   #img = Image.fromarray(output_mask)
   #img.save(os.path.join(SAVE_DIR, str(idx)+'.png'))
   #plt.imsave(os.path.join(SAVE_DIR, im), output_mask)
   #print (r['rois'])
   visualize.display_instances(SAVE_DIR, im, image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
