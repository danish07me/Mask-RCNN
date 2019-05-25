import os,sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from config import Config
from PIL import Image
from skimage.measure import label, regionprops
import utils
import model as modellib
#import visualize
from model import log
import argparse
import coco
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import pickle
from skimage.transform import resize
root_dir = os.getcwd()
MODEL_DIR = os.path.join(root_dir, 'logs/onlycoco')
#pretrained_model = 'pretrained_weights/mask_rcnn_coco.h5'
pretrained_model = os.path.join(MODEL_DIR, "cows20180815T0117/mask_rcnn_cows_0005.h5")
#pretrained_model = os.path.join(MODEL_DIR, "cows20180709T1147/mask_rcnn_cows_0003.h5")
#new_model = os.path.join(MODEL_DIR, "cows20180109T1344/mask_rcnn_cows_0002.h5")

def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


class CowConfig(Config):
   NAME = 'cows'
   GPU_COUNT = 1
   IMAGES_PER_GPU = 2
   NUM_CLASSES = 1 + 1
   DETECTION_MIN_CONFIDENCE = 0.5
   IMAGE_MIN_DIM = 512
   IMAGE_MAX_DIM = 512
   VALIDATION_STEPS = 1
   RPN_ANCHOR_SCALES = (16,32,64,128,256)
   TRAIN_ROIS_PER_IMAGE = 32

   STEPS_PER_EPOCH = 5000
   VALIDATION_STEPS = 5

class CowTestConfig(Config):
   NAME = 'cows'
   GPU_COUNT = 1
   DETECTION_MIN_CONFIDENCE = 0.5
   IMAGES_PER_GPU = 1
   NUM_CLASSES = 1 + 1
   IMAGE_MIN_DIM = 1280
   IMAGE_MAX_DIM = 1280
   VALIDATION_STEPS = 5
   #RPN_ANCHOR_SCALES = (16,32,64,128,256)
   #TRAIN_ROIS_PER_IMAGE = 32

   # roughly 5000 divides 4, the mini-batch size, I think...
   #STEPS_PER_EPOCH = 300
   #VALIDATION_STEPS = 5

class CowDataset(utils.Dataset):

   # this creates the 'dataset' without loading the actual images
   def load_cows(self, count, stage):
      # randomize the set
      self.add_class("cow_dataset", 1, "cow")
      # if this is a validation set, select 100 imgaes at random
      if stage == 'validation':
           list_of_validation_items = np.random.choice(range(count), size= 2, replace=False)
      # idx will be a random number
      if stage == 'training':
         for fs in os.listdir('/Users/dhananjaykittur/mask/kaggle-ds-bowl-2018-baseline/cow_imgs_train'):
            if fs.split('.')[1] == 'png':
              #self.add_image("cow_dataset", image_id = idx, path = os.path.join('/home/ICTDOMAIN/453615/NewData/total_cows_large_clean/total_training_clean', str(idx) + '.png'), width=width, height=height, stage = 'training')
              self.add_image("cow_dataset", image_id = fs.split('.')[0], path = os.path.join('/Users/dhananjaykittur/mask/kaggle-ds-bowl-2018-baseline/cow_imgs_train', fs), stage = 'training')

              print (fs, count, self._image_ids)
      elif stage == 'validation':

          for fs in os.listdir('/Users/dhananjaykittur/mask/kaggle-ds-bowl-2018-baseline/cow_imgs_val'):
             if fs.split('.')[1] == 'png':
               #self.add_image("cow_dataset", image_id = idx, path = os.path.join('/home/ICTDOMAIN/453615/NewData/total_cows_large_clean/total_training_clean', str(idx) + '.png'), width=width, height=height, stage = 'training')
               self.add_image("cow_dataset", image_id = fs.split('.')[0], path = os.path.join('/Users/dhananjaykittur/mask/kaggle-ds-bowl-2018-baseline/cow_imgs_val', fs), stage = 'validation')
               print (fs, count, self._image_ids)
         #full_list = natural_sort(os.listdir('/home/ICTDOMAIN/453615/NewData/total_cows_large_clean/validation_set_clean'))
         #full_list = natural_sort(os.listdir('/Users/dhananjaykittur/mask/kaggle-ds-bowl-2018-baseline/cow_imgs_val'))

         #for idx in list_of_validation_items:
             #print (idx, count, self._image_ids)
             #self.add_image("cow_dataset", image_id = idx, path = os.path.join('/Users/dhananjaykittur/mask/kaggle-ds-bowl-2018-baseline/cow_imgs_val', full_list[idx]), width=width, height=height, stage= '$
             #self.add_image("cow_dataset", image_id = idx, path = os.path.join('/home/ICTDOMAIN/453615/NewData/total_cows_large_clean/validation_set_clean', full_list[idx]), width=width, height=height, stage= 'validation', num_img = full_list[idx])
      elif stage == 'testing_mydata':
          for fs in os.listdir('/Users/dhananjaykittur/mask/kaggle-ds-bowl-2018-baseline/cow_imgs_val_large'):
             if fs.split('.')[1] == 'png':
               #self.add_image("cow_dataset", image_id = idx, path = os.path.join('/home/ICTDOMAIN/453615/NewData/total_cows_large_clean/total_training_clean', str(idx) + '.png'), width=width, height=height, stage = 'training')
               self.add_image("cow_dataset", image_id = fs.split('.')[0], path = os.path.join('/Users/dhananjaykittur/mask/kaggle-ds-bowl-2018-baseline/cow_imgs_val_large', fs), stage = 'testing_mydata')
               print (fs, count, self._image_ids)
        #full_list = natural_sort(os.listdir('/home/ICTDOMAIN/453615/NewData/voc_cows_val')
        #  full_list = natural_sort(os.listdir('/Users/dhananjaykittur/mask/kaggle-ds-bowl-2018-baseline /cow_imgs_val_large'))
         #full_list = natural_sort(os.listdir('/home/ICTDOMAIN/453615/NewData/total_cows_large_clean/Frames500_valid4'))
        #  for idx in full_list:
        #      self.add_image("cow_dataset", image_id = idx, path = os.path.join('/hdd5/home/ICTDOMAIN/453615/cow_imgs_val_large', idx), stage= 'testing_mydata', num_img = idx)
        #      #self.add_image("cow_dataset", image_id = idx, path = os.path.join('/home/ICTDOMAIN/453615/NewData/voc_cows_val', idx), stage= 'testing_mydata', num_img = idx)

   def load_mask(self, image_id):

       if self.image_info[image_id]['stage'] == 'training':
          #path = '/home/ICTDOMAIN/453615/NewData/total_cows_large_clean/total_training_labels_clean'
          #this_mask = np.array(Image.open(os.path.join(path, str(image_id) + '.png')))
          path = '/Users/dhananjaykittur/mask/kaggle-ds-bowl-2018-baseline/cow_imgs_gt_train'
          #this_mask = np.array(Image.open(os.path.join(path, str(image_id) + '.png')))
          # ALEX: unpickle the file
          #print (self.image_info[image_id], image_id, "AAAAAAAASDFF")
          this_mask = np.array(pickle.load(open(os.path.join(path, self.image_info[image_id]['id'] + '.r'), "rb")))
          #print ('YATTTTTA', os.path.join(path, self.image_info[image_id]['id'] + '.r'))

       elif self.image_info[image_id]['stage'] == 'validation' or self.image_info[image_id]['stage'] == 'testing_mydata':
          #full_list = natural_sort(os.listdir('/home/ICTDOMAIN/453615/NewData/voc_cows_gt_val_clean_contour'))
          #path = '/home/ICTDOMAIN/453615/NewData/voc_cows_gt_val_clean_contour'
          #full_list = natural_sort(os.listdir('/Users/dhananjaykittur/mask/kaggle-ds-bowl-2018-baseline/cow_imgs_gt_val_large_allbits'))
          #path = '/Users/dhananjaykittur/mask/kaggle-ds-bowl-2018-baseline/cow_imgs_gt_val_large_allbits'
          path = '/Users/dhananjaykittur/mask/kaggle-ds-bowl-2018-baseline/cows_imgs_val_gt_large'
          #print (image_id, self.image_info[image_id]['num_img'], full_list[image_id])
          #this_mask = pickle.load(open(os.path.join(path, image_id + '.r'), "rb"))
          this_mask = np.array(pickle.load(open(os.path.join(path, self.image_info[image_id]['id'] + '.r'), "rb")))
          #print ('RDRDRDRD', os.path.join(path, self.image_info[image_id]['id'] + '.r'))
          #this_mask = np.array(Image.open(os.path.join(path, full_list[image_id])))
          # ALEX: Unpickle the input file that you upload from the source with gt
          #path = '/home/ICTDOMAIN/453615/cow_imgs_gt'
          #this_mask = np.array(Image.open(os.path.join(path, str(self.image_info[image_id]['num_img'])))))

       class_ids = np.array(10*[1], dtype = np.int32)
       return this_mask, class_ids

   def image_reference(self, path, img_id):
        return os.path.join(path, str(img_id) + '.png')



if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Mask R-CNN on cow data')
    parser.add_argument("stage", help="'train' or 'val'")
    args = parser.parse_args()
    if args.stage == 'train':
         # load model
         source = '/Users/dhananjaykittur/mask/kaggle-ds-bowl-2018-baseline'
         print (MODEL_DIR)
         config = CowConfig()
         config.display()
         model = modellib.MaskRCNN(mode="training", config = config, model_dir=MODEL_DIR)
         model.load_weights(pretrained_model, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

         #print (model.find_last()[1], 'LAST')
         #model.load_weights(model.find_last()[1], by_name=True)
         #model.load_weights(pretrained_model, by_name=True)
         print ('Weigths loaded')
         # load dataset
         train_dataset = CowDataset()
         count = len(os.listdir(os.path.join(source, 'cow_imgs_train')))
         print ('c', count)
         train_dataset.load_cows(count = count, stage = 'training')
         train_dataset.prepare()
         val_dataset = CowDataset()
         count = len(os.listdir(os.path.join(source, 'cow_imgs_val')))
         print ('s', count)
         val_dataset.load_cows(count=count, stage = 'validation')
         val_dataset.prepare()
         print ('train', train_dataset.image_ids)
         print ('val', val_dataset.image_ids)
         # OK train model#############################
         # HEADS ONLY! same learning rate as in config
         #############################################
         model.train(train_dataset, val_dataset, learning_rate = config.LEARNING_RATE, epochs=5, layers='all')

    elif args.stage == 'inference_mydata':

         #source = '/home/ICTDOMAIN/453615/NewData/total_cows_large_clean/'
         # just for the base model
         good_classes=[20]
         source = '/Users/dhananjaykittur/mask/kaggle-ds-bowl-2018-baseline'
         config = CowTestConfig()
         config.display()
         model = modellib.MaskRCNN(mode="inference", config = config, model_dir='pretrained_weights')
         model.load_weights(pretrained_model, by_name=True)
         print ("Weights loaded")
         # load dataset
         test_dataset = CowDataset()
         count = len(os.listdir(os.path.join(source, 'cow_imgs_val_large')))
         #count = len(os.listdir(os.path.join(source, 'cow_imgs_val')))
         test_dataset.load_cows(count=count, stage = 'testing_mydata')
         test_dataset.prepare()
         print("Images: {}\nClasses: {}".format(len(test_dataset.image_ids), test_dataset.class_names))
         #print (test_dataset.image_ids)
         all_AP_list=[]
         thresholds = np.linspace(0.5,0.95,10)
         for th in thresholds:
           AP_list=[]
           for img in test_dataset.image_ids:
                #image, image_meta, gt_mask = modellib.load_image_gt(test_dataset, config, img)
                image, image_meta, class_ids, gt_bbox, gt_mask= modellib.load_image_gt(test_dataset, config, img)
                print (img)
                plt.imsave(str(img)+'.png', image)
                mask,gt_class_ids = test_dataset.load_mask(img)
                #print ('mask', mask.shape, image.shape)
                results = model.detect([image], verbose=1)
                r = results[0]
                #print (r.keys())
                plt.imsave('gt' + str(img)+'.png', mask[:,:,0])
                #print ("get class ids")
                #print (r['rois'], r['class_ids'], r['scores'])
                # for the base model-keep only cow predictions
                present_class = r['class_ids']
                for idx, cls in enumerate(present_class):
                    if cls not in good_classes:
                          np.delete(r['rois'], idx, axis=0)
                          np.delete(r['class_ids'], idx, axis=0)
                          np.delete(r['scores'], idx, axis=0)
                #print ('class', r['rois'].shape, r['scores'].shape, r['class_ids'].shape, r['masks'].shape, gt_bbox.shape, mask.shape)
                AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_ids, mask, r['rois'], r['class_ids'], r['scores'], r['masks'], iou_threshold = th)
                print (AP, precisions, recalls)
                AP_list.append(AP)
                print (np.mean(AP_list), th)

           all_AP_list.append(np.mean(AP_list))
           print ("mAP", np.mean(AP_list))
         all_AP_list.append(np.mean(all_AP_list))


print ('all_AP_list', all_AP_list)

#pickle.dump(str(all_AP_list), open("coco_cows_val_base_640.r","wb"))
