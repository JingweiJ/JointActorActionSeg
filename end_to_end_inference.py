import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.initializers as KI
import keras.engine as KE
import keras.models as KM
import numpy as np
import os
import cv2
import skimage
import matplotlib.pyplot as plt
import sys
from cfg.config import Config
import utils.matterport_utils as matterport_utils
import utils.matterport_visualize as matterport_visualize
import utils.flowlib as flowlib
import utils.flow as flow
from models import data_generator, model_misc

from models.padnet_v2_1 import PADNet
from dataIO.a2d_dataset import A2DDataset
from models.data_generator import load_image_gt_preprocessed, load_image_gt

from distutils.version import LooseVersion
assert LooseVersion(tf.__version__) >= LooseVersion("1.3")
assert LooseVersion(keras.__version__) >= LooseVersion('2.0.8')

dataset_dir = 'A2D'
vids = []
with open(os.path.join(dataset_dir, 'inference_video.txt'), 'r') as f:
	vids = f.readlines()


vid_name = 'test_video'
vid_file = vid_name+'.mp4'
if vid_name+'\n' in vids:
	new_vid = False
	#ALREADY RAN VID
	print('ALREADY HAVE VID: ' + vid_file)
else:
	new_vid = True
	print('Preprocessing video: ' + vid_file)
	vidcap = cv2.VideoCapture(vid_file)
	success,image = vidcap.read()
	prev_frame = image
	count = 0
	image_dir = os.path.join(dataset_dir,'Images',vid_name)
	flow_dir = os.path.join(dataset_dir,'TVL1FlowsPrevToCurr',vid_name)
	os.mkdir(image_dir)
	os.mkdir(flow_dir)

	while success:
		count += 1
		#write image
		filename = '%05d' % count
		cv2.imwrite(os.path.join(image_dir,filename+'.png'), image)     # save frame as png file      
		
		#compute and write flow
		fl = flow.generate_flow(prev_frame,image)
		flow.write_flow(fl, os.path.join(flow_dir,filename+'.flo'))


		prev_frame = image
		success,image = vidcap.read()
		print('Read a new frame: ', success)
	with open(os.path.join(dataset_dir, 'inference_video.txt'), 'a') as f:
		f.write(vid_name+'\n')


#RUN INFERENCE
os.getenv('CUDA_VISIBLE_DEVICES')

MODEL_DIR = 'outputs/padnet_v1.0_overfit_test'


class A2DConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "a2d"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_ACTOR_CLASSES = 1 + 7  # background + 7 objects
    NUM_ACTION_CLASSES = 1 + 9

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 64

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 1000

    # use small validation steps since the epoch is small
    VALIDATION_STPES = 100

    TIMESTEPS = 5

    BATCH_SIZE = IMAGES_PER_GPU * GPU_COUNT
    IMAGE_H = 256
    IMAGE_W = 256
    IMAGE_SHAPE = np.array([IMAGE_H, IMAGE_W, 3])
    RGB_CLIP_SHAPE = np.array([TIMESTEPS, IMAGE_H, IMAGE_W, 3])
    FLOW_CLIP_SHAPE = np.array([TIMESTEPS, IMAGE_H, IMAGE_W, 2])

    DETECTION_MIN_CONFIDENCE = 0.7
    DETECTION_ACTION_MIN_CONFIDENCE = 0.3

    RESNET = 'resnet101'
    USE_DROPOUT = False

class InferenceConfig(A2DConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()


# Recreate the model in inference mode
model = PADNet(mode="inference", config=inference_config, model_dir=MODEL_DIR)

model_path = 'padnet_a2d.h5'
with tf.device('/cpu:0'):
	model.load_weights(model_path, by_name=True, verbose=True, verboseverbose=True) 


dataset_val = A2DDataset('inference_video', 'A2D')
dataset_val.prepare()

model.config.DETECTION_MIN_CONFIDENCE = 0.7
model.config.DETECTION_ACTION_MIN_CONFIDENCE = 0.3

image_ids = dataset_val.image_ids

print('READY TO DETECT on ', image_ids)
actor_APs, action_APs, results, gt_boxeses = [], [], [], []
gts = []
for i, image_id in enumerate(image_ids):
    if i % 200 == 0:
        print('[%04d/%04d] running mean - actor mAP: %.03f; action mAP: %.03f' % 
              (i, len(image_ids), np.mean(actor_APs), np.mean(action_APs)))
    _, image_meta, rgb_clip, flow_clip, labeled_frame_id, _, _ = \
        load_image_gt(dataset_val, inference_config, image_id)

    rgb_clip = model_misc.mold_rgb_clip(rgb_clip, inference_config)
    print(i)
    result = model.detect(np.expand_dims(rgb_clip, 0), 
                          np.expand_dims(flow_clip, 0), 
                          np.expand_dims(labeled_frame_id, 0), 
                          np.expand_dims(image_meta, 0))

    image = dataset_val.load_image(image_id)
    results.append(result)
    gts.append(
        {
            'image': image,
        }
    )


    # actor_APs.append(actor_AP); action_APs.append(action_AP)
print("actor mAP: ", np.mean(actor_APs))
print("action mAP: ", np.mean(action_APs))
actor_class_names = ['BG', 'adult', 'baby', 'ball', 'bird', 'car', 'cat', 'dog']
action_class_names = ['BG', 'climbing', 'crawling', 'eating', 'flying', 'jumping', 'rolling', 'running', 'walking', 'none']


# writing results to video
# out = cv2.VideoWriter('test_video_results.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (image.shape[1],image.shape[0]))
if new_vid:
	actor_dir = os.path.join('A2D','Detections', vid_name,'actor_segmentations')
	action_dir = os.path.join('A2D','Detections', vid_name,'action_segmentations')
	combined_dir = os.path.join('A2D','Detections', vid_name,'combined_segmentations')
	os.mkdir(actor_dir)
	os.mkdir(action_dir)
	os.mkdir(combined_dir)

for i, image_id in enumerate(image_ids):
    filename = '%05d.png' % (i+1)
    print('saving', filename)
    actor_file_name = os.path.join('A2D','Detections', vid_name,'actor_segmentations',filename)
    action_file_name = os.path.join('A2D','Detections', vid_name,'action_segmentations',filename)
    file_name = os.path.join('A2D','Detections', vid_name,'combined_segmentations',filename)
    matterport_visualize.save_detections(
    	gts[i]['image'], results[i]['rois'], results[i]['masks'], results[i]['actor_class_ids'], 
    	actor_class_names, results[i]['actor_scores'], results[i]['rois'], results[i]['masks'], results[i]['action_class_ids'], 
    	action_class_names, results[i]['action_scores'], file_name,
    	title=None)

    matterport_visualize.save_detection(
    	gts[i]['image'], results[i]['rois'], results[i]['masks'], results[i]['actor_class_ids'], 
    	actor_class_names,actor_file_name, scores=results[i]['actor_scores'], 
    	title='actors')

    matterport_visualize.save_detection(
    	gts[i]['image'], results[i]['rois'], results[i]['masks'], results[i]['action_class_ids'], 
    	action_class_names,action_file_name, scores=results[i]['action_scores'], 
    	title='actions')



