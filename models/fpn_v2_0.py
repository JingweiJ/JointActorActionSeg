import tensorflow as tf
import keras.backend as K
import keras.layers as KL
import sys
sys.path.append('..')
from models.roi_align import PyramidROIAlign
from models.batchnorm import BatchNorm


def fpn_action_classifier_graph(rois, feature_maps,
                                image_shape, pool_size, num_action_classes, tower, use_dropout):
    # ROI Pooling
    aligned_feature = PyramidROIAlign([pool_size, pool_size], image_shape,
                  name=tower+"_action_roi_align_classifier")([rois] + feature_maps)
    ######################## Actor classifier and bbox regressor ##############################
    # Shape: [batch, num_boxes, pool_height, pool_width, channels]
    x = KL.TimeDistributed(KL.Conv2D(1024, (pool_size, pool_size), padding="valid"),
                           name=tower+"_action_class_conv1")(aligned_feature)
    x = KL.Activation('relu')(x)
    if use_dropout:
        x = KL.Dropout(0.5)(x)

    # Another fc layer on the top of joint feature maps
    x = KL.TimeDistributed(KL.Conv2D(1024, (1, 1)),
                           name=tower+"_action_class_conv2")(x)
    x = KL.Activation('relu')(x)

    # [batch, boxes, 1024]
    x = KL.Lambda(lambda x: K.squeeze(K.squeeze(x, 3), 2),
                       name="action_pool_squeeze")(x)

    # [batch, boxes, num_action_classes]
    action_class_logits = KL.TimeDistributed(KL.Dense(num_action_classes),
                                            name='action_class_logits')(x)
    action_probs = KL.TimeDistributed(KL.Activation("softmax"),
                                     name="action_class")(action_class_logits)

    return action_class_logits, action_probs

def fpn_classifier_graph(rois, feature_maps,
                         image_shape, pool_size, num_actor_classes, tower, use_dropout):
    """ Input:
        r_rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
              coordinates from rgb tower.
        f_rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
              coordinates from flow tower.
        r_feature_maps: List of feature maps from diffent layers of the pyramid,
                      [rP[2], rP[3], rP[4], rP[5]]. Each has a different resolution.
        f_feature_maps: List of feature maps from diffent layers of the pyramid,
                      [fP[2], fP[3], fP[4], fP[5]]. Each has a different resolution.
        image_shape: [height, width, depth]
        pool_size: The width of the square feature map generated from ROI Pooling.
        num_actor_classes: number of classes of actors, which determines the depth of the results
        num_action_classes: number of classes of actions

        Returns:
        logits: [N, NUM_CLASSES] classifier logits (before softmax)
        probs: [N, NUM_CLASSES] classifier probabilities
        bbox_deltas: [N, (dy, dx, log(dh), log(dw))] Deltas to apply to 
                     proposal boxes
    """
    # ROI Pooling
    aligned_feature = PyramidROIAlign([pool_size, pool_size], image_shape,
                  name=tower+"_roi_align_classifier")([rois] + feature_maps)
    ######################## Actor classifier and bbox regressor ##############################
    # Shape: [batch, num_boxes, pool_height, pool_width, channels]
    x = KL.TimeDistributed(KL.Conv2D(1024, (pool_size, pool_size), padding="valid"),
                           name=tower+"_mrcnn_class_conv1")(aligned_feature)
    x = KL.TimeDistributed(BatchNorm(axis=3), name=tower+'_mrcnn_class_bn1')(x)
    x = KL.Activation('relu')(x)
    if use_dropout:
        x = KL.Dropout(0.5)(x)

    # Another fc layer on the top of joint feature maps
    x = KL.TimeDistributed(KL.Conv2D(1024, (1, 1)),
                           name="mrcnn_class_conv2")(x)
    x = KL.TimeDistributed(BatchNorm(axis=3),
                           name='mrcnn_class_bn2')(x)
    x = KL.Activation('relu')(x)

    # [batch, boxes, 1024]
    shared = KL.Lambda(lambda x: K.squeeze(K.squeeze(x, 3), 2),
                       name="pool_squeeze")(x)

    # Actor classifier head
    # [batch, boxes, num_actor_classes]
    mrcnn_actor_class_logits = KL.TimeDistributed(KL.Dense(num_actor_classes),
                                            name='mrcnn_actor_class_logits')(shared)
    mrcnn_actor_probs = KL.TimeDistributed(KL.Activation("softmax"),
                                     name="mrcnn_actor_class")(mrcnn_actor_class_logits)

    # BBox head
    # [batch, boxes, num_actor_classes * (dy, dx, log(dh), log(dw))]
    x = KL.TimeDistributed(KL.Dense(num_actor_classes*4, activation='linear'),
                           name='mrcnn_bbox_fc')(shared)
    # Reshape to [batch, boxes, num_classes, (dy, dx, log(dh), log(dw))]
    s = K.int_shape(x)
    mrcnn_bbox = KL.Reshape((s[1], num_actor_classes, 4), name="mrcnn_bbox")(x)

    return mrcnn_actor_class_logits, mrcnn_actor_probs, mrcnn_bbox


def build_fpn_mask_graph(rois, feature_maps,
                         image_shape, pool_size, num_classes, tower):
    """Builds the computation graph of the mask head of Feature Pyramid Network.

    aligned_feature: [batch, num_rois, pool_height, pool_width, channels], roi aligned mrcnn feature
    num_classes: number of classes, which determines the depth of the results
    tower: 'rgb' or 'flow'

    Returns: Masks [batch, num_rois, height, width, num_classes]
    """
    # ROI Pooling
    # Shape: [batch, boxes, pool_height, pool_width, channels]
    aligned_feature = PyramidROIAlign([pool_size, pool_size], image_shape,
                        name=tower+'_'+"roi_align_mask")([rois] + feature_maps)
    # Conv layers
    # Shape: [batch, boxes, pool_height, pool_width, channels]
    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name=tower+'_'+"mrcnn_mask_conv1")(aligned_feature)
    x = KL.TimeDistributed(BatchNorm(axis=3),
                           name=tower+'_'+'mrcnn_mask_bn1')(x)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name=tower+'_'+"mrcnn_mask_conv2")(x)
    x = KL.TimeDistributed(BatchNorm(axis=3),
                           name=tower+'_'+'mrcnn_mask_bn2')(x)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name=tower+'_'+"mrcnn_mask_conv3")(x)
    x = KL.TimeDistributed(BatchNorm(axis=3),
                           name=tower+'_'+'mrcnn_mask_bn3')(x)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name=tower+'_'+"mrcnn_mask_conv4")(x)
    x = KL.TimeDistributed(BatchNorm(axis=3),
                           name=tower+'_'+'mrcnn_mask_bn4')(x)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2DTranspose(256, (2,2), strides=2, activation="relu"),
                           name=tower+'_'+"mrcnn_mask_deconv")(x)
    x = KL.TimeDistributed(KL.Conv2D(num_classes, (1, 1), strides=1, activation="sigmoid"),
                           name=tower+'_'+"mrcnn_mask")(x)
    # TODO: how does the size of output masks here match with target masks?
    return x
