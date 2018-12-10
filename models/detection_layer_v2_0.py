import sys
sys.path.append('..')
import utils.matterport_utils as matterport_utils

import numpy as np
import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.initializers as KI
import keras.engine as KE
import keras.models as KM
from models.model_misc import parse_image_meta


def clip_to_window(window, boxes):
    """
    window: (y1, x1, y2, x2). The window in the image we want to clip to.
    boxes: [N, (y1, x1, y2, x2)]
    """
    boxes[:, 0] = np.maximum(np.minimum(boxes[:, 0], window[2]), window[0])
    boxes[:, 1] = np.maximum(np.minimum(boxes[:, 1], window[3]), window[1])
    boxes[:, 2] = np.maximum(np.minimum(boxes[:, 2], window[2]), window[0])
    boxes[:, 3] = np.maximum(np.minimum(boxes[:, 3], window[3]), window[1])
    return boxes


def refine_detections(rois, actor_probs, action_probs, deltas, window, config, return_keep_idx=True):
    """Refine classified proposals and filter overlaps and return final
    detections.

    Inputs:
        rois: [N, (y1, x1, y2, x2)] in normalized coordinates
        actor_probs: [N, num_actor_classes]. Actor class probabilities.
        action_probs: [N, num_action_classes]. Actor class probabilities.
        deltas: [N, num_classes, (dy, dx, log(dh), log(dw))]. Class-specific
                bounding box deltas.
        window: (y1, x1, y2, x2) in image coordinates. The part of the image
            that contains the image excluding the padding.

    Returns detections shaped: [N, (y1, x1, y2, x2, class_id, score)]
    """
    # Class IDs per ROI
    actor_class_ids = np.argmax(actor_probs, axis=1)
    action_class_ids = np.argmax(action_probs, axis=1)
    # Class probability of the top class of each ROI
    actor_class_scores = actor_probs[np.arange(actor_class_ids.shape[0]), actor_class_ids]
    action_class_scores = action_probs[np.arange(action_class_ids.shape[0]), action_class_ids]
    # Class-specific bounding box deltas
    deltas_specific = deltas[np.arange(deltas.shape[0]), actor_class_ids]
    # Apply bounding box deltas
    # Shape: [boxes, (y1, x1, y2, x2)] in normalized coordinates
    refined_rois = matterport_utils.apply_box_deltas(
        rois, deltas_specific * config.BBOX_STD_DEV)
    # Convert coordiates to image domain
    # TODO: better to keep them normalized until later
    height, width = config.IMAGE_SHAPE[:2]
    refined_rois *= np.array([height, width, height, width])
    # Clip boxes to image window
    refined_rois = clip_to_window(window, refined_rois)
    # Round and cast to int since we're deadling with pixels now
    refined_rois = np.rint(refined_rois).astype(np.int32)

    # TODO: Filter out boxes with zero area

    # Filter out background boxes
    #keep = np.where(actor_class_ids > 0)[0]
    keep = np.where(np.logical_and(
        actor_class_ids > 0,
        action_class_ids > 0,
    ))[0]
    # Filter out low confidence boxes
    if config.DETECTION_MIN_CONFIDENCE and config.DETECTION_ACTION_MIN_CONFIDENCE:
        #keep = np.intersect1d(
        #    keep, np.where(np.logical_and(
        #        actor_class_scores >= config.DETECTION_MIN_CONFIDENCE,
        #        action_class_scores >= config.DETECTION_MIN_CONFIDENCE,
        #    ))[0]) # TODO: also use action score to filter?
        keep = np.intersect1d(
            keep, np.where(np.logical_and(
                actor_class_scores >= config.DETECTION_MIN_CONFIDENCE,
                action_class_scores >= config.DETECTION_ACTION_MIN_CONFIDENCE,
            ))[0]) # TODO: also use action score to filter?

    # Apply per-class NMS # TODO: how to include action nms here?
    pre_nms_class_ids = actor_class_ids[keep]
    pre_nms_scores = actor_class_scores[keep] # * action_class_scores[keep]
    pre_nms_rois = refined_rois[keep]
    nms_keep = []
    for actor_class_id in np.unique(pre_nms_class_ids):
        # Pick detections of this class
        ixs = np.where(pre_nms_class_ids == actor_class_id)[0]
        # Apply NMS
        class_keep = matterport_utils.non_max_suppression(
            pre_nms_rois[ixs], pre_nms_scores[ixs],
            config.DETECTION_NMS_THRESHOLD)
        # Map indicies
        class_keep = keep[ixs[class_keep]]
        nms_keep = np.union1d(nms_keep, class_keep)
    keep = np.intersect1d(keep, nms_keep).astype(np.int32)

    # Keep top detections
    roi_count = config.DETECTION_MAX_INSTANCES
    top_ids = np.argsort(actor_class_scores[keep])[::-1][:roi_count]
    keep = keep[top_ids]

    # Arrange output as [N, (y1, x1, y2, x2, actor_class_id, action_class_id, actor_class_score, action_class_score)]
    # Coordinates are in image domain.
    result = np.hstack((refined_rois[keep],
                        actor_class_ids[keep][..., np.newaxis],
                        action_class_ids[keep][..., np.newaxis],
                        actor_class_scores[keep][..., np.newaxis],
                        action_class_scores[keep][..., np.newaxis]
                       ))
    if return_keep_idx:
        return result, keep
    else:
        return result


class DetectionLayer(KE.Layer):
    """Takes classified proposal boxes and their bounding box deltas and
    returns the final detection boxes.

    # TODO: Add support for batch_size > 1

    Returns:
    [batch, num_detections, (y1, x1, y2, x2, actor_class_id, action_class_id, actor_class_score, action_class_score)] in pixels
    """
    def __init__(self, config=None, **kwargs):
        super(DetectionLayer, self).__init__(**kwargs)
        self.config = config

    def call(self, inputs):
        def wrapper(rois, mrcnn_actor_class, mrcnn_action_class, mrcnn_bbox, image_meta):
            # currently supports one image per batch
            b = 0
            _, _, window, _ = parse_image_meta(image_meta)
            detections, keep_idx = refine_detections(
                rois[b], mrcnn_actor_class[b], mrcnn_action_class[b], mrcnn_bbox[b], window[b], self.config,
                return_keep_idx=True)

            # Pad with zeros if detections < DETECTION_MAX_INSTANCES
            gap = self.config.DETECTION_MAX_INSTANCES - detections.shape[0]
            assert gap >= 0
            if gap > 0:
                detections = np.pad(detections, [(0, gap), (0, 0)],
                                    'constant', constant_values=0)

            # Cast to float32
            # TODO: track where float64 is introduced
            detections = detections.astype(np.float32)
            keep_idx = keep_idx.astype(np.int64)

            # Reshape output
            # [batch, num_detections, (y1, x1, y2, x2, actor_class_id, action_class_id, \
            # actor_class_score, action_class_score)] in pixels
            return [np.reshape(detections, [1, self.config.DETECTION_MAX_INSTANCES, 8]), np.reshape(keep_idx, [1, -1])]

        # Return wrapped function
        return tf.py_func(wrapper, inputs, [tf.float32, tf.int64])

    def compute_output_shape(self, input_shape):
        return [(None, self.config.DETECTION_MAX_INSTANCES, 8), (None, None)]

    def compute_mask(self, inputs, mask=None):
        return [None, None]
