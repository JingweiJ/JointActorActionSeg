import numpy as np
import tensorflow as tf
import keras.layers as KL
import keras.backend as K
import h5py


############################################################
#  Data Formatting
############################################################

def compose_image_meta(image_id, image_shape, window, active_actor_class_ids, active_action_class_ids):
    """Takes attributes of an image and puts them in one 1D array. Use
    parse_image_meta() to parse the values back.
    
    image_id: An int ID of the image. Useful for debugging.
    image_shape: [height, width, channels]
    window: (y1, x1, y2, x2) in pixels. The area of the image where the real
            image is (excluding the padding)
    active_actor_class_ids: List of actor_class_ids available in the dataset from which
        the image came. Useful if training on images from multiple datasets
        where not all classes are present in all datasets.
    active_action_class_ids: List of action_class_ids available in the dataset from
        which the image came. Useful if training on images from multiple datasets
        where not all classes are present in all datasets.
    """
    meta = np.array(
        [image_id] + # size=1
        list(image_shape) + # size=3
        list(window) + # size=4 (y1, x1, y2, x2) in image cooredinates
        list(active_actor_class_ids) + # size=num_actor_classes
        list(active_action_class_ids) # size=num_action_classes
    )
    return meta


# Two functions (for Numpy and TF) to parse image_meta tensors.
def parse_image_meta(meta):
    """Parses an image info Numpy array to its components.
    See compose_image_meta() for more details.
    """
    image_id = meta[:, 0]
    image_shape = meta[:, 1:4]
    window = meta[:, 4:8] # (y1, x1, y2, x2) window of image in in pixels
    active_class_ids = meta[:, 8:]
    return image_id, image_shape, window, active_class_ids


def parse_image_meta_graph(meta, config=None):
    """Parses a tensor that contains image attributes to its components.
    See compose_image_meta() for more details.

    meta: [batch, meta length] where meta length depends on NUM_ACTOR_CLASSES and NUM_ACTION_CLASSES
    """
    image_id = meta[:, 0]
    image_shape = meta[:, 1:4]
    window = meta[:, 4:8]
    #active_class_ids = meta[:, 8:]
    active_actor_class_ids = meta[:, 8:8+config.NUM_ACTOR_CLASSES]
    active_action_class_ids = meta[:, 8+config.NUM_ACTOR_CLASSES:]
    # active_class_ids will be further parsed into actor and action parts outside this function. return [image_id, image_shape, window, active_class_ids]
    return [image_id, image_shape, window, active_actor_class_ids, active_action_class_ids]



def mold_image(images, config):
    """Takes RGB images with 0-255 values and subtraces
    the mean pixel and converts it to float. Expects image
    colors in RGB order.
    """
    return images.astype(np.float32) - config.MEAN_PIXEL


def unmold_image(normalized_images, config):
    """Takes a image normalized with mold() and returns the original."""
    return (normalized_images + config.MEAN_PIXEL).astype(np.uint8)


def mold_rgb_clip(rgb_clip, config):
    """Takes RGB clip with 0-255 values and subtraces
    the mean pixel and converts it to float. Expects clip
    colors in RGB order.
    """
    return rgb_clip.astype(np.float32) - config.MEAN_PIXEL

def unmold_rgb_clip(normalized_rgb_clip, config):
    """Takes a image normalized with mold() and returns the original."""
    return (normalized_rgb_clip + config.MEAN_PIXEL).astype(np.uint8)




############################################################
#  Miscellenous Graph Functions
############################################################

def trim_zeros_graph(boxes):
    """Often boxes are represented with matricies of shape [N, 4] and
    are padded with zeros. This removes zero boxes.

    boxes: [N, 4] matrix of boxes.

    TODO: use this function to reduce code duplication
    """
    area = tf.boolean_mask(boxes, tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1),
                           tf.bool))


def batch_pack_graph(x, counts, num_rows):
    """Picks different number of values from each row
    in x depending on the values in counts.
    """
    outputs = []
    for i in range(num_rows):
        outputs.append(x[i, :counts[i]])
    return tf.concat(outputs, axis=0)

def flow_xy_to_xym(flow_clip):
    """ Compute flow magnitudes and concatenate them to raw xy flows.
        flow_clip: [batch, timesteps, height, width, 2]
    """
    magnitude = KL.Lambda(lambda x: K.sqrt(K.sum(K.square(x), axis=-1, keepdims=True)),
                          name='flow_magnitude')(flow_clip)
    return KL.Concatenate(axis=-1)([flow_clip, magnitude])

############################################################
#  Initializer
############################################################

def identity_initialize_temporal_layer(tlayer):
    if not tlayer.weights:
        print('no weights in %s' % tlayer.name)
        return
    kernel, bias = tlayer.weights
    assert len(kernel.shape) == 5
    assert len(bias.shape) == 1
    k, _, _, fin, fout = map(int, tlayer.weights[0].shape)
    assert k % 2 == 1, 'k must be odd number.'
    assert fin == fout, \
        ''' input channel %d does not equal to output channel %d,
            which makes it impossible to have identity weights. '''
    bias_np = np.zeros(fout)
    kernel_np = np.expand_dims(np.expand_dims(np.expand_dims(np.eye(fout), 0), 0), 0)
    # TODO: pad zeros
    kernel_np = np.pad(kernel_np,
                       (((k-1)//2, (k-1)//2),(0,0),(0,0),(0,0),(0,0)),
                       'constant',
                       constant_values=0)
    weight_value_tuples = [
        (kernel, kernel_np),
        (bias, bias_np),
    ]
    K.batch_set_value(weight_value_tuples)

def assign_coco_heads_weights(model, mrcnn_coco_pretrained_filename, verbose=True):
    '''Use specific coco pretrained weights to initialize padnet's actor cls, mask and bbox heads on A2D dataset.

    model: a keras model or its inner model if it has one.
    mrcnn_coco_pretrained_filename: file path to the coco pretrained weights to be loaded.

    A2D : COCO
    0 - BG : 0 - BG
    1 - adult : 1 - person
    2 - baby : 1 - person
    3 - ball : 33 - sports ball
    4 - bird : 15 - bird
    5 - car : 3 - car
    6 - cat : 16 - cat
    7 - dog : 17 - dog
    '''

    with h5py.File(mrcnn_coco_pretrained_filename, mode='r') as f:
        print("Assigning coco heads' weights ...")
        # COCO matrix indices to load
        coco_indices = [0,1,1,33,15,3,16,17]
        coco_bbox_indices = []
        for ind in coco_indices:
            coco_bbox_indices += list(range(ind*4, (ind+1)*4))

        mask_kernel, mask_bias = model.get_layer('rgb_mrcnn_mask').weights
        coco_mask_kernel = np.array(f['mrcnn_mask']['mrcnn_mask/kernel:0'])[:,:,:,coco_indices]
        coco_mask_bias = np.array(f['mrcnn_mask']['mrcnn_mask/bias:0'])[coco_indices]

        actor_class_kernel, actor_class_bias = model.get_layer('mrcnn_actor_class_logits').weights
        coco_class_kernel = np.array(f['mrcnn_class_logits']['mrcnn_class_logits/kernel:0'])[:,coco_indices]
        coco_class_bias = np.array(f['mrcnn_class_logits']['mrcnn_class_logits/bias:0'])[coco_indices]

        bbox_kernel, bbox_bias = model.get_layer('mrcnn_bbox_fc').weights
        coco_bbox_kernel = np.array(f['mrcnn_bbox_fc']['mrcnn_bbox_fc/kernel:0'])[:,coco_bbox_indices]
        coco_bbox_bias = np.array(f['mrcnn_bbox_fc']['mrcnn_bbox_fc/bias:0'])[coco_bbox_indices]

        weights_assign_tuples = [
            (mask_kernel, coco_mask_kernel),
            (mask_bias, coco_mask_bias),
            (actor_class_kernel, coco_class_kernel),
            (actor_class_bias, coco_class_bias),
            (bbox_kernel, coco_bbox_kernel),
            (bbox_bias, coco_bbox_bias)
        ]
        if verbose:
            print(weights_assign_tuples)

        K.batch_set_value(weights_assign_tuples)

