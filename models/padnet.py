import os, sys, glob, random, math, datetime, itertools, json, re, logging, numpy as np,\
    scipy.misc, tensorflow as tf
import keras, keras.backend as K, keras.layers as KL, keras.initializers as KI, \
    keras.engine as KE, keras.models as KM
from collections import OrderedDict

sys.path.append('..')
import utils.matterport_utils as matterport_utils
from models.model_misc import *
from models.batchnorm import BatchNorm
from models.data_generator import data_generator
#from resnet import resnet_graph
from models.i3d_v0_0 import i3d_graph
from models.rpn import build_rpn_model
from models.proposal_layer import ProposalLayer
from models.detection_layer import DetectionLayer
from models.detection_target_layer import DetectionTargetLayer
from models.fpn import fpn_classifier_graph, build_fpn_mask_graph
from models.loss import rpn_class_loss_graph, rpn_bbox_loss_graph, mrcnn_class_loss_graph, \
    mrcnn_bbox_loss_graph, mrcnn_mask_loss_graph
from models.parallel_model import ParallelModel
from distutils.version import LooseVersion

def log(text, array=None):
    """Prints a text message. And, optionally, if a Numpy array is provided it
    prints it's shape, min, and max values.
    """
    if array is not None:
        text = text.ljust(25)
        text += ("shape: {:20}  min: {:10.5f}  max: {:10.5f}".format(
            str(array.shape),
            array.min() if array.size else "",
            array.max() if array.size else ""))
    print(text)

class PADNet():
    """Encapsulates the PAD Net model functionality.
    The actual Keras model is in the keras_model property.
    """

    def __init__(self, mode, config, model_dir):
        """
        mode: Either "training" or "inference"
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        assert mode in ['training', 'inference']
        self.version = LooseVersion('0.0')
        self.mode = mode
        self.config = config
        self.model_dir = model_dir
        self.set_log_dir()
        self.keras_model = self.build(mode=mode, config=config)

    def set_trainable(self, layer_regex, keras_model=None, indent=0, verbose=1):
        """Sets model layers as trainable if their names match
        the given regular expression.
        """
        # Print message on the first call (but not on recursive calls)
        if verbose > 0 and keras_model is None:
            log("Selecting layers to train")

        keras_model = keras_model or self.keras_model

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
            else keras_model.layers

        for layer in layers:
            # Is the layer a model?
            if layer.__class__.__name__ == 'Model':
                print("In model: ", layer.name)
                self.set_trainable(layer_regex, keras_model=layer, indent=indent+4)
                continue

            if not layer.weights:
                continue
            # Is it trainable?
            trainable = bool(re.fullmatch(layer_regex, layer.name))
            # Update layer. If layer is a container, update inner layer.
            if layer.__class__.__name__ == 'TimeDistributed':
                layer.layer.trainable = trainable
            else:
                layer.trainable = trainable
            # Print trainble layer names
            if trainable and verbose > 0:
                log("{}{:20}   ({})".format(" " * indent, layer.name,
                                            layer.__class__.__name__))

    def compile(self, learning_rate, momentum):
        """Gets the model ready for training. Adds losses, regularization, and
        metrics. Then calls the Keras compile() function.
        """
        # Optimizer object
        optimizer = keras.optimizers.SGD(lr=learning_rate, momentum=momentum,
                                         clipnorm=5.0)
        # Add Losses
        # First, clear previously set losses to avoid duplication
        self.keras_model._losses = []
        self.keras_model._per_input_losses = {}
        loss_names = ["rpn_class_loss", "rpn_bbox_loss", "mrcnn_actor_class_loss",
                    "mrcnn_action_class_loss", "mrcnn_bbox_loss", "mrcnn_mask_loss"]
        for name in loss_names:
            layer = self.keras_model.get_layer(name)
            if layer.output in self.keras_model.losses:
                continue
            self.keras_model.add_loss(tf.reduce_mean(layer.output, keep_dims=True))

        # Add L2 Regularization
        reg_losses = [keras.regularizers.l2(self.config.WEIGHT_DECAY)(w)
                    for w in self.keras_model.trainable_weights]
        self.keras_model.add_loss(tf.add_n(reg_losses))

        # Compile
        self.keras_model.compile(optimizer=optimizer, loss=[None]*len(self.keras_model.outputs))

        # Add metrics
        for name in loss_names:
            if name in self.keras_model.metrics_names:
                continue
            layer = self.keras_model.get_layer(name)
            self.keras_model.metrics_names.append(name)
            self.keras_model.metrics_tensors.append(tf.reduce_mean(layer.output,
                                                    keep_dims=True))

    def train(self, train_dataset, val_dataset, learning_rate, epochs, layers):
        """Train the model.
        train_dataset, val_dataset: Training and validation Dataset objects.
        learning_rate: The learning rate to train with
        epochs: Number of training epochs. Note that previous training epochs
                are considered to be done alreay, so this actually determines
                the epochs to train in total rather than in this particaular
                call.
        layers: Allows selecting wich layers to train. It can be:
            - A regular expression to match layer names to train
            - One of these predefined values:
              heaads: The RPN, classifier and mask heads of the network
              all: All the layers
              3+: Train Resnet stage 3 and up
              4+: Train Resnet stage 4 and up
              5+: Train Resnet stage 5 and up
        """
        assert self.mode == "training", "Create model in training mode."

        # Pre-defined layer regular expressions
        layer_regex = { # TODO: modify the regular expressions
            # all layers but the backbone
            "heads": r"(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # From Resnet stage 4 layers and up
            "3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "5+": r"(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # All layers
            "all": ".*",
        }
        if layers in layer_regex.keys():
            layers = layer_regex[layers]

        # Data generators
        train_generator = data_generator(train_dataset, self.config, shuffle=True, 
                                                batch_size=self.config.BATCH_SIZE)
        val_generator = data_generator(val_dataset, self.config, shuffle=True, 
                                            batch_size=self.config.BATCH_SIZE)

        # Callbacks
        callbacks = [
            keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                        histogram_freq=0, write_graph=True, write_images=False),
            keras.callbacks.ModelCheckpoint(self.checkpoint_path,
                                            verbose=0, save_weights_only=True),
        ]
        
        # Common parameters to pass to fit_generator()
        fit_kwargs = {
            "steps_per_epoch": self.config.STEPS_PER_EPOCH, # TODO: how is it related to dataset size?
            "callbacks": callbacks,
            "validation_data": next(val_generator),
            "validation_steps": self.config.VALIDATION_STPES,
            "max_queue_size": 100,
            "workers": max(self.config.BATCH_SIZE // 2, 2), # TODO: may want more than 2
            "use_multiprocessing": True,
        }
        
        # Train
        #log("\nStarting at epoch {}. LR={}\n".format(self.epoch, learning_rate)) # TODO: what's "log"?
        #log("Checkpoint Path: {}".format(self.checkpoint_path))
        print("\nStarting at epoch {}. LR={}\n".format(self.epoch, learning_rate)) # TODO: what's "log"?
        print("Checkpoint Path: {}".format(self.checkpoint_path))
        self.set_trainable(layers)
        self.compile(learning_rate, self.config.LEARNING_MOMENTUM)

        self.keras_model.fit_generator(
            train_generator,
            initial_epoch=self.epoch,
            epochs=epochs,
            **fit_kwargs
            )
        self.epoch = max(self.epoch, epochs)

    def build(self, mode, config):
        """ Build PADNet architecture.
            mode: either "training" or "inference". The inputs and outputs of the model
            differ accordingly.
        """
        assert mode in ['training', 'inference']

        # Image size must be dividable by 2 multiple times
        h, w = config.IMAGE_SHAPE[:2]
        if h/2**6 != int(h/2**6) or w/2**6 != int(w/2**6):
            raise Exception("Image size must be dividable by 2 at least 6 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")

        # Inputs:
        input_rgb_clip = KL.Input(shape=config.RGB_CLIP_SHAPE.tolist(), name="input_rgb_clip")
        input_flow_clip = KL.Input(shape=config.FLOW_CLIP_SHAPE.tolist(), name="input_flow_clip")
        input_image_meta = KL.Input(shape=[None], name="input_image_meta")
        input_labeled_frame_id = KL.Input(shape=[config.TIMESTEPS], name="input_labeled_frame_id", dtype=tf.int32) # 1 frame labeled
        if mode == "training":
            # RPN GT
            input_rpn_match = KL.Input(shape=[None, 1], name="input_rpn_match", dtype=tf.int32)
            input_rpn_bbox = KL.Input(shape=[None, 4], name="input_rpn_bbox", dtype=tf.float32)
            # GT Boxes (zero padded)
            # [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2, actor_class_id, action_class_id)] in image coordinates
            input_gt_boxes = KL.Input(shape=[None, 6], name="input_gt_boxes", dtype=tf.int32)
            # Normalize coordinates
            h, w = K.shape(input_rgb_clip)[2], K.shape(input_rgb_clip)[3]
            image_scale = K.cast(K.stack([h, w, h, w, 1, 1], axis=0), tf.float32)
            gt_boxes = KL.Lambda(lambda x: K.cast(x, tf.float32) / image_scale)(input_gt_boxes)
            # GT Masks (zero padded)
            # [batch, height, width, MAX_GT_INSTANCES]
            if config.USE_MINI_MASK:
                input_gt_masks = KL.Input(
                    shape=[config.MINI_MASK_SHAPE[0], config.MINI_MASK_SHAPE[1], None],
                    name="input_gt_masks", dtype=bool)
            else:
                input_gt_masks = KL.Input(
                    shape=[config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], None],
                    name="input_gt_masks", dtype=bool)

        # TODO: consider using name scope for better tensorboard visualization

        ri3d_end_points = i3d_graph(input_rgb_clip, mode, config, 'rgb') # TODO: mode should be inference
        rC2, rC3, rC4, rC5 = ri3d_end_points['rgb_Conv3d_2c_3x3'], ri3d_end_points['rgb_Mixed_3c'], \
                ri3d_end_points['rgb_Mixed_4f'], ri3d_end_points['rgb_Mixed_5c']
        #shapes:
        #    rC2: (bs, T/2, H/4, W/4, 192)
        #    rC3: (bs, T/2, H/8, W/8, 480)
        #    rC4: (bs, T/4, H/16, W/16, 832)
        #    rC5: (bs, T/8, H/32, W/32, 1024)

        _rP5 = KL.Conv3D(128, (1,1,1), name='fpn_rc5p5')(rC5) # TODO: mask rcnn used 256
        _rP4 = KL.Add(name='fpn_rp4add')([
            KL.UpSampling3D(size=(1,2,2), name='fpn_rp5upsampled')(_rP5),
            KL.Conv3D(128, (1,1,1), name='fpn_rc4p4')(rC4)]) # TODO: shared conv2d layers may be better than these conv3d
        _rP3 = KL.Add(name='fpn_rp3add')([
            KL.UpSampling3D(size=(1,2,2), name='fpn_rp4upsampled')(_rP4),
            KL.Conv3D(128, (1,1,1), name='fpn_rc3p3')(rC3)])
        _rP2 = KL.Add(name='fpn_rp2add')([
            KL.UpSampling3D(size=(1,2,2), name='fpn_rp3upsampled')(_rP3),
            KL.Conv3D(128, (1,1,1), name='fpn_rc2p2')(rC2)])
        # Attach 3x3x3 conv to all P layers to get the final feature maps.
        rP2 = KL.Conv3D(128, (3,3,3), padding='SAME', name='fpn_rp2')(_rP2)
        rP3 = KL.Conv3D(128, (3,3,3), padding='SAME', name='fpn_rp3')(_rP3)
        rP4 = KL.Conv3D(128, (3,3,3), padding='SAME', name='fpn_rp4')(_rP4)
        rP5 = KL.Conv3D(128, (3,3,3), padding='SAME', name='fpn_rp5')(_rP5)

        fi3d_end_points = i3d_graph(input_flow_clip, mode, config, 'flow') # TODO: mode should be inference
        fC2, fC3, fC4, fC5 = fi3d_end_points['flow_Conv3d_2c_3x3'], fi3d_end_points['flow_Mixed_3c'], \
                fi3d_end_points['flow_Mixed_4f'], fi3d_end_points['flow_Mixed_5c']

        _fP5 = KL.Conv3D(128, (1,1,1), name='fpn_fc5p5')(fC5)
        _fP4 = KL.Add(name='fpn_fp4add')([
            KL.UpSampling3D(size=(1,2,2), name='fpn_fp5upsampled')(_fP5),
            KL.Conv3D(128, (1,1,1), name='fpn_fc4p4')(fC4)]) # TODO: shared conv2d layers may be better than these conv3d
        _fP3 = KL.Add(name='fpn_fp3add')([
            KL.UpSampling3D(size=(1,2,2), name='fpn_fp4upsampled')(_fP4),
            KL.Conv3D(128, (1,1,1), name='fpn_fc3p3')(fC3)])
        _fP2 = KL.Add(name='fpn_fp2add')([
            KL.UpSampling3D(size=(1,2,2), name='fpn_fp3upsampled')(_fP3),
            KL.Conv3D(128, (1,1,1), name='fpn_fc2p2')(fC2)])
        # Attach 3x3x3 conv to all P layers to get the final feature maps.
        fP2 = KL.Conv3D(128, (3,3,3), padding='SAME', name='fpn_fp2')(_fP2)
        fP3 = KL.Conv3D(128, (3,3,3), padding='SAME', name='fpn_fp3')(_fP3)
        fP4 = KL.Conv3D(128, (3,3,3), padding='SAME', name='fpn_fp4')(_fP4)
        fP5 = KL.Conv3D(128, (3,3,3), padding='SAME', name='fpn_fp5')(_fP5)

        P5d, P = OrderedDict(), OrderedDict()
        P5d[2] = KL.Concatenate(name='fpn_p2')([rP2, fP2])
        P5d[3] = KL.Concatenate(name='fpn_p3')([rP3, fP3])
        P5d[4] = KL.Concatenate(name='fpn_p4')([rP4, fP4])
        P5d[5] = KL.Concatenate(name='fpn_p5')([rP5, fP5])
        P5d[6] = KL.MaxPooling3D(pool_size=(1,1,1), strides=(1,2,2), name='fpn_p6')(P5d[5])

        # TODO: better way for slicing?
        def expand_inds(inds, h=None, w=None, f=None, dtype=K.dtype(input_rgb_clip)):
            return K.cast(K.tile(K.expand_dims(K.expand_dims(K.expand_dims(inds))),
                                 [1,1,h,w,f]), dtype)
        _scale = {2:4, 3:8, 4:16, 5:32, 6:64}
        for ii in range(2, 7):
            mask = KL.Lambda(expand_inds,
                             arguments={'h':h//_scale[ii], 'w':w//_scale[ii], 'f':256})(input_labeled_frame_id)
            product = KL.Multiply()([P5d[ii], mask])

            P[ii] = KL.Lambda(lambda x: K.sum(x, axis=1, keepdims=False))(product)
                #KL.Multiply([P5d[ii], mask]))

        if config.AGGREGATE == 'mean':
            for ii in range(2, 7):
                P[ii] = KL.Add()([
                    KL.Lambda(lambda x: K.mean(x, axis=1, keepdims=False))(P5d[ii]),
                    P[ii]])

        rpn_feature_maps = [P[2], P[3], P[4], P[5], P[6]]
        mrcnn_feature_maps = [P[2], P[3], P[4], P[5]]

        # Generate Anchors
        self.anchors = matterport_utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                                       config.RPN_ANCHOR_RATIOS,
                                                       config.BACKBONE_SHAPES,
                                                       config.BACKBONE_STRIDES,
                                                       config.RPN_ANCHOR_STRIDE)

        # RPN Model
        rpn = build_rpn_model(config.RPN_ANCHOR_STRIDE,
                              len(config.RPN_ANCHOR_RATIOS), 256) # TODO: 256 in mask rcnn
        # Loop through pyramid layers
        layer_outputs = []  # list of lists
        for p in rpn_feature_maps:
            layer_outputs.append(rpn([p]))
        # Concatenate layer outputs
        # Convert from list of lists of level outputs to list of lists 
        # of outputs across levels. 
        # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"]
        outputs = list(zip(*layer_outputs))
        outputs = [KL.Concatenate(axis=1, name=n)(list(o)) 
                    for o, n in zip(outputs, output_names)]

        rpn_class_logits, rpn_class, rpn_bbox = outputs


        # Generate proposals
        # Proposals are [N, (y1, x1, y2, x2)] in normalized coordinates.
        proposal_count = config.POST_NMS_ROIS_TRAINING if mode == "training"\
                         else config.POST_NMS_ROIS_INFERENCE
        rpn_rois = ProposalLayer(proposal_count=proposal_count,
                                 nms_threshold=0.7,
                                 name="ROI",
                                 anchors=self.anchors,
                                 config=config)([rpn_class, rpn_bbox])


        if mode == "training":
            # Class ID mask to mark class IDs supported by the dataset the image
            # came from.
            _, _, _, active_actor_class_ids, active_action_class_ids \
                    = KL.Lambda(parse_image_meta_graph,
                                mask=[None, None, None, None, None],
                                arguments={'config': config}
                               )(input_image_meta)
            #_, _, _, active_class_ids \
            #        = KL.Lambda(lambda x: parse_image_meta_graph(x),
            #                    mask=[None, None, None, None])(input_image_meta)
            #active_actor_class_ids = active_class_ids[:, :config.NUM_ACTOR_CLASSES]
            #active_action_class_ids = active_class_ids[:, config.NUM_ACTOR_CLASSES:]

            if not config.USE_RPN_ROIS:
                # Ignore predicted ROIs and use ROIs provided as an input.
                input_rois = KL.Input(shape=[config.POST_NMS_ROIS_TRAINING, 4],
                                    name="input_roi", dtype=np.int32)
                # Normalize coordinates to 0-1 range.
                target_rois = KL.Lambda(lambda x: K.cast(x, tf.float32) / image_scale[:4])(input_rois)
            else:
                target_rois = rpn_rois

            # Generate detection targets
            # Subsamples proposals and generates target outputs for training
            # Note that proposals, gt_boxes, and gt_masks might be zero padded
            # Equally, returned rois and targets might be zero padded as well
            
            # TODO: delete tower arg
            rois, target_actor_class_ids, target_action_class_ids, target_bbox, target_mask =\
                DetectionTargetLayer(config, tower='rgb', name="proposal_targets")([
                    target_rois, gt_boxes, input_gt_masks])


            # Network Heads
            # TODO: verify that this handles zero padded ROIs
            mrcnn_actor_class_logits, mrcnn_actor_probs, \
                mrcnn_action_class_logits, mrcnn_action_probs, mrcnn_bbox = \
                fpn_classifier_graph(rois, mrcnn_feature_maps, config.IMAGE_SHAPE,
                                     config.POOL_SIZE, config.NUM_ACTOR_CLASSES, config.NUM_ACTION_CLASSES)

            mrcnn_mask = build_fpn_mask_graph(rois, mrcnn_feature_maps,
                                              config.IMAGE_SHAPE,
                                              config.MASK_POOL_SIZE,
                                              config.NUM_ACTOR_CLASSES)


            # TODO: clean up (use tf.identify if necessary)
            output_rois = KL.Lambda(lambda x: x * 1, name="output_rois")(rois)


            # Losses
            rpn_class_loss = KL.Lambda(lambda x: rpn_class_loss_graph(*x), name="rpn_class_loss")(
                [input_rpn_match, rpn_class_logits])
            rpn_bbox_loss = KL.Lambda(lambda x: rpn_bbox_loss_graph(config, *x), name="rpn_bbox_loss")(
                [input_rpn_bbox, input_rpn_match, rpn_bbox])
            actor_class_loss = KL.Lambda(lambda x: mrcnn_class_loss_graph(*x),
                 name="mrcnn_actor_class_loss")([target_actor_class_ids, mrcnn_actor_class_logits,
                                                 active_actor_class_ids])
            action_class_loss = KL.Lambda(lambda x: mrcnn_class_loss_graph(*x),
                name="mrcnn_action_class_loss")([target_action_class_ids, mrcnn_action_class_logits,
                                                 active_action_class_ids])
            bbox_loss = KL.Lambda(lambda x: mrcnn_bbox_loss_graph(*x), name="mrcnn_bbox_loss")(
                [target_bbox, target_actor_class_ids, mrcnn_bbox])
            mask_loss = KL.Lambda(lambda x: mrcnn_mask_loss_graph(*x), name="mrcnn_mask_loss")(
                [target_mask, target_actor_class_ids, mrcnn_mask])

            # Model
            inputs = [input_rgb_clip, input_flow_clip, input_image_meta, input_labeled_frame_id,
                    input_rpn_match, input_rpn_bbox, input_gt_boxes, input_gt_masks]
            if not config.USE_RPN_ROIS:
                inputs.append(input_rois)
            outputs = [rpn_class_logits, rpn_class, rpn_bbox,
                    mrcnn_actor_class_logits, mrcnn_actor_probs, mrcnn_action_class_logits, mrcnn_action_probs,
                    mrcnn_bbox, mrcnn_mask,
                    rpn_rois, output_rois,
                    rpn_class_loss, rpn_bbox_loss, actor_class_loss, action_class_loss, bbox_loss, mask_loss]
            model = KM.Model(inputs, outputs, name='mask_rcnn')
        else:
            # Network Heads
            # Proposal classifier and BBox regressor heads
            mrcnn_actor_class_logits, mrcnn_actor_class, \
                mrcnn_action_class_logits, mrcnn_action_class, mrcnn_bbox =\
                fpn_classifier_graph(rpn_rois, mrcnn_feature_maps, config.IMAGE_SHAPE,
                                     config.POOL_SIZE, config.NUM_ACTOR_CLASSES, config.NUM_ACTION_CLASSES)

            # Detections
            # output is [batch, num_detections, (y1, x1, y2, x2, actor_class_id, action_class_id, score)]
            # in image coordinates
            detections = DetectionLayer(config, name="mrcnn_detection")(
                [rpn_rois, mrcnn_actor_class, mrcnn_action_class, mrcnn_bbox, input_image_meta])

            # Convert boxes to normalized coordinates
            # TODO: let DetectionLayer return normalized coordinates to avoid
            #       unnecessary conversions
            h, w = config.IMAGE_SHAPE[:2]
            detection_boxes = KL.Lambda(lambda x: x[...,:4]/np.array([h, w, h, w]))(detections)

            # Create masks for detections
            mrcnn_mask = build_fpn_mask_graph(detection_boxes, mrcnn_feature_maps,
                                              config.IMAGE_SHAPE,
                                              config.MASK_POOL_SIZE,
                                              config.NUM_ACTOR_CLASSES)

            model = KM.Model([input_rgb_clip, input_flow_clip, input_image_meta, input_labeled_frame_id],
                        [detections, mrcnn_actor_class, mrcnn_action_class, mrcnn_bbox, mrcnn_mask,
                         rpn_rois, rpn_class, rpn_bbox],
                        name='mask_rcnn')

        # Add multi-GPU support.
        if config.GPU_COUNT > 1:
            model = ParallelModel(model, config.GPU_COUNT)

        return model


    def set_log_dir(self, model_path=None):
        """Sets the model log directory and epoch counter.

        model_path: If None, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise,
            extract the log directory and the epoch counter from the file
            name.
        """
        # Set date and epoch counter as if starting a new model
        self.epoch = 0
        now = datetime.datetime.now()

        # If we have a model path with date and epochs use them
        if model_path:
            # Continue from we left of. Get epoch and date from the file name
            # A sample model path might look like:
            # /path/to/logs/a2d20171029T2315/padnet_a2d_0001.h5
            regex = r".*/\w+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})/padnet\_\w+(\d{4})\.h5"
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                        int(m.group(4)), int(m.group(5)))
                self.epoch = int(m.group(6)) + 1

        # Directory for training logs
        self.log_dir = os.path.join(self.model_dir, "{}{:%Y%m%dT%H%M}".format(
            self.config.NAME.lower(), now))

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(self.log_dir, "padnet_{}_*epoch*.h5".format(
            self.config.NAME.lower()))
        self.checkpoint_path = self.checkpoint_path.replace("*epoch*", "{epoch:04d}")

    def find_last(self):
        """Finds the last checkpoint file of the last trained model in the
        model directory.
        Returns:
            log_dir: The directory where events and weights are saved
            checkpoint_path: the path to the last checkpoint file
        """
        # Get directory names. Each directory corresponds to a model
        dir_names = next(os.walk(self.model_dir))[1]
        key = self.config.NAME.lower()
        dir_names = filter(lambda f: f.startswith(key), dir_names)
        dir_names = sorted(dir_names)
        if not dir_names:
            return None, None
        # Pick last directory
        dir_name = os.path.join(self.model_dir, dir_names[-1])
        # Find the last checkpoint
        checkpoints = next(os.walk(dir_name))[2]
        checkpoints = filter(lambda f: f.startswith("padnet"), checkpoints)
        checkpoints = sorted(checkpoints)
        if not checkpoints:
            return dir_name, None
        checkpoint = os.path.join(dir_name, checkpoints[-1])
        return dir_name, checkpoint

    def load_weights(self, filepath, by_name=False, exclude=None):
        """Modified version of the correspoding Keras function with
        the addition of multi-GPU support and the ability to exclude
        some layers from loading.
        exlude: list of layer names to excluce
        """
        import h5py
        from keras.engine import topology

        if exclude:
            by_name = True

        if h5py is None:
            raise ImportError('`load_weights` requires h5py.')
        f = h5py.File(filepath, mode='r')
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        keras_model = self.keras_model
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
            else keras_model.layers

        # Exclude some layers
        if exclude:
            layers = filter(lambda l: l.name not in exclude, layers)
        
        if by_name:
            topology.load_weights_from_hdf5_group_by_name(f, layers)
        else:
            topology.load_weights_from_hdf5_group(f, layers)
        if hasattr(f, 'close'):
            f.close()

        # Update the log directory
        self.set_log_dir(filepath)

if __name__ == '__main__':
    sys.path.append('..')
    from cfg.config import Config
    class TestConfig(Config):
        def __init__(self):
            super(TestConfig, self).__init__()
            self.NAME='Test'
            self.BATCH_SIZE = 8
            self.TIMESTEPS = 16
            self.IMAGE_H = 256
            self.IMAGE_W = 256
            self.IMAGE_SHAPE = np.array([self.IMAGE_H, self.IMAGE_W, 3])
            self.RGB_CLIP_SHAPE = np.array([self.TIMESTEPS, self.IMAGE_H, self.IMAGE_W, 3])
            self.FLOW_CLIP_SHAPE = np.array([self.TIMESTEPS, self.IMAGE_H, self.IMAGE_W, 2])
            self.AGGREGATE = 'mean'
            self.NUM_ACTOR_CLASSES = 1 + 7 # background + 7 actors
            self.NUM_ACTION_CLASSES = 9
            self.I3D_NUM_CLASSES = 400
            self.I3D_DROPOUT_KEEP_PROB = 1.0
            self.I3D_SPATIAL_SQUEEZE = True
            self.GPU_COUNT = 4
    padnet_train = PADNet('training', TestConfig(), '../outputs/tmp')
    padnet_test = PADNet('inference', TestConfig(), '../outputs/tmp')

