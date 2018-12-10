import os, sys, glob, random, math, datetime, itertools, json, re, logging, numpy as np,\
    scipy.misc, tensorflow as tf
import keras, keras.backend as K, keras.layers as KL, keras.initializers as KI, \
    keras.engine as KE, keras.models as KM
from keras.layers import TimeDistributed as TD
from collections import OrderedDict

sys.path.append('..')
import utils.matterport_utils as matterport_utils
from models.model_misc import *
from models.data_generator import data_generator
from models.resnet_3d import resnet_3d_graph
from models.rpn import build_rpn_model
from models.proposal_layer import ProposalLayer
from models.detection_layer_v2_0 import DetectionLayer # TODO: or models.detection_layer_v2_0
from models.detection_target_layer import DetectionTargetLayer
from models.fpn_v2_0 import fpn_classifier_graph, build_fpn_mask_graph, \
    fpn_action_classifier_graph
from models.loss import rpn_class_loss_graph, rpn_bbox_loss_graph, mrcnn_class_loss_graph, \
    mrcnn_bbox_loss_graph, mrcnn_mask_loss_graph
from models.parallel_model import ParallelModel
from distutils.version import LooseVersion
from pdb import set_trace

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
    """
    Version 2.1

    Inputs are rgb and flow clips. Three top-down models: one for single frame features, one another
    3d model for aggregating temporal information in rgb clip, and one 3d model in flow clip.

    Encapsulates the PAD Net model functionality.
    The actual Keras model is in the keras_model property.
    """

    def __init__(self, mode, config, model_dir):
        """
        mode: Either "training" or "inference"
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        assert mode in ['training', 'inference']
        self.version = LooseVersion('2.1')
        self.mode = mode
        self.config = config
        self.model_dir = model_dir
        self.set_log_dir()
        self.keras_model = self.build(mode=mode, config=config)

    def set_trainable(self, layer_regex, keras_model=None, indent=0, verbose=1, exclude=''):
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
            # Is the layer excluded?
            if bool(re.fullmatch(exclude, layer.name)):
                continue

            # Is the layer a model?
            if layer.__class__.__name__ == 'Model':
                print("In model: ", layer.name)
                self.set_trainable(layer_regex, keras_model=layer, indent=indent+4, exclude=exclude)
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
        print('Compiling the following losses:')
        print(self.config.LOSSES)
        loss_names = self.config.LOSSES
        #loss_names = ['r_rpn_class_loss', 'f_rpn_class_loss', 'r_rpn_bbox_loss', 'f_rpn_bbox_loss',
        #              'r_actor_class_loss', 'f_actor_class_loss', 'actor_class_loss',
        #              'r_action_class_loss', 'f_action_class_loss', 'action_class_loss',
        #              'mrcnn_bbox_loss', 'mrcnn_mask_loss'
        #             ]
        for name in loss_names:
            layer = self.keras_model.get_layer(name)
            if layer.output in self.keras_model.losses:
                continue
            self.keras_model.add_loss(tf.reduce_mean(layer.output, keep_dims=True))

        # Add L2 Regularization if wanted
        if self.config.REGULARIZATION:
            print('adding L2 regularization...')
            reg_losses = [keras.regularizers.l2(self.config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
                          for w in self.keras_model.trainable_weights
                          if 'gamma' not in w.name and 'beta' not in w.name]
            self.keras_model.add_loss(tf.add_n(reg_losses))
        else:
            print('No L2 regularization applied.')

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

    def train(self, train_dataset, val_dataset, learning_rate, epochs, layers, exclude=''):
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
              heads: The RPN, classifier and mask heads of the network
              all: All the layers
              3+: Train Resnet stage 3 and up
              4+: Train Resnet stage 4 and up
              5+: Train Resnet stage 5 and up
        """
        assert self.mode == "training", "Create model in training mode."

        # Pre-defined layer regular expressions
        layer_regex = { # TODO: modify the regular expressions
            # all layers but the backbone
            "heads": r"(.*mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "temporal+heads": r"(t.*)|(.*mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)|(.*action\_.*)",
            "3dtd+temporal+heads-rpn": r"(3d\_fpn\_.*)|(t.*)|(.*mrcnn\_.*)|(.*action\_.*)",
            "3dtd+temporal+heads": r"(3d\_fpn\_.*)|(t.*)|(.*mrcnn\_.*)|(.*action\_.*)|(rpn\_.*)",
            "resnet+3dtd+temporal+action": r"(res.*)|(3d\_fpn\_.*)|(t.*)|(.*action\_.*)",
            # From Resnet stage 4 layers and up
            #"3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            #"4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            #"5+": r"(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # All layers
            "topdown": r"fpn\_.*",
            "rpn": r"rpn\_.*",
            "rpn+topdown": r"(fpn\_.*)|(rpn\_.*)",
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
        class DebugCallback(keras.callbacks.Callback):
            def on_batch_begin(self, batch, logs={}):
                from pdb import set_trace; set_trace()

            def on_batch_end(self, batch, logs={}):
                from pdb import set_trace; set_trace()

        callbacks = [
            keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                        histogram_freq=0, write_graph=True, write_images=False),
            keras.callbacks.ModelCheckpoint(self.checkpoint_path,
                                            verbose=0, save_weights_only=True),
            #DebugCallback(),
        ]

        # Common parameters to pass to fit_generator()
        fit_kwargs = {
            "steps_per_epoch": self.config.STEPS_PER_EPOCH,
            "callbacks": callbacks,
            "validation_data": val_generator,
            "validation_steps": self.config.VALIDATION_STPES,
            "max_queue_size": self.config.MAX_QUEUE_SIZE,
            "workers": self.config.WORKERS, #max(self.config.BATCH_SIZE // 2, 2), # TODO: may want more than 2
            "use_multiprocessing": self.config.USE_MULTIPROCESSING,
        }

        # Train
        print("\nStarting at epoch {}. LR={}\n".format(self.epoch, learning_rate))
        print("Checkpoint Path: {}".format(self.checkpoint_path))
        self.set_trainable(layers, exclude=exclude)
        self.compile(learning_rate, self.config.LEARNING_MOMENTUM)

        self.keras_model.fit_generator(
            train_generator,
            initial_epoch=self.epoch,
            epochs=epochs,
            **fit_kwargs
            )
        self.epoch = max(self.epoch, epochs)

    def build_top_down_model(self, mode, config, tower, temporal):
        assert tower in ['rgb', 'flow']
        dim = '3d_' if temporal else ''
        # Top-down Layers
        T, H, W = config.TIMESTEPS, config.IMAGE_H, config.IMAGE_W
        C2 = KL.Input(shape=(T, H//4, W//4, 256))
        C3 = KL.Input(shape=(T, H//8, W//8, 512))
        C4 = KL.Input(shape=(T, H//16, W//16, 1024))
        C5 = KL.Input(shape=(T, H//32, W//32, 2048))
        P5 = TD(KL.Conv2D(256, (1, 1)), name=dim+'fpn_c5p5')(C5)
        P4 = KL.Add(name="fpn_p4add")([
            TD(KL.UpSampling2D(size=(2, 2)), name=dim+"fpn_p5upsampled")(P5),
            TD(KL.Conv2D(256, (1, 1)), name=dim+'fpn_c4p4')(C4)])
        P3 = KL.Add(name="fpn_p3add")([
            TD(KL.UpSampling2D(size=(2, 2)), name=dim+"fpn_p4upsampled")(P4),
            TD(KL.Conv2D(256, (1, 1)), name=dim+'fpn_c3p3')(C3)])
        P2 = KL.Add(name="fpn_p2add")([
            TD(KL.UpSampling2D(size=(2, 2)), name=dim+"fpn_p3upsampled")(P3),
            TD(KL.Conv2D(256, (1, 1)), name=dim+'fpn_c2p2')(C2)])
        # Attach 3x3 conv to all P layers to get the final feature maps.
        P2 = TD(KL.Conv2D(256, (3, 3), padding="SAME"), name=dim+"fpn_p2")(P2)
        P3 = TD(KL.Conv2D(256, (3, 3), padding="SAME"), name=dim+"fpn_p3")(P3)
        P4 = TD(KL.Conv2D(256, (3, 3), padding="SAME"), name=dim+"fpn_p4")(P4)
        P5 = TD(KL.Conv2D(256, (3, 3), padding="SAME"), name=dim+"fpn_p5")(P5)
        # P6 is used for the 5th anchor scale in RPN. Generated by
        # subsampling from P5 with stride of 2.
        P6 = TD(KL.MaxPooling2D(pool_size=(1, 1), strides=2), name=dim+"fpn_p6")(P5)
        #if temporal:
        #    P2 = KL.Conv3D(256, (3,1,1), padding='same', name='tfpn_p2')(P2)
        #    P3 = KL.Conv3D(256, (3,1,1), padding='same', name='tfpn_p3')(P3)
        #    P4 = KL.Conv3D(256, (3,1,1), padding='same', name='tfpn_p4')(P4)
        #    P5 = KL.Conv3D(256, (3,1,1), padding='same', name='tfpn_p5')(P5)
        return KM.Model([C2, C3, C4, C5], [P2, P3, P4, P5, P6], name=tower+'_'+dim+'top_down')


    def expand_inds(self, inds, h=None, w=None, f=None, dtype=None):
        """ Helper function for `self.slice_one_frame`. Expand inds into 5D mask tensor. """
        return K.cast(K.tile(K.expand_dims(K.expand_dims(K.expand_dims(inds))), [1,1,h,w,f]), dtype)


    def slice_one_frame(self, clip_feature, input_labeled_frame_id, h, w, scale, channel):
        """ Slice one frame's feature out of a clip's feature.
            The frame's id is included in `input_labeled_frame_id`.

            clip_features: [batch, timesteps, h//scale, w//scale, channel], feature map of a clip.
            input_labeled_frame_id: [batch, timesteps]. Each row is a one-hot vector indicating the
            frame's location.
            h: image height.
            w: image width.
            scale: the scale factor to spatial dimensions. Should match with `clip_feature`'s shape.
            channel: number of channels of clip_feature.
        """
        arguments={'h':h//scale, 'w':w//scale, 'f':channel, 'dtype':clip_feature.dtype}
        mask = KL.Lambda(self.expand_inds, arguments=arguments)(input_labeled_frame_id)
        product = KL.Multiply()([clip_feature, mask])
        return KL.Lambda(lambda x: K.sum(x, axis=1, keepdims=False))(product)


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
        # input_flow_xym_clip's last dim is 3: x flow, y flow, magnitude
        input_flow_xym_clip = KL.Input(shape=config.RGB_CLIP_SHAPE.tolist(), name='input_flow_xym_clip')
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

        r_outputs = resnet_3d_graph(input_rgb_clip, mode, config.RESNET, stage5=True, temporal=False)
        self.rgb_resnet_3d_model = KM.Model([input_rgb_clip], r_outputs, name='rgb_resnet_3d')
        f_outputs = resnet_3d_graph(input_flow_xym_clip, mode, config.RESNET, stage5=True, temporal=False)
        self.flow_resnet_3d_model = KM.Model([input_flow_xym_clip], f_outputs, name='flow_resnet_3d')

        _, rC2, rC3, rC4, rC5 = self.rgb_resnet_3d_model([input_rgb_clip])
        flow_xym_clip = flow_xy_to_xym(input_flow_clip)
        _, fC2, fC3, fC4, fC5 = self.flow_resnet_3d_model([flow_xym_clip])


        # RGB 2D Top Down
        self.rgb_top_down_model = self.build_top_down_model(mode, config, 'rgb', temporal=False)
        rP_clip = {}
        rP_clip[2], rP_clip[3], rP_clip[4], rP_clip[5], rP_clip[6] = self.rgb_top_down_model([rC2, rC3, rC4, rC5])


        # RGB 3D Top Down
        self.rgb_3d_top_down_model = self.build_top_down_model(mode, config, 'rgb', temporal=True) # not doing temporal inside
        trP_clip = {}
        trP_clip[2], trP_clip[3], trP_clip[4], trP_clip[5], trP_clip[6] = self.rgb_3d_top_down_model([rC2, rC3, rC4, rC5])
        trP_clip[2] = KL.Conv3D(256, (3,1,1), padding='same', name='tfpn_p2')(trP_clip[2])
        trP_clip[3] = KL.Conv3D(256, (3,1,1), padding='same', name='tfpn_p3')(trP_clip[3])
        trP_clip[4] = KL.Conv3D(256, (3,1,1), padding='same', name='tfpn_p4')(trP_clip[4])
        trP_clip[5] = KL.Conv3D(256, (3,1,1), padding='same', name='tfpn_p5')(trP_clip[5])


        # Flow 3D Top Down
        self.flow_3d_top_down_model = self.build_top_down_model(mode, config, 'flow', temporal=True)
        tfP_clip = {}
        tfP_clip[2], tfP_clip[3], tfP_clip[4], tfP_clip[5], tfP_clip[6] = self.flow_3d_top_down_model([fC2, fC3, fC4, fC5])
        tfP_clip[2] = KL.Conv3D(256, (3,1,1), padding='same', name='t_f_fpn_p2')(tfP_clip[2])
        tfP_clip[3] = KL.Conv3D(256, (3,1,1), padding='same', name='t_f_fpn_p3')(tfP_clip[3])
        tfP_clip[4] = KL.Conv3D(256, (3,1,1), padding='same', name='t_f_fpn_p4')(tfP_clip[4])
        tfP_clip[5] = KL.Conv3D(256, (3,1,1), padding='same', name='t_f_fpn_p5')(tfP_clip[5])

        # slice the single frame feature maps for rpn and mrcnn heads
        _scale = {2:4, 3:8, 4:16, 5:32, 6:64}
        rP = {}
        for ii in range(2, 6+1):
            rP[ii] = self.slice_one_frame(rP_clip[ii], input_labeled_frame_id, h, w, _scale[ii], 256)

        r_rpn_feature_maps = [rP[2], rP[3], rP[4], rP[5], rP[6]]
        r_mrcnn_feature_maps = [rP[2], rP[3], rP[4], rP[5]]


        # slice the temporal feature maps for action recognition
        trP = {}
        for ii in range(2, 5+1):
            trP[ii] = self.slice_one_frame(trP_clip[ii], input_labeled_frame_id, h, w, _scale[ii], 256)

        r_action_feature_maps = [trP[2], trP[3], trP[4], trP[5]]

        tfP = {}
        for ii in range(2, 5+1):
            tfP[ii] = self.slice_one_frame(tfP_clip[ii], input_labeled_frame_id, h, w, _scale[ii], 256)

        f_action_feature_maps = [tfP[2], tfP[3], tfP[4], tfP[5]]

        tP = {}
        for ii in range(2, 5+1):
            tP[ii] = KL.Concatenate(axis=-1, name='action_feature_maps_'+str(ii))(
                [trP[ii], tfP[ii]])

        action_feature_maps = [tP[2], tP[3], tP[4], tP[5]]


        # Generate Anchors
        self.anchors = matterport_utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                                       config.RPN_ANCHOR_RATIOS,
                                                       config.BACKBONE_SHAPES,
                                                       config.BACKBONE_STRIDES,
                                                       config.RPN_ANCHOR_STRIDE)

        # RPN Model
        self.r_rpn = build_rpn_model(config.RPN_ANCHOR_STRIDE,
                                     len(config.RPN_ANCHOR_RATIOS), 256, name='rgb_rpn_model')

        # Loop through pyramid layers
        layer_outputs = []  # list of lists
        for p in r_rpn_feature_maps:
            layer_outputs.append(self.r_rpn([p]))
        # Concatenate layer outputs
        # Convert from list of lists of level outputs to list of lists 
        # of outputs across levels. 
        # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        output_names = ["rgb_rpn_class_logits", "rgb_rpn_class", "rgb_rpn_bbox"]
        outputs = list(zip(*layer_outputs))
        outputs = [KL.Concatenate(axis=1, name=n)(list(o)) 
                    for o, n in zip(outputs, output_names)]

        r_rpn_class_logits, r_rpn_class, r_rpn_bbox = outputs


        # Generate proposals
        # Proposals are [N, (y1, x1, y2, x2)] in normalized coordinates.
        proposal_count = config.POST_NMS_ROIS_TRAINING if mode == "training"\
                         else config.POST_NMS_ROIS_INFERENCE
        r_rpn_rois = ProposalLayer(proposal_count=proposal_count,
                                   nms_threshold=0.7,
                                   name="rgb_ROI",
                                   anchors=self.anchors,
                                   config=config)([r_rpn_class, r_rpn_bbox])


        if mode == "training":
            # Class ID mask to mark class IDs supported by the dataset the image
            # came from.
            _, _, _, active_actor_class_ids, active_action_class_ids \
                    = KL.Lambda(parse_image_meta_graph,
                                mask=[None, None, None, None, None],
                                arguments={'config': config}
                               )(input_image_meta)

            if not config.USE_RPN_ROIS: # TODO: not implemented yet
                # Ignore predicted ROIs and use ROIs provided as an input.
                input_rois = KL.Input(shape=[config.POST_NMS_ROIS_TRAINING, 4],
                                    name="input_roi", dtype=np.int32)
                # Normalize coordinates to 0-1 range.
                target_rois = KL.Lambda(lambda x: K.cast(x, tf.float32) / image_scale[:4])(input_rois)
            else:
                r_target_rois = r_rpn_rois

            # Generate detection targets
            # Subsamples proposals and generates target outputs for training
            # Note that proposals, gt_boxes, and gt_masks might be zero padded
            # Equally, returned rois and targets might be zero padded as well
            r_rois, r_target_actor_class_ids, r_target_action_class_ids, r_target_bbox, r_target_mask =\
                DetectionTargetLayer(config, tower='rgb', name="rgb_proposal_targets")([
                    r_target_rois, gt_boxes, input_gt_masks])

            target_actor_class_ids = r_target_actor_class_ids
            target_action_class_ids = r_target_action_class_ids
            target_bbox = r_target_bbox
            target_mask = r_target_mask


            # Network Heads
            # Actor
            mrcnn_actor_class_logits, mrcnn_actor_probs, mrcnn_bbox = \
                fpn_classifier_graph(r_rois, r_mrcnn_feature_maps,
                                     config.IMAGE_SHAPE, config.POOL_SIZE, config.NUM_ACTOR_CLASSES,
                                     tower = 'rgb',
                                     use_dropout = config.USE_DROPOUT,
                                    )
            r_mrcnn_actor_class_logits = mrcnn_actor_class_logits

            # Mask
            r_mrcnn_mask = build_fpn_mask_graph(r_rois, r_mrcnn_feature_maps,
                                                config.IMAGE_SHAPE,
                                                config.MASK_POOL_SIZE,
                                                config.NUM_ACTOR_CLASSES, 'rgb')
            mrcnn_mask = r_mrcnn_mask

            # Action
            action_class_logits, action_probs = \
                fpn_action_classifier_graph(r_rois, action_feature_maps,
                                            config.IMAGE_SHAPE, config.POOL_SIZE, config.NUM_ACTION_CLASSES,
                                            tower = 'rgb',
                                            use_dropout = config.USE_DROPOUT,
                                           )


            # TODO: clean up (use tf.identify if necessary)
            r_output_rois = KL.Lambda(lambda x: x * 1, name="r_output_rois")(r_rois)


            # Losses
            # TODO: verify `input_rpn_match` and `input_rpn_bbox` could be shared in two towers.
            r_rpn_class_loss = KL.Lambda(lambda x: rpn_class_loss_graph(*x), name="r_rpn_class_loss")(
                [input_rpn_match, r_rpn_class_logits])
            r_rpn_bbox_loss = KL.Lambda(lambda x: rpn_bbox_loss_graph(config, *x), name="r_rpn_bbox_loss")(
                [input_rpn_bbox, input_rpn_match, r_rpn_bbox])

            r_actor_class_loss = KL.Lambda(lambda x: mrcnn_class_loss_graph(*x),
                 name="r_actor_class_loss")([r_target_actor_class_ids, r_mrcnn_actor_class_logits,
                                                 active_actor_class_ids])

            action_class_loss = KL.Lambda(lambda x: mrcnn_class_loss_graph(*x),
                name="action_class_loss")([r_target_action_class_ids, action_class_logits,
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
            outputs = [r_rpn_class_logits, r_rpn_class, r_rpn_bbox,
                       mrcnn_actor_class_logits, mrcnn_actor_probs, action_class_logits, action_probs,
                       mrcnn_bbox, mrcnn_mask,
                       r_rpn_rois,
                       r_output_rois,
                       r_rpn_class_loss,
                       r_rpn_bbox_loss,
                       r_actor_class_loss,
                       action_class_loss,
                       bbox_loss, mask_loss,
                       r_target_action_class_ids,
                      ]
            model = KM.Model(inputs, outputs, name='padnet_v2.0')
        else: # TODO: not modified yet
            # Network Heads
            # Proposal classifier and BBox regressor heads
            mrcnn_actor_class_logits, mrcnn_actor_class, mrcnn_bbox = \
                fpn_classifier_graph(r_rpn_rois, r_mrcnn_feature_maps,
                                     config.IMAGE_SHAPE,
                                     config.POOL_SIZE,
                                     config.NUM_ACTOR_CLASSES,
                                     tower='rgb',
                                     use_dropout = config.USE_DROPOUT
                                    )
            action_class_logits, action_class = \
                fpn_action_classifier_graph(r_rpn_rois, action_feature_maps,
                                            config.IMAGE_SHAPE,
                                            config.POOL_SIZE,
                                            config.NUM_ACTION_CLASSES,
                                            tower='rgb',
                                            use_dropout = config.USE_DROPOUT
                                           )


            # Detections
            # output is [batch, num_detections, (y1, x1, y2, x2, actor_class_id, action_class_id, actor_score, action_score)]
            # in image coordinates
            rpn_rois = r_rpn_rois
            detections, keep_idx = DetectionLayer(config, name="mrcnn_detection")(
                [rpn_rois, mrcnn_actor_class, action_class, mrcnn_bbox, input_image_meta])
            # TODO: detections, keep_idx

            # Convert boxes to normalized coordinates
            # TODO: let DetectionLayer return normalized coordinates to avoid
            #       unnecessary conversions
            h, w = config.IMAGE_SHAPE[:2]
            detection_boxes = KL.Lambda(lambda x: x[...,:4]/np.array([h, w, h, w]))(detections)

            # Create masks for detections
            # Only using rgb feature maps for mask prediction
            mrcnn_mask = build_fpn_mask_graph(detection_boxes, r_mrcnn_feature_maps,
                                              config.IMAGE_SHAPE,
                                              config.MASK_POOL_SIZE,
                                              config.NUM_ACTOR_CLASSES, 'rgb')

            # Model
            inputs = [input_rgb_clip, input_flow_clip, input_image_meta, input_labeled_frame_id]
            outputs = [detections, mrcnn_actor_class, action_class,
                       mrcnn_bbox, mrcnn_mask,
                       r_rpn_rois,
                       r_rpn_class,
                       r_rpn_bbox, keep_idx]
            model = KM.Model(inputs, outputs, name='padnet_v2.0')

        # Add multi-GPU support.
        if config.GPU_COUNT > 1:
            model = ParallelModel(model, config.GPU_COUNT)

        return model

    def detect(self, rgb_clip, flow_clip, labeled_frame_id, image_meta):
        """Runs the detection pipeline.

        images: List of images, potentially of different sizes.

        Returns a list of dicts, one dict per image. The dict contains:
        rois: [N, (y1, x1, y2, x2)] detection bounding boxes
        class_ids: [N] int class IDs
        scores: [N] float probability scores for the class IDs
        masks: [H, W, N] instance binary masks
        """
        assert self.mode == "inference", "Create model in inference mode."

        #rgb_clip = np.expand_dims(rgb_clip, 0)
        #flow_clip = np.expand_dims(flow_clip, 0)
        #labeled_frame_id = np.expand_dims(labeled_frame_id, 0)
        #image_meta = np.expand_dims(image_meta, 0)
        #print(rgb_clip.shape, flow_clip.shape, labeled_frame_id.shape, image_meta.shape)
        # Run object detection
        detections, mrcnn_actor_fullscores, mrcnn_action_fullscores, mrcnn_bbox, mrcnn_mask, \
        r_rpn_rois, r_rpn_class, r_rpn_bbox, keep_idx = \
            self.keras_model.predict([rgb_clip, flow_clip, image_meta, labeled_frame_id], verbose=0)

        mrcnn_actor_fullscores = mrcnn_actor_fullscores[:,keep_idx[0],:] # TODO: check
        mrcnn_action_fullscores = mrcnn_action_fullscores[:,keep_idx[0],:] # TODO: check

        image_shape = image_meta[0,1:4]
        window = image_meta[0,4:8]
        final_rois, final_actor_class_ids, final_action_class_ids, \
        final_actor_scores, final_action_scores, final_masks, final_actor_fullscores, final_action_fullscores = \
            self.unmold_detections(detections[0], mrcnn_mask[0], mrcnn_actor_fullscores[0], mrcnn_action_fullscores[0], image_shape, window)

        result = {
            "rois": final_rois,
            "actor_class_ids": final_actor_class_ids,
            "action_class_ids": final_action_class_ids,
            "actor_scores": final_actor_scores,
            "action_scores": final_action_scores,
            "masks": final_masks,
            "actor_fullscores": final_actor_fullscores,
            "action_fullscores": final_action_fullscores,
        }
        return result

    def unmold_detections(self, detections, mrcnn_mask, mrcnn_actor_fullscores, mrcnn_action_fullscores, image_shape, window):
        """Reformats the detections of one image from the format of the neural
        network output to a format suitable for use in the rest of the
        application.

        detections: [N, (y1, x1, y2, x2, class_id, score)]
        mrcnn_mask: [N, height, width, num_classes]
        image_shape: [height, width, depth] Original size of the image before resizing
        window: [y1, x1, y2, x2] Box in the image where the real image is
                excluding the padding.
        
        Returns:
        boxes: [N, (y1, x1, y2, x2)] Bounding boxes in pixels
        class_ids: [N] Integer class IDs for each bounding box
        scores: [N] Float probability scores of the class_id
        masks: [height, width, num_instances] Instance masks
        """
        # How many detections do we have?
        # Detections array is padded with zeros. Find the first class_id == 0.
        zero_ix = np.where(detections[:,4] == 0)[0]
        N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]
        
        # Extract boxes, class_ids, scores, and class-specific masks
        boxes = detections[:N, :4]
        actor_class_ids = detections[:N, 4].astype(np.int32)
        action_class_ids = detections[:N, 5].astype(np.int32)
        actor_scores = detections[:N, 6]
        action_scores = detections[:N, 7]
        masks = mrcnn_mask[np.arange(N), :, :, actor_class_ids]
        actor_fullscores = mrcnn_actor_fullscores[:N]
        action_fullscores = mrcnn_action_fullscores[:N]

        # Filter out detections with zero area. Often only happens in early
        # stages of training when the network weights are still a bit random.
        exclude_ix = np.where((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
        if exclude_ix.shape[0] > 0:
            boxes = np.delete(boxes, exclude_ix, axis=0)
            actor_class_ids = np.delete(actor_class_ids, exclude_ix, axis=0)
            action_class_ids = np.delete(action_class_ids, exclude_ix, axis=0)
            actor_scores = np.delete(actor_scores, exclude_ix, axis=0)
            action_scores = np.delete(action_scores, exclude_ix, axis=0)
            masks = np.delete(masks, exclude_ix, axis=0)
            actor_fullscores = np.delete(actor_fullscores, exclude_ix, axis=0)
            action_fullscores = np.delete(action_fullscores, exclude_ix, axis=0)
            N = actor_class_ids.shape[0]

        # Compute scale and shift to translate coordinates to image domain.
        h_scale = image_shape[0] / (window[2] - window[0])
        w_scale = image_shape[1] / (window[3] - window[1])
        scale = min(h_scale, w_scale)
        shift = window[:2]  # y, x
        scales = np.array([scale, scale, scale, scale])
        shifts = np.array([shift[0], shift[1], shift[0], shift[1]])
        
        # Translate bounding boxes to image domain
        boxes = np.multiply(boxes - shifts, scales).astype(np.int32)
        
        # Resize masks to original image size and set boundary threshold.
        full_masks = []
        for i in range(N):
            # Convert neural network mask to full size mask
            full_mask = matterport_utils.unmold_mask(masks[i], boxes[i], image_shape)
            full_masks.append(full_mask)
        full_masks = np.stack(full_masks, axis=-1)\
                    if full_masks else np.empty((0,) + masks.shape[1:3])
        
        return boxes, actor_class_ids, action_class_ids, actor_scores, action_scores, full_masks, actor_fullscores, action_fullscores


        # Process detections
        results = []
        for i, image in enumerate(images):
            final_rois, final_class_ids, final_scores, final_masks =\
                self.unmold_detections(detections[i], mrcnn_mask[i],
                                       image.shape, windows[i])
            results.append({
                "rois": final_rois,
                "class_ids": final_class_ids,
                "scores": final_scores,
                "masks": final_masks,
            })
        return results


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

    def load_weights(self, filepath, by_name=False, exclude=None, verbose=False, verboseverbose=False):
        """Modified version of the correspoding Keras function with
        the addition of multi-GPU support and the ability to exclude
        some layers from loading.
        exlude: list of layer names to excluce
        """
        import h5py
        from keras.engine import topology
        from utils.misc import load_weights_from_hdf5_group_by_name

        if exclude:
            by_name = True

        if h5py is None:
            raise ImportError('`load_weights` requires h5py.')
        with h5py.File(filepath, mode='r') as f:
            # This function only handles f with 'layer_names'
            #if 'layer_names' not in f.attrs and 'model_weights' in f:
            #    f = f['model_weights']

            # In multi-GPU training, we wrap the model. Get layers
            # of the inner model because they have the weights.
            keras_model = self.keras_model
            layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
                else keras_model.layers

            # Exclude some layers
            if exclude:
                layers = filter(lambda l: l.name not in exclude, layers)

            if by_name:
                # use our own loading function in utils.misc
                load_weights_from_hdf5_group_by_name(f, layers, verbose=verbose, verboseverbose=verboseverbose)
            else:
                topology.load_weights_from_hdf5_group(f, layers)

        # Update the log directory
        self.set_log_dir(filepath)

if __name__ == '__main__':
    sys.path.append('..')
    from cfg.config import Config
    class TestConfig(Config):
        def __init__(self):
            super(TestConfig, self).__init__()
            self.NAME='Test'
            self.IMAGES_PER_GPU = 4
            self.TIMESTEPS = 5
            self.IMAGE_H = 256
            self.IMAGE_W = 256
            self.IMAGE_MIN_DIM = 256
            self.IMAGE_MAX_DIM = 256
            self.IMAGE_SHAPE = np.array([self.IMAGE_H, self.IMAGE_W, 3])
            self.RGB_CLIP_SHAPE = np.array([self.TIMESTEPS, self.IMAGE_H, self.IMAGE_W, 3])
            self.FLOW_CLIP_SHAPE = np.array([self.TIMESTEPS, self.IMAGE_H, self.IMAGE_W, 2])
            self.NUM_ACTOR_CLASSES = 1 + 7 # background + 7 actors
            self.NUM_ACTION_CLASSES = 1 + 9 # background + 9 actions. Note that `None` is different from background
            self.I3D_NUM_CLASSES = 400
            self.I3D_DROPOUT_KEEP_PROB = 1.0
            self.I3D_SPATIAL_SQUEEZE = True
            self.GPU_COUNT = 4
            self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT
            self.RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
            self.RESNET = 'resnet101'
            self.TEMPORAL_LAYERS = True
    padnet_train = PADNet('training', TestConfig(), '../outputs/tmp')
    #padnet_test = PADNet('inference', TestConfig(), '../outputs/tmp')
    from IPython import embed; embed()

