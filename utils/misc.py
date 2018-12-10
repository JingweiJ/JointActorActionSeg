import numpy as np
import keras.backend as K
from keras.engine.topology import preprocess_weights_for_loading
import warnings

def RGB2Hex(R, G, B):
    assert R in range(256) and G in range(256) and B in range(256)
    return '0x' + '%06x' % (R * 256**2 + G * 256 + B)

def mask2bbox(mask, mode='xywh'):
    '''
    @input
        mask: np array of (h, w). Entries are 0 (background) or 1 (foreground).
    @return
        a tuple of (x, y, w, h), where (x, y) is the coordinates of the upperleft corner of bbox,
        w, h is the width and height of bbox.
    '''
    h, w = mask.shape
    fg_loc = np.where(mask)
    if mode == 'xywh':
        y = np.min(fg_loc[0])
        h = np.max(fg_loc[0]) - y
        x = np.min(fg_loc[1])
        w = np.max(fg_loc[1]) - x
        return (x, y, w, h)
    else:
        raise NotImplementedError()

def load_weights_from_hdf5_group_by_weights_name(f, weights, weight_name_map={}, verbose=False):
    ''' f: a hdf5 group of weights.
        weights: weights in the current graph to be assigned.
        weight_name_map: {key: value} == {weight name in current graph: corresponding weight name in f
        verbose: if True, print weight value tuples before assigning.
    '''
    weight_value_tuples = []
    for w in weights:
        try:
            if w.name in weight_name_map:
                match_weight_name = weight_name_map[w.name]
            else:
                match_weight_name = w.name
            weight_value_tuples.append(
                (w, f[match_weight_name])
            )
        except:
            print('Error!')
            print(w, w.name, match_weight_name)
    if verbose:
        print('Weight value tuples:')
        from pprint import pprint
        pprint(weight_value_tuples)
    K.batch_set_value(weight_value_tuples)


def load_weights_from_hdf5_group_by_name(f, layers, layer_name_map={}, skip_mismatch=False, verbose=False, verboseverbose=False):
    """
    Adapted from the function with same name in keras.engine.topology.
    Added warnings when a layer in the hdf5 file fails to match any layers
    in argument `layers`. Also print all assigned weights' names.

    Implements name-based weight loading.

    (instead of topological weight loading).

    Layers that have no matching name are skipped.

    # Arguments
        f: A pointer to a HDF5 group.
        layers: A list of target layers.
        skip_mismatch: Boolean, whether to skip loading of layers
            where there is a mismatch in the number of weights,
            or a mismatch in the shape of the weights.

    # Raises
        ValueError: in case of mismatch between provided layers
            and weights file and skip_mismatch=False.
    """
    if 'keras_version' in f.attrs:
        original_keras_version = f.attrs['keras_version'].decode('utf8')
    else:
        original_keras_version = '1'
    if 'backend' in f.attrs:
        original_backend = f.attrs['backend'].decode('utf8')
    else:
        original_backend = None

    # New file format.
    layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]

    # Reverse index of layer name to list of layers with name.
    index = {}
    for layer in layers:
        if layer.name:
            if layer.name in layer_name_map:
                index.setdefault(layer_name_map[layer.name], []).append(layer)
            else:
                index.setdefault(layer.name, []).append(layer)

    # We batch weight value assignments in a single backend call
    # which provides a speedup in TensorFlow.
    weight_value_tuples = []
    for k, name in enumerate(layer_names):
        g = f[name]
        weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
        weight_values = [g[weight_name] for weight_name in weight_names]

        for layer in index.get(name, []):
            symbolic_weights = layer.weights
            # Skip preprocessing
            #weight_values = preprocess_weights_for_loading(
            #    layer,
            #    weight_values,
            #    original_keras_version,
            #    original_backend)
            if len(weight_values) != len(symbolic_weights):
                if skip_mismatch:
                    warnings.warn('Skipping loading of weights for layer {}'.format(layer.name) +
                                  ' due to mismatch in number of weights' +
                                  ' ({} vs {}).'.format(len(symbolic_weights), len(weight_values)))
                    continue
                else:
                    raise ValueError('Layer #' + str(k) +
                                     ' (named "' + layer.name +
                                     '") expects ' +
                                     str(len(symbolic_weights)) +
                                     ' weight(s), but the saved weights' +
                                     ' have ' + str(len(weight_values)) +
                                     ' element(s).')
            # Set values.
            for i in range(len(weight_values)):
                if skip_mismatch:
                    # weights' order in `symbolic_weights` may not align with the order in `weight_values` and `weight_names`.
                    try:
                        if K.int_shape(symbolic_weights[i]) != weight_values[weight_names.index(symbolic_weights[i].name)].shape:
                            warnings.warn('Skipping loading of weights for layer {}'.format(layer.name) +
                                          ' due to mismatch in shape' +
                                          ' ({} vs {}).'.format(
                                              symbolic_weights[i].shape,
                                              weight_values[weight_names.index(symbolic_weights[i].name)].shape))
                            continue
                    except:
                        from pdb import set_trace; set_trace()

                weight_value_tuples.append((symbolic_weights[i],
                                           weight_values[weight_names.index(symbolic_weights[i].name)]))

    if len(weight_value_tuples) == 0:
        warnings.warn('No layer is loaded.')
        #return

    weights_in_layers = []
    for layer in layers:
        if layer.weights:
            weights_in_layers += layer.weights
    weights_to_be_assigned = [x for x, _ in weight_value_tuples]
    for wil in weights_in_layers:
        if wil not in weights_to_be_assigned:
            if verbose:
                warnings.warn('%s is not loaded.' % wil.name)
    if verboseverbose:
        print('Weight value tuples:')
        from pprint import pprint
        pprint(weight_value_tuples)
    K.batch_set_value(weight_value_tuples)

def load_weights_from_hdf5_group_by_name_assume_weight_order(f, layers, layer_name_map={}, skip_mismatch=False, verbose=False):
    """
    Adapted from the function with same name in keras.engine.topology.
    Added warnings when a layer in the hdf5 file fails to match any layers
    in argument `layers`. Also print all assigned weights' names.

    Implements name-based weight loading.

    (instead of topological weight loading).

    Layers that have no matching name are skipped.

    # Arguments
        f: A pointer to a HDF5 group.
        layers: A list of target layers.
        skip_mismatch: Boolean, whether to skip loading of layers
            where there is a mismatch in the number of weights,
            or a mismatch in the shape of the weights.

    # Raises
        ValueError: in case of mismatch between provided layers
            and weights file and skip_mismatch=False.
    """
    if 'keras_version' in f.attrs:
        original_keras_version = f.attrs['keras_version'].decode('utf8')
    else:
        original_keras_version = '1'
    if 'backend' in f.attrs:
        original_backend = f.attrs['backend'].decode('utf8')
    else:
        original_backend = None

    # New file format.
    layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]

    # Reverse index of layer name to list of layers with name.
    index = {}
    for layer in layers:
        if layer.name:
            if layer.name in layer_name_map:
                index.setdefault(layer_name_map[layer.name], []).append(layer)
            else:
                index.setdefault(layer.name, []).append(layer)

    # We batch weight value assignments in a single backend call
    # which provides a speedup in TensorFlow.
    weight_value_tuples = []
    for k, name in enumerate(layer_names):
        g = f[name]
        weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
        weight_values = [g[weight_name] for weight_name in weight_names]

        for layer in index.get(name, []):
            symbolic_weights = layer.weights
            weight_values = preprocess_weights_for_loading(
                layer,
                weight_values,
                original_keras_version,
                original_backend)
            if len(weight_values) != len(symbolic_weights):
                if skip_mismatch:
                    warnings.warn('Skipping loading of weights for layer {}'.format(layer.name) +
                                  ' due to mismatch in number of weights' +
                                  ' ({} vs {}).'.format(len(symbolic_weights), len(weight_values)))
                    continue
                else:
                    raise ValueError('Layer #' + str(k) +
                                     ' (named "' + layer.name +
                                     '") expects ' +
                                     str(len(symbolic_weights)) +
                                     ' weight(s), but the saved weights' +
                                     ' have ' + str(len(weight_values)) +
                                     ' element(s).')
            # Set values.
            for i in range(len(weight_values)):
                if skip_mismatch:
                    if K.int_shape(symbolic_weights[i]) != weight_values[i].shape:
                        warnings.warn('Skipping loading of weights for layer {}'.format(layer.name) +
                                      ' due to mismatch in shape' +
                                      ' ({} vs {}).'.format(
                                          symbolic_weights[i].shape,
                                          weight_values[i].shape))
                        continue

                weight_value_tuples.append((symbolic_weights[i],
                                            weight_values[i]))

    if len(weight_value_tuples) == 0:
        warnings.warn('No layer is loaded.')
        #return

    weights_in_layers = []
    for layer in layers:
        if layer.weights:
            weights_in_layers += layer.weights
    weights_to_be_assigned = [x for x, _ in weight_value_tuples]
    for wil in weights_in_layers:
        if wil not in weights_to_be_assigned:
            if verbose:
                warnings.warn('%s is not loaded.' % wil.name)
    if verbose:
        print('Weight value tuples:')
        from pprint import pprint
        pprint(weight_value_tuples)
    K.batch_set_value(weight_value_tuples)

def get_cross_label(actor_label, action_label, num_actor_class, num_action_class):
    ''' Given actor label and action label, compute the cross-product label.
    Inverse function of `decouple_cross_label`.

    actor_label: size of (batch_size, ), values are {1, ..., num_actor_class}.
    action_label: size of (batch_size, ), values are {1, ..., num_action_class}.
    num_actor_class: not including 'BG'. e.g. in A2D, num_actor_class = 7
    num_action_class: not including 'BG'. e.g. in A2D, num_action_class = 9

    return:
        cross_label: size of (batch_size,), int, values are {1, ..., num_actor_class
        * num_action_class}.
    '''
    cross_label = (actor_label - 1) * num_action_class + action_label
    return cross_label

def decouple_cross_label(cross_label, num_actor_class, num_action_class):
    ''' Given the cross-product label, compute the actor label and action label.
    Inverse function of `get_cross_label`.

    cross_label: size of (batch_size,), int, values are {1, ..., num_actor_class
    * num_action_class}.
    num_actor_class: not including 'BG'. e.g. in A2D, num_actor_class = 7
    num_action_class: not including 'BG'. e.g. in A2D, num_action_class = 9

    return:
        actor_label: size of (batch_size, ), values are {1, ..., num_actor_class}.
        action_label: size of (batch_size, ), values are {1, ..., num_action_class}.
    '''
    actor_label = (cross_label - 1) // num_action_class + 1
    action_label = cross_label - (actor_label - 1) * num_action_class
    return actor_label, action_label

