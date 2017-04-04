import numpy as np
import data_transforms
import data_iterators
import pathfinder
import lasagne as nn
from collections import namedtuple
from functools import partial
import lasagne
import theano.tensor as T
import utils
import configs_fpred_patch.luna_c3 as patch_class_config

restart_from_save = None
rng = np.random.RandomState(42)

# transformations
p_transform = {'patch_size': (48, 48, 48),
               'mm_patch_size': (48, 48, 48),
               'pixel_spacing': (1., 1., 1.)
               }
p_transform_augment = {
    'translation_range_z': [-3, 3],
    'translation_range_y': [-3, 3],
    'translation_range_x': [-3, 3],
    'rotation_range_z': [-180, 180],
    'rotation_range_y': [-180, 180],
    'rotation_range_x': [-180, 180]
}

positive_proportion = 0.8


# data preparation function
def data_prep_function(data, patch_center, pixel_spacing, luna_origin, p_transform,
                       p_transform_augment, world_coord_system, **kwargs):
    x, patch_annotation_tf = data_transforms.transform_patch3d(data=data,
                                                               luna_annotations=None,
                                                               patch_center=patch_center,
                                                               p_transform=p_transform,
                                                               p_transform_augment=p_transform_augment,
                                                               pixel_spacing=pixel_spacing,
                                                               luna_origin=luna_origin,
                                                               world_coord_system=world_coord_system)
    x = data_transforms.pixelnormHU(x)
    return x


def label_prep_function(label):
    return label


data_prep_function_train = partial(data_prep_function, p_transform_augment=p_transform_augment,
                                   p_transform=p_transform, world_coord_system=True)
data_prep_function_valid = partial(data_prep_function, p_transform_augment=None,
                                   p_transform=p_transform, world_coord_system=True)

# data iterators
batch_size = 16
nbatches_chunk = 1
chunk_size = batch_size * nbatches_chunk

train_valid_ids = utils.load_pkl(pathfinder.LUNA_VALIDATION_SPLIT_PATH)
train_pids, valid_pids = train_valid_ids['train'], train_valid_ids['valid']

train_data_iterator = data_iterators.CandidatesLunaDataGeneratorBetter(data_path=pathfinder.LUNA_DATA_PATH,
                                                                       batch_size=chunk_size,
                                                                       transform_params=p_transform,
                                                                       data_prep_fun=data_prep_function_train,
                                                                       rng=rng,
                                                                       patient_ids=train_pids,
                                                                       full_batch=True, random=True, infinite=True,
                                                                       positive_proportion=positive_proportion,
                                                                       label_prep_fun=label_prep_function,
                                                                       random_negative_samples=True)

valid_data_iterator = data_iterators.CandidatesLunaValidDataGenerator(data_path=pathfinder.LUNA_DATA_PATH,
                                                                      transform_params=p_transform,
                                                                      data_prep_fun=data_prep_function_valid,
                                                                      patient_ids=valid_pids,
                                                                      label_prep_fun=label_prep_function)

nchunks_per_epoch = train_data_iterator.nsamples / chunk_size
max_nchunks = nchunks_per_epoch * 50

validate_every = int(10. * nchunks_per_epoch)
save_every = int(2. * nchunks_per_epoch)

learning_rate_schedule = {
    0: 1e-4,
    int(max_nchunks * 0.5): 5e-5,
    int(max_nchunks * 0.6): 2e-5,
    int(max_nchunks * 0.7): 1e-5,
    int(max_nchunks * 0.8): 5e-6,
    int(max_nchunks * 0.9): 2e-7
}

untrained_weigths_grad_scale = 5

# model

dense = partial(lasagne.layers.DenseLayer,
                W=lasagne.init.Orthogonal(),
                nonlinearity=lasagne.nonlinearities.very_leaky_rectify)


def build_nodule_classification_model(l_in):
    metadata_dir = utils.get_dir_path('models', pathfinder.METADATA_PATH)
    metadata_path = utils.find_model_metadata(metadata_dir, patch_class_config.__name__.split('.')[-1])
    metadata = utils.load_pkl(metadata_path)
    model = patch_class_config.build_model(l_in)
    nn.layers.set_all_param_values(model.l_out, metadata['param_values'])
    return model


def build_model(l_in=None):
    l_in = nn.layers.InputLayer((None, 1,) + p_transform['patch_size']) if l_in is None else l_in
    l_target = nn.layers.InputLayer((None, 1))

    nodule_classification_model = build_nodule_classification_model(l_in)

    l_out = nn.layers.DenseLayer(nodule_classification_model.l_out.input_layer,
                                 num_units=1,
                                 W=nn.init.Constant(0.),
                                 b=nn.init.Constant(4.),
                                 nonlinearity=nn.nonlinearities.rectify)

    nodule_classification_model.l_out.input_layer.W.tag.grad_scale = untrained_weigths_grad_scale / 2.
    nodule_classification_model.l_out.input_layer.b.tag.grad_scale = untrained_weigths_grad_scale / 2.
    l_out.W.tag.grad_scale = untrained_weigths_grad_scale
    l_out.b.tag.grad_scale = untrained_weigths_grad_scale
    return namedtuple('Model', ['l_in', 'l_out', 'l_target'])(l_in, l_out, l_target)


def build_objective(model, deterministic=False, epsilon=1e-12):
    predictions = nn.layers.get_output(model.l_out, deterministic=deterministic)
    targets = nn.layers.get_output(model.l_target)
    loss = T.mean((predictions - targets) ** 2)
    return loss


def build_updates(train_loss, model, learning_rate):
    params = nn.layers.get_all_params(model.l_out)
    grads = T.grad(train_loss, params)
    for idx, param in enumerate(params):
        grad_scale = getattr(param.tag, 'grad_scale', 1)
        if grad_scale != 1:
            grads[idx] *= grad_scale

    updates = nn.updates.adam(grads, params, learning_rate)
    return updates