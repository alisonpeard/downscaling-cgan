import h5py
from tensorflow.keras import backend as K


class Nontrainable(object):
    """Creates a non-trainable environment for a model, use with 'with'."""
    def __init__(self, models):
        models = ensure_list(models)
        self.models = models

    def __enter__(self):
        """Set all models to nontrainable and keep record of original status."""
        self.trainable_status = [m.trainable for m in self.models]
        for m in self.models:
            m.trainable = False
        return self.models

    def __exit__(self, type, value, traceback):
        """Return models to original trainable status."""
        for (m, t) in zip(self.models, self.trainable_status):
            m.trainable = t


def save_opt_weights(model, filepath):
    with h5py.File(filepath, 'w') as f:
        # Save optimizer weights.
        symbolic_weights = getattr(model.optimizer, 'weights')
        if symbolic_weights:
            optimizer_weights_group = f.create_group('optimizer_weights')
            weight_values = K.batch_get_value(symbolic_weights)
            weight_names = []
            for i, (w, val) in enumerate(zip(symbolic_weights,
                                             weight_values)):
                if hasattr(w, 'name') and w.name:
                    name = str(w.name)
                else:
                    name = 'param_' + str(i)
                weight_names.append(name.encode('utf8'))
            optimizer_weights_group.attrs['weight_names'] = weight_names
            for name, val in zip(weight_names, weight_values):
                param_dset = optimizer_weights_group.create_dataset(
                    name,
                    val.shape,
                    dtype=val.dtype)
                if not val.shape:
                    # scalar
                    param_dset[()] = val
                else:
                    param_dset[:] = val


def load_opt_weights(model, filepath):
    with h5py.File(filepath, mode='r') as f:
        optimizer_weights_group = f['optimizer_weights']  # h5py group
        optimizer_weight_names = optimizer_weights_group.attrs['weight_names']
        for name in optimizer_weight_names:
            optimizer_weight_values = optimizer_weights_group[name]
        model.optimizer.set_weights(optimizer_weight_values)


def ensure_list(x):
    if type(x) != list:
        x = [x]
    return x


def input_shapes(model, prefix):
    """Grab input shapes for each model (optionally filter by prefix)"""
    shapes = [il.shape[1:] for il in
              model.inputs if il.name.startswith(prefix)]
    shapes = [tuple([d for d in dims]) for dims in shapes]
    return shapes
