from . import backend as K
import numpy as np
from pyeddl.utils.utils import iterate_minibatches


class Model(object):
    """The `Model` class adds training & evaluation routines to a `Network`.
    """

    def __init__(self, cmodel=None):
        if cmodel:
            self.c_model = cmodel

    @classmethod
    def from_model(cls, name):
        return cls(cmodel=K.get_model(name))

    def compile(self, optimizer,
                losses=None,
                metrics=None,
                loss_weights=None,
                sample_weight_mode=None,
                weighted_metrics=None,
                target_tensors=None,
                device='cpu',
                **kwargs):
        """Configures the model for training.
        # Arguments
           optimizer: String (name of optimizer) or optimizer instance.
           loss: String (name of objective function) or objective function.
               If the model has multiple outputs, you can use a different loss
               on each output by passing a dictionary or a list of losses.
               The loss value that will be minimized by the model
               will then be the sum of all individual losses.
           metrics: List of metrics to be evaluated by the model
               during training and testing.
               Typically you will use `metrics=['accuracy']`.
               To specify different metrics for different outputs of a
               multi-output model, you could also pass a dictionary,
               such as `metrics={'output_a': 'accuracy'}`.
           loss_weights: Optional list or dictionary specifying scalar
               coefficients (Python floats) to weight the loss contributions
               of different model outputs.
               The loss value that will be minimized by the model
               will then be the *weighted sum* of all individual losses,
               weighted by the `loss_weights` coefficients.
               If a list, it is expected to have a 1:1 mapping
               to the model's outputs. If a dict, it is expected to map
               output names (strings) to scalar coefficients.
           sample_weight_mode: If you need to do timestep-wise
               sample weighting (2D weights), set this to `"temporal"`.
               `None` defaults to sample-wise weights (1D).
               If the model has multiple outputs, you can use a different
               `sample_weight_mode` on each output by passing a
               dictionary or a list of modes.
           weighted_metrics: List of metrics to be evaluated and weighted
               by sample_weight or class_weight during training and testing.
           target_tensors: By default, Keras will create placeholders for the
               model's target, which will be fed with the target data during
               training. If instead you would like to use your own
               target tensors (in turn, Keras will not expect external
               Numpy data for these targets at training time), you
               can specify them via the `target_tensors` argument. It can be
               a single tensor (for a single-output model), a list of tensors,
               or a dict mapping output names to target tensors.
           **kwargs: When using the Theano/CNTK backends, these arguments
               are passed into `K.function`.
               When using the TensorFlow backend,
               these arguments are passed into `tf.Session.run`.
        # Raises
           ValueError: In case of invalid arguments for
               `optimizer`, `loss`, `metrics` or `sample_weight_mode`.
        """
        self.optimizer = optimizer
        self.losses = losses or []
        self.metrics = metrics or []
        self.loss_weights = loss_weights
        self.sample_weight_mode = sample_weight_mode
        self.weighted_metrics = weighted_metrics
        self.device = device

        # Compile model
        K.compile(self.c_model, self.optimizer, self.losses, self.metrics, self.device)


    def fit(self,
            x=None,
            y=None,
            batch_size=None,
            epochs=1,
            verbose=1,
            callbacks=None,
            validation_split=0.,
            validation_data=None,
            shuffle=True,
            class_weight=None,
            sample_weight=None,
            initial_epoch=0,
            steps_per_epoch=None,
            validation_steps=None,
            validation_freq=1,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False,
            **kwargs):
        """Trains the model for a fixed number of epochs (iterations on a dataset).
        Arguments:
            x: Input data. It could be:
                - A Numpy array (or array-like), or a list of arrays
                  (in case the model has multiple inputs).
                - A dict mapping input names to the corresponding
                  array/tensors, if the model has named inputs.
            y: Target data. Like the input data `x`,
                it could be either Numpy array(s), framework-native tensor(s),
                list of Numpy arrays (if the model has multiple outputs) or
                None (default) if feeding from framework-native tensors
                (e.g. TensorFlow data tensors).
                If output layers in the model are named, you can also pass a
                dictionary mapping output names to Numpy arrays.
            batch_size: Integer or `None`.
                Number of samples per gradient update.
                If unspecified, `batch_size` will default to 32.
                Do not specify the `batch_size` if your data is in the
                form of symbolic tensors, generators, or `Sequence` instances
                (since they generate batches).
            epochs: Integer. Number of epochs to train the model.
                An epoch is an iteration over the entire `x` and `y`
                data provided.
                Note that in conjunction with `initial_epoch`,
                `epochs` is to be understood as "final epoch".
                The model is not trained for a number of iterations
                given by `epochs`, but merely until the epoch
                of index `epochs` is reached.
            verbose: Integer. 0, 1, or 2. Verbosity mode.
                0 = silent, 1 = progress bar, 2 = one line per epoch.
            callbacks: List of callbacks to apply during training and validation
            validation_split: Float between 0 and 1.
                Fraction of the training data to be used as validation data.
                The model will set apart this fraction of the training data,
                will not train on it, and will evaluate
                the loss and any model metrics
                on this data at the end of each epoch.
                The validation data is selected from the last samples
                in the `x` and `y` data provided, before shuffling.
                This argument is not supported when `x` is a generator or
                `Sequence` instance.
            validation_data: Data on which to evaluate
                the loss and any model metrics at the end of each epoch.
                The model will not be trained on this data.
                `validation_data` will override `validation_split`.
                `validation_data` could be:
                    - tuple `(x_val, y_val)` of Numpy arrays or tensors
                    - tuple `(x_val, y_val, val_sample_weights)` of Numpy arrays
                    - dataset or a dataset iterator
                For the first two cases, `batch_size` must be provided.
                For the last case, `validation_steps` must be provided.
            shuffle: Boolean (whether to shuffle the training data
                before each epoch) or str (for 'batch').
                'batch' is a special option for dealing with the
                limitations of HDF5 data; it shuffles in batch-sized chunks.
                Has no effect when `steps_per_epoch` is not `None`.
            class_weight: Optional dictionary mapping class indices (integers)
                to a weight (float) value, used for weighting the loss function
                (during training only).
                This can be useful to tell the model to
                "pay more attention" to samples from
                an under-represented class.
            sample_weight: Optional Numpy array of weights for
                the training samples, used for weighting the loss function
                (during training only). You can either pass a flat (1D)
                Numpy array with the same length as the input samples
                (1:1 mapping between weights and samples),
                or in the case of temporal data,
                you can pass a 2D array with shape
                `(samples, sequence_length)`,
                to apply a different weight to every timestep of every sample.
                In this case you should make sure to specify
                `sample_weight_mode="temporal"` in `compile()`. This argument
                is not supported when `x` generator, or `Sequence` instance,
                instead provide the sample_weights as the third element of `x`.
            initial_epoch: Integer.
                Epoch at which to start training
                (useful for resuming a previous training run).
            steps_per_epoch: Integer or `None`.
                Total number of steps (batches of samples)
                before declaring one epoch finished and starting the
                next epoch. When training with input tensors such as
                TensorFlow data tensors, the default `None` is equal to
                the number of samples in your dataset divided by
                the batch size, or 1 if that cannot be determined.
            validation_steps: Only relevant if `steps_per_epoch`
                is specified. Total number of steps (batches of samples)
                to validate before stopping.
            validation_steps: Only relevant if `validation_data` is provided
                and is a generator. Total number of steps (batches of samples)
                to draw before stopping when performing validation at the end
                of every epoch.
            validation_freq: Only relevant if validation data is provided. Integer
                or list/tuple/set. If an integer, specifies how many training
                epochs to run before a new validation run is performed, e.g.
                `validation_freq=2` runs validation every 2 epochs. If a list,
                tuple, or set, specifies the epochs on which to run validation,
                e.g. `validation_freq=[1, 2, 10]` runs validation at the end
                of the 1st, 2nd, and 10th epochs.
            max_queue_size: Integer. Used for generator.
                Maximum size for the generator queue.
                If unspecified, `max_queue_size` will default to 10.
            workers: Maximum number of processes to spin up
                when using process-based threading. If unspecified, `workers`
                will default to 1. If 0, will execute the generator on the main
                thread.
            use_multiprocessing: Boolean. If `True`, use process-based
                threading. If unspecified, `use_multiprocessing` will default to
                `False`. Note that because this implementation relies on
                multiprocessing, you should not pass non-picklable arguments to
                the generator as they can't be passed easily to children processes.
            **kwargs: Used for backwards compatibility.
        Returns:
            A `History` object. Its `History.history` attribute is
            a record of training loss values and metrics values
            at successive epochs, as well as validation loss values
            and validation metrics values (if applicable).
        Raises:
            RuntimeError: If the model was never compiled.
            ValueError: In case of mismatch between the provided input data
                and what the model expects.
        """

        # Check optimizer
        # Check input sizes

        print("Training model...")
        for e in range(epochs):
            print("Epoch #{}...".format(e))
            for i, batch in enumerate(iterate_minibatches(x, y, batch_size, shuffle=shuffle)):
                print('\t- Training batch #{}...'.format(i))

                x_batch, y_batch = batch
                self.train_on_batch(x_batch, y_batch)

            print('\t=> Epoch #{}\tTrain loss: {:.5f}\tAcc: {:.5f}\tVal. loss: {:.5f}\tVal. acc: {:.5f}'.format(e, 0,0,0,0))


    def evaluate(self,
                 x=None,
                 y=None,
                 batch_size=None,
                 verbose=1,
                 sample_weight=None,
                 steps=None,
                 callbacks=None,
                 max_queue_size=10,
                 workers=1,
                 use_multiprocessing=False):
        """Returns the loss value & metrics values for the model in test mode.
        Computation is done in batches.
        # Arguments
            x: Input data. It could be:
                - A Numpy array (or array-like), or a list of arrays
                  (in case the model has multiple inputs).
                - A dict mapping input names to the corresponding
                  array/tensors, if the model has named inputs.
                - None (default) if feeding from framework-native
                  tensors (e.g. TensorFlow data tensors).
            y: Target data. Like the input data `x`,
                it could be either Numpy array(s), framework-native tensor(s),
                list of Numpy arrays (if the model has multiple outputs) or
                None (default) if feeding from framework-native tensors
                (e.g. TensorFlow data tensors).
                If output layers in the model are named, you can also pass a
                dictionary mapping output names to Numpy arrays.
            batch_size: Integer or `None`.
                Number of samples per gradient update.
                If unspecified, `batch_size` will default to 32.
                Do not specify the `batch_size` is your data is in the
                form of symbolic tensors, generators
            verbose: 0 or 1. Verbosity mode.
                0 = silent, 1 = progress bar.
            sample_weight: Optional Numpy array of weights for
                the test samples, used for weighting the loss function.
                You can either pass a flat (1D)
                Numpy array with the same length as the input samples
                (1:1 mapping between weights and samples),
                or in the case of temporal data,
                you can pass a 2D array with shape
                `(samples, sequence_length)`,
                to apply a different weight to every timestep of every sample.
                In this case you should make sure to specify
                `sample_weight_mode="temporal"` in `compile()`.
            steps: Integer or `None`.
                Total number of steps (batches of samples)
                before declaring the evaluation round finished.
                Ignored with the default value of `None`.
            callbacks: List of callbacks to apply during evaluation.
            max_queue_size: Integer. Maximum size for the generator queue.
                If unspecified, `max_queue_size` will default to 10.
            workers: Integer. Maximum number of processes to spin up when using
                process-based threading. If unspecified, `workers` will default
                to 1. If 0, will execute the generator on the main thread.
            use_multiprocessing: Boolean. If `True`, use process-based
                threading. If unspecified, `use_multiprocessing` will default to
                `False`. Note that because this implementation relies on
                multiprocessing, you should not pass non-picklable arguments to
                the generator as they can't be passed easily to children processes.
        # Raises
            ValueError: in case of invalid arguments.
        # Returns
            Scalar test loss (if the model has a single output and no metrics)
            or list of scalars (if the model has multiple outputs
            and/or metrics). The attribute `model.metrics_names` will give you
            the display labels for the scalar outputs.
        """
        pass

    def train_on_batch(self, x, y,
                       sample_weight=None,
                       class_weight=None):
        """Runs a single gradient update on a single batch of data.

        Args:
            x: Numpy array of training data,
                or list of Numpy arrays if the model has multiple inputs.
                If all inputs in the model are named,
                you can also pass a dictionary
                mapping input names to Numpy arrays.
            y: Numpy array of target data,
                or list of Numpy arrays if the model has multiple outputs.
                If all outputs in the model are named,
                you can also pass a dictionary
                mapping output names to Numpy arrays.
            sample_weight: Optional array of the same length as x, containing
                weights to apply to the model's loss for each sample.
                In the case of temporal data, you can pass a 2D array
                with shape (samples, sequence_length),
                to apply a different weight to every timestep of every sample.
                In this case you should make sure to specify
                sample_weight_mode="temporal" in compile().
            class_weight: Optional dictionary mapping
                class indices (integers) to
                a weight (float) to apply to the model's loss for the samples
                from this class during training.
                This can be useful to tell the model to "pay more attention" to
                samples from an under-represented class.

        Returns:
            Scalar training loss
            (if the model has a single output and no metrics)
            or list of scalars (if the model has multiple outputs
            and/or metrics). The attribute `model.metrics_names` will give you
            the display labels for the scalar outputs.

        """
        K.train_batch(self.c_model, x, y)

    def summary(self):
        return K.summary(self.c_model)

    def plot(self, filename='model.pdf'):
        return K.plot(self.c_model, filename)
