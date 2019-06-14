"""MNIST handwritten digits dataset.
"""
import numpy as np

from pyeddl.utils.data_utils import get_file


def load_data(path='mnist.npz'):
  """Loads the MNIST dataset.

  Arguments:
      path: path where to cache the dataset locally

  Returns:
      Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.

  License:
      Yann LeCun and Corinna Cortes hold the copyright of MNIST dataset,
      which is a derivative work from original NIST datasets.
      MNIST dataset is made available under the terms of the
      [Creative Commons Attribution-Share Alike 3.0 license.](
      https://creativecommons.org/licenses/by-sa/3.0/)

  """

  origin_folder = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/'
  path = get_file(
      path,
      origin=origin_folder + 'mnist.npz',
      file_hash='8a61469f7ea1b51cbae51d4f78837e45')
  with np.load(path) as f:
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']

    return (x_train, y_train), (x_test, y_test)
