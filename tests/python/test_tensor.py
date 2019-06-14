import unittest

import numpy as np

# from pyeddl.layers import Tensor
from pyeddl._C import Tensor, DEV_CPU

t_array = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12]
])


class TestTensors(unittest.TestCase):

    # def test_device(self):
    #     for d in [0, 1, 2]:
    #         t = Tensor(t_array.shape, dev=d)
    #         self.assertEqual(d, t.get_device())

    def test_ndim(self):
        t = Tensor(t_array.shape, DEV_CPU)
        self.assertEqual(len(t_array.shape), t.ndim)

    def test_size(self):
        t = Tensor(t_array.shape, DEV_CPU)
        self.assertEqual(t_array.size, t.size)

    def test_shape(self):
        t_shape = t_array.shape
        t = Tensor(t_shape, DEV_CPU)
        c_shape = t.shape

        # Check for dim
        self.assertEqual(len(t_shape), len(c_shape))
        for ts, cs in zip(t_shape, c_shape):
            self.assertEqual(ts, cs)
    #
    # def test_point2data(self):
    #     new_arr = np.asarray(t_array, dtype=np.float32)
    #     t = Tensor(new_arr.shape)
    #
    #     # Add data, get data
    #     t.point2data(new_arr)
    #
    #     # Check if there are the same
    #     self.assertTrue(np.array_equal(new_arr, t.get_data()))
    #
    #     # Modify original array
    #     new_arr *= 2.0
    #
    #     # Check if there are the same (again)
    #     self.assertTrue(np.array_equal(new_arr, t.get_data()))
    #
    # def test_add_data(self):
    #     new_arr = np.asarray(t_array, dtype=np.float32)
    #     t = Tensor(new_arr.shape)
    #
    #     # Add data, get data
    #     t.addData(new_arr)
    #
    #     # Check if there are the same
    #     self.assertTrue(np.array_equal(new_arr, t.get_data()))
    #
    #     # Modify original array
    #     new_arr *= 2.0
    #
    #     # Check if there are NOT the same
    #     self.assertFalse(np.array_equal(new_arr, t.get_data()))


if __name__ == "__main__":
    unittest.main()
