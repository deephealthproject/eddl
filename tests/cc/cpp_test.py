import unittest
import subprocess
import os


class MainTest(unittest.TestCase):
    def test_cpp(self):
        print("\n\nTesting C++ code...")
        test_bin_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../bin', 'eddl_test'))
        subprocess.check_call(test_bin_path)
        print("\nResuming Python tests...\n")


if __name__ == '__main__':
    unittest.main()
