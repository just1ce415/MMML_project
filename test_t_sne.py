import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import unittest
from t_sne import *


class TestTSNE(unittest.TestCase):
    def setUp(self):
        print("Running Set Up")
        self.X_test = np.array([[1, 2, 3], [1, 1, 1], [3, 3, 3]], dtype="float64")
        self.Y_test = np.power(self.X_test, 1e-4)
        self.D = np.array([[0.0, 5.0, 5.0], [5.0, 0.0, 12.0], [5.0, 12.0, 0.0]])

    def tearDown(self):
        print("Running Tear Down")

    def test_pairwise_sq_distances(self):
        D = pairwise_sq_distances(self.X_test)
        self.assertTrue(np.allclose(D, self.D))


if __name__ == "__main__":
    unittest.main()
