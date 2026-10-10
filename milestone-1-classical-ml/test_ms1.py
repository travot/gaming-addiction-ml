"""
Module for auto-testing student projects.
This is the Milestone 1 version.
"""

import re
import sys
import os
import unittest
import importlib
from pathlib import Path

import numpy as np


class HidePrints:
    """Disable normal printing for calling student code."""

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class NoHidePrints:
    """Don't disable normal printing for calling student code."""

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class TestProject(unittest.TestCase):

    @staticmethod
    def title(msg):
        print(f"\n==============\n> {msg} ...")

    def test_1_folder_structure(self):
        """Test the framework structure (folder and files)."""
        self.title("Testing folder structure")
        self.assertTrue(project_path.exists(), f"No folder found at {project_path}")

        # Main files
        for file in ["main.py"]:
            with self.subTest(f"Checking file {file}"):
                self.assertTrue((project_path / file).exists(), f"No file {file} found at {project_path}")

        # Source code
        src_path = project_path / "src"
        self.assertTrue(src_path.exists(), f"{src_path} not found")
        for file in ["__init__.py", "utils.py"]:
            with self.subTest(f"Checking file src/{file}"):
                self.assertTrue((src_path / file).exists(), f"No file {file} found at {src_path}")
        # Methods
        method_path = src_path / "methods"
        self.assertTrue(method_path.exists(), f"{method_path} not found")
        for file in ["__init__.py", "dummy_methods.py", "logistic_regression.py", "knn.py", "linear_regression.py"]:
            with self.subTest(f"Checking file methods/{file}"):
                self.assertTrue((method_path / file).exists(), f"No file {file} found at {method_path}")

    def _import_and_test(self, name, class_name, *args, **kwargs):
        """Test the import of the method and its functions."""
        # Code structure
        module = importlib.import_module(f"src.methods.{name}")
        method = module.__getattribute__(class_name)(*args, **kwargs)
        for fn in ["fit", "predict"]:
            _ = method.__getattribute__(fn)

        # Functions inputs and outputs
        N, D = 10, 3
        training_data = np.random.rand(N, D)
        training_labels = np.random.randint(0, D, N)
        test_data = np.random.rand(N, D)
        with no_print():
            pred_labels = method.fit(training_data, training_labels)
        self.assertIsInstance(pred_labels, np.ndarray,
                              f"{name}.{class_name}.fit() should output an array, not {type(pred_labels)}")
        self.assertEqual(pred_labels.shape, training_labels.shape,
                         f"{name}.{class_name}.fit() output has wrong shape ({pred_labels.shape} != {training_labels.shape})")
        with no_print():
            pred_labels = method.predict(test_data)
        self.assertIsInstance(pred_labels, np.ndarray,
                              f"{name}.{class_name}.predict() should output an array, not {type(pred_labels)}")
        self.assertEqual(pred_labels.shape, training_labels.shape,
                         f"{name}.{class_name}.predict() output has wrong shape ({pred_labels.shape} != {training_labels.shape})")

        return method

    def test_2_dummy_methods(self):
        """Test the dummy methods."""
        self.title("Testing dummy methods")

        _ = self._import_and_test("dummy_methods", "DummyClassifier",
                                  arg1=1)

    def test_3a_knn(self):
        """Test KNN."""
        self.title("Testing KNN")

        knn_model = self._import_and_test("knn", "KNN", k=1)

        #  Test on easy dummy data
        training_data = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
        training_labels = np.array([0, 1, 2, 3])
        test_data = np.array([[0., 0.1], [1.2, -0.2], [0.1, 0.9], [20., 20.]])
        test_labels = np.array([0, 1, 2, 3])
        with no_print():
            pred_labels_train = knn_model.fit(training_data, training_labels)
            pred_labels_test = knn_model.predict(test_data)
        self.assertTrue(np.equal(pred_labels_train, training_labels).all(), f"KNN.fit() is not working on dummy data")
        self.assertTrue(np.equal(pred_labels_test, test_labels).all(), f"KNN.predict() is not working on dummy data")

    def test_3b_logistic_regression(self):
        """Test Logistic Regression."""
        self.title("Testing Logistic Regression")

        logistic_regression = self._import_and_test("logistic_regression", "LogisticRegression",
                                                    lr=1e-3, max_iters=500)

        #  Test on easy dummy data
        N = 20
        training_data = np.concatenate([
            np.linspace(-5, -0.25, N // 2)[:, None],
            np.linspace(0.25, 5, N // 2)[:, None]
        ], axis=0)
        training_labels = (training_data[:, 0] > 0.).astype(int)
        test_data = np.array([-10., -5., -1., 1., 5., 10.])[:, None]
        test_labels = (test_data[:, 0] > 0.).astype(int)
        with no_print():
            pred_labels_train = logistic_regression.fit(training_data, training_labels)
            pred_labels_test = logistic_regression.predict(test_data)
        self.assertTrue((pred_labels_train == training_labels).all(),
                        f"LogisticRegression.fit() is not working on dummy data")
        self.assertTrue((pred_labels_test == test_labels).all(),
                        f"LogisticRegression.predict() is not working on dummy data")

    def test_3c_linear_regression(self):
        """Test Linear Regression."""
        self.title("Testing Linear Regression")

        LR_model = self._import_and_test("linear_regression", "LinearRegression")

        #  Test on easy dummy data
        N = 20
        training_data = np.linspace(-1, 1, N)[:, None]
        training_labels = 2 * training_data[:, 0]
        test_data = np.random.rand(N, 1) * 2 - 1
        test_labels = 2 * test_data[:, 0]
        with no_print():
            pred_labels_train = LR_model.fit(training_data, training_labels)
            pred_labels_test = LR_model.predict(test_data)
        self.assertTrue(np.isclose(pred_labels_train, training_labels).all(),
                        f"LinearRegression.fit() is not working on dummy data")
        self.assertTrue(np.isclose(pred_labels_test, test_labels).all(),
                        f"LinearRegression.predict() is not working on dummy data")


def warn(msg):
    print(f"\33[33m/!\\ Warning: {msg}\33[39m")


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--no-hide', action='store_true', help='Enable printing from the student code')
    args = parser.parse_args()

    project_path = Path(".")

    if args.no_hide:
        no_print = NoHidePrints
    else:
        no_print = HidePrints

    unittest.main(argv=[''], verbosity=0)
