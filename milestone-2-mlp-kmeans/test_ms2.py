"""
Module for auto-testing student projects.
This is the Milestone 2 version.
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
        for file in ["__init__.py", "utils.py", "losses.py", "activations.py"]:
            with self.subTest(f"Checking file src/{file}"):
                self.assertTrue((src_path / file).exists(), f"No file {file} found at {src_path}")
        # Methods
        method_path = src_path / "methods"
        self.assertTrue(method_path.exists(), f"{method_path} not found")
        for file in ["__init__.py", "dummy_methods.py", "mlp.py", "kmeans.py"]:
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

    def test_3a_kmeans(self):
        """Test K-Means."""
        self.title("Testing K-Means")

        self._import_and_test("kmeans", "KMeans", K=3)

    def test_3b_mlp(self):
        """Test MLP."""
        self.title("Testing MLP")

        from src.methods.mlp import MLP
        from src.activations import Sigmoid
        from src.losses import MSE

        N, D, C = 20, 3, 2
        mlp = MLP(dimensions=(D, 10, C), activations=(Sigmoid, Sigmoid))

        for fn in ["fit", "predict", "feed_forward", "back_prop"]:
            self.assertTrue(hasattr(mlp, fn), f"MLP should have a '{fn}' method")

        training_data = np.random.rand(N, D)
        training_labels = np.random.randint(0, C, N)
        y_one_hot = np.zeros((N, C))
        y_one_hot[np.arange(N), training_labels] = 1
        test_data = np.random.rand(N, D)

        with no_print():
            mlp.fit(training_data, y_one_hot, loss=MSE, epochs=5, batch_size=5)

        with no_print():
            pred = mlp.predict(test_data)

        self.assertIsInstance(pred, np.ndarray,
                              f"MLP.predict() should output an array, not {type(pred)}")
        self.assertEqual(pred.shape, (N, C),
                         f"MLP.predict() output has wrong shape ({pred.shape} != {(N, C)})")


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
