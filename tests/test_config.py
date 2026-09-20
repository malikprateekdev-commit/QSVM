"""
This module contains automated tests to verify the project configuration.
Automated tests are scripts that run our code and check if the output matches what we expect, helping prevent accidental errors.
"""

# pytest is a testing framework for Python.
# We import it to execute our test functions and report whether they pass or fail.
import pytest

# sys and os are standard Python libraries for interacting with the operating system and file paths.
# We import them to temporarily modify the Python path so this test script can find and import 'config.py' from the parent directory.
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# We import the configuration file so we can verify its variables.
import config

def test_configuration_constants():
    """
    Verifies that the constants defined in config.py exactly match the strict requirements of the experimental design.
    
    Input:
        - None. The function directly reads variables from the imported config module.
        
    Output:
        - None. The 'assert' statements will silently pass if the condition is true, or raise an AssertionError if false, which pytest detects as a failure.
    """
    assert config.PCA_COMPONENTS == 16
    assert config.NUM_SAMPLES == 200
    assert config.TRAIN_TEST_SPLIT == 0.2
    assert "basis_encoding" in config.ENCODING_METHODS
    assert "classical_rbf" in config.ENCODING_METHODS
