"""
Unit and regression test for the mrthermo package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import mrthermo


def test_mrthermo_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "mrthermo" in sys.modules
