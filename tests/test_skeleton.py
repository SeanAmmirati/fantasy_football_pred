# -*- coding: utf-8 -*-

import pytest
from fantasy_football_predictions.skeleton import fib

__author__ = "Sean Ammirati"
__copyright__ = "Sean Ammirati"
__license__ = "mit"


def test_fib():
    assert fib(1) == 1
    assert fib(2) == 1
    assert fib(7) == 13
    with pytest.raises(AssertionError):
        fib(-10)
