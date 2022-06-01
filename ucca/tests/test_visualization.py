import pytest

from ucca.visualization import draw, tikz
from .conftest import PASSAGES

"""Tests the visualization module functions and classes."""


@pytest.mark.parametrize("create", PASSAGES)
def test_draw(create):
    import matplotlib
    matplotlib.use('Agg')
    draw(create())


@pytest.mark.parametrize("create", PASSAGES)
def test_tikz(create):
    tikz(create())
