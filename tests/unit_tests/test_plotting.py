import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from alphadia.plotting.utils import (
    lighten_color
)

from alphadia.plotting.cycle import (
    plot_cycle
)

def test_lighten_color():
    
    color = "#000000"
    lightened_color = lighten_color(color, 0.5)
    assert lightened_color == (0.5, 0.5, 0.5)

    color = (0, 0, 0)
    lightened_color = lighten_color(color, 0.5)
    assert lightened_color == (0.5, 0.5, 0.5)

def test_plot_cycle():

    # set backend to agg to avoid display issues
    matplotlib.use('agg')

    mobility_cycle = np.array([[
        [[-1., -1.],[-1., -1.],[-1., -1.],[-1., -1.]],
        [[100., 200.],[100., 200.],[300., 400.],[300., 400.]],
        [[200., 300.],[200., 300.],[400., 500.],[400., 500.]],
    ]])

    plot_cycle(mobility_cycle)

    no_mobility_cycle = mobility_cycle[:, :, [1]]
    plot_cycle(no_mobility_cycle)