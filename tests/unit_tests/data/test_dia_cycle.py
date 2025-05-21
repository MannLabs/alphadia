import numpy as np

from alphadia.data.dia_cycle import _get_cycle_length, _get_cycle_start, _is_valid_cycle


def test_cycle():
    rand_cycle_start = np.random.randint(0, 100)
    rand_cycle_length = np.random.randint(5, 10)
    rand_num_cycles = np.random.randint(10, 50)

    cycle = np.zeros(rand_cycle_start)
    cycle = np.append(cycle, np.tile(np.arange(rand_cycle_length), rand_num_cycles))

    cycle_length = _get_cycle_length(cycle)
    cycle_start = _get_cycle_start(cycle, cycle_length)
    cycle_valid = _is_valid_cycle(cycle, cycle_length, cycle_start)

    assert cycle_valid
    assert cycle_length == rand_cycle_length
    assert cycle_start == rand_cycle_start
