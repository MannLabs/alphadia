# native imports
import os

import matplotlib.pyplot as plt
import numpy as np

# alphadia imports
# alpha family imports
# third party imports
from matplotlib import patches


def _generate_patch_collection_nomobility(
    cycle: np.ndarray, cmap_name: str, start_val: float = 0.4, stop_val: float = 1.0
) -> list[dict]:
    """Generate a collection of patches for a DIA cycle for an experiment without ion mobility separation.

    Parameters
    ----------

    cycle : np.ndarray, shape = (1, n_mz, 1, 2), dtype = np.float32
        DIA cycle to plot

    cmap_name : str, optional, default = 'YlOrRd'
        Name of the colormap to use

    start_val : float, optional, default = 0.4
        Start value for the colormap

    stop_val : float, optional, default = 1.0
        Stop value for the colormap

    Returns
    -------
    typing.List[dict]
        typing.List of dicts for building the patches. Can be plotted by feeding them into plotting._plot_patch_collection

    """

    # remove first dim for convenience
    _cycle = cycle[0]

    n_frames = _cycle.shape[0]

    cmap = plt.get_cmap(cmap_name)

    # a slice is a rectangular region in the quadrupole, scan space
    slice_collection = []

    ms2_cycle = _cycle[_cycle[:, 0, 0] > 0]
    min_mz = np.min(ms2_cycle)
    max_mz = np.max(ms2_cycle)

    for i, frame in enumerate(_cycle):
        current_limit = frame[0]
        if np.all(current_limit < 0):
            current_limit = np.array([min_mz, max_mz])

        slice_collection.append(
            {
                "scan": np.array([i, i + 1]),
                "limits": current_limit,
                "color": cmap(start_val + (1 - start_val) * i / n_frames),
            }
        )

    return slice_collection


def _plot_patch_collection(
    patch_collection: list[dict], ax: plt.Axes = None, alpha: float = 0.5
):
    """Plot a collection of patches.

    Parameters
    ----------

    patch_collection : typing.List[dict]
        typing.List of dicts for building the patches.

    ax : plt.Axes, optional
        Axes to plot on. If None, the current axes will be used.

    alpha : float, optional, default = 0.5
        Alpha value for the patches

    """
    if ax is None:
        ax = plt.gca()

    for element in patch_collection:
        ax.add_patch(
            patches.Rectangle(
                (element["limits"][0], element["scan"][0]),
                element["limits"][1] - element["limits"][0],
                element["scan"][1] - element["scan"][0],
                color=element["color"],
                alpha=alpha,
            )
        )


def plot_dia_cycle_nomobility(
    cycle: np.ndarray,
    quad_start: float = None,
    quad_stop: float = None,
    ax: plt.Axes = None,
    cmap_name: str = "YlOrRd",
):
    """Plot a DIA cycle for an experiment without ion mobility separation.

    Parameters
    ----------

    cycle : np.ndarray, shape = (1, n_mz, 1, 2), dtype = np.float32
        DIA cycle to plot

    quad_start : float
        Start of the quadrupole scan range

    quad_stop : float
        End of the quadrupole scan range

    ax : plt.Axes, optional
        Axes to plot on. If None, the current axes will be used.

    cmap_name : str, optional, default = 'YlOrRd'
        Name of the colormap to use
    """

    if ax is None:
        ax = plt.gca()

    patch_collection = _generate_patch_collection_nomobility(cycle, cmap_name)
    _plot_patch_collection(patch_collection, ax)

    min_scan = min([s["scan"][0] for s in patch_collection])
    max_scan = max([s["scan"][1] for s in patch_collection])
    min_mz = min([s["limits"][0] for s in patch_collection])
    max_mz = max([s["limits"][1] for s in patch_collection])

    if (quad_start is not None) and (quad_stop is not None):
        ax.add_patch(
            plt.Rectangle(
                (quad_start, min_scan),
                quad_stop - quad_start,
                max_scan - min_scan,
                color="blue",
                alpha=0.5,
            )
        )

    ax.set_xlim(min_mz, max_mz)
    ax.set_ylim(min_scan, max_scan)

    ax.set_xlabel("Quadrupole m/z")
    ax.set_ylabel("Scan")

    # remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _generate_patch_collection_mobility(
    fragment_cycle: np.ndarray, cmap_name: str, start_val: float = 0.4
):
    """Generate a collection of patches for a DIA cycle for an experiment with ion mobility separation.

    Parameters
    ----------

    fragment_cycle : np.ndarray, shape = (n_frames, n_mz, n_scan, 2), dtype = np.float32
        DIA cycle to plot

    cmap_name : str, optional, default = 'YlOrRd'
        Name of the colormap to use

    start_val : float, optional, default = 0.4
        Start value for the colormap

    Returns
    -------
    typing.List[dict]
        typing.List of dicts for building the patches. Can be plotted by feeding them into plotting._plot_patch_collection

    """

    cmap = plt.get_cmap(cmap_name)

    # a slice is a rectangular reagion in the quadrupole, scan space
    patch_collection = []

    for i, frame in enumerate(fragment_cycle):
        current_limit = frame[0]
        scan_start = 0

        for j, slice in enumerate(frame):
            new_limit = slice
            if not np.all(new_limit == current_limit) and new_limit[0] != 0:
                patch_collection.append(
                    {
                        "scan": np.array([scan_start, j]),
                        "limits": current_limit,
                        "color": cmap(
                            start_val + (1 - start_val) * i / len(fragment_cycle)
                        ),
                    }
                )
                current_limit = new_limit
                scan_start = j

        patch_collection.append(
            {
                "scan": np.array([scan_start, j]),
                "limits": current_limit,
                "color": cmap(start_val + (1 - start_val) * i / len(fragment_cycle)),
            }
        )

    return patch_collection


def plot_dia_cycle_mobility(
    cycle: np.ndarray,
    quad_start: float,
    quad_stop: float,
    scan_start: float,
    scan_stop: float,
    mobility_limits=None,
    ax: plt.Axes = None,
    cmap_name: str = "YlOrRd",
):
    """Plot a DIA cycle for an experiment with ion mobility separation.

    Parameters
    ----------

    cycle : np.ndarray, shape = (1, n_mz, n_scan, 2), dtype = np.float32
        DIA cycle to plot

    quad_start : float
        Start of the quadrupole scan range

    quad_stop : float
        End of the quadrupole scan range

    scan_start : float
        Start of the ion mobility scan range

    scan_stop : float
        End of the ion mobility scan range

    ax : plt.Axes, optional
        Axes to plot on. If None, the current axes will be used.

    cmap_name : str, optional, default = 'YlOrRd'
        Name of the colormap to use
    """

    if ax is None:
        ax = plt.gca()

    # remove pure precursor frames
    fragment_frames = ~np.all(cycle == np.array([-1.0, -1.0]), axis=(2, 3))

    # cycle object with only fragment frames and without empty first dim
    # (1, 9, 928, 2) => (8, 928, 2)
    fragment_cycle = cycle[fragment_frames]

    patch_collection = _generate_patch_collection_mobility(fragment_cycle, cmap_name)
    _plot_patch_collection(patch_collection, ax)

    if (
        (quad_start is not None)
        and (quad_stop is not None)
        and (scan_start is not None)
        and (scan_stop is not None)
    ):
        ax.add_patch(
            patches.Rectangle(
                (quad_start, scan_start),
                quad_stop - quad_start,
                scan_stop - scan_start,
                color="blue",
                alpha=0.5,
            )
        )

    ax.set_xlim((np.min(fragment_cycle[fragment_cycle > 0]), np.max(fragment_cycle)))
    ax.set_ylim((fragment_cycle.shape[1], 0))

    ax.set_xlabel("Quadrupole m/z")
    ax.set_ylabel("Scan")

    # remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_cycle(
    cycle: np.ndarray,
    quad_start: float = None,
    quad_stop: float = None,
    scan_start: float = None,
    scan_stop: float = None,
    figure_path: str = None,
    # neptune_run = None
):
    """Plot the DIA cycle

    Parameters
    ----------

    cycle : np.ndarray, shape = (1, n_mz, n_ion_mobility, 2), dtype = np.float32
        DIA cycle to plot

    quad_start : float
        Start of the quadrupole scan range

    quad_stop : float
        End of the quadrupole scan range

    scan_start : float
        Start of the ion mobility scan range

    scan_stop : float
        End of the ion mobility scan range

    figure_path : str, optional
        Path to save the figure to

    neptune_run : neptune.Run, optional
        Neptune run to upload the figure to
    """

    fig, ax = plt.subplots(figsize=(5, 5))

    if cycle.shape[2] == 1:
        plot_dia_cycle_nomobility(
            cycle, quad_start=quad_start, quad_stop=quad_stop, ax=ax
        )
    elif cycle.shape[2] > 1:
        plot_dia_cycle_mobility(
            cycle,
            quad_start=quad_start,
            quad_stop=quad_stop,
            scan_start=scan_start,
            scan_stop=scan_stop,
            ax=ax,
        )
    else:
        raise ValueError("DIA cycle has invalid shape for plotting")

    if figure_path is not None:
        fig.savefig(os.path.join(figure_path, "cycle.png"), dpi=300)

    # if neptune_run is not None:
    #    neptune_run['cycle'].upload(fig)
