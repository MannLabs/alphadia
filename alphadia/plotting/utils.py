# native imports

# alphadia imports

# alpha family imports

# third party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Parameters
    ----------
    color : str, tuple
        color to lighten

    amount : float, default 0.5
        amount to lighten the color

    Returns
    -------
    tuple
        lightened color
    """
    import colorsys

    import matplotlib.colors as mc

    try:
        c = mc.cnames[color]
    except KeyError:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def density_scatter(
    x: np.ndarray | pd.Series | pd.DataFrame,
    y: np.ndarray | pd.Series | pd.DataFrame,
    axis: plt.Axes = None,
    bw_method=None,
    s: float = 1,
    **kwargs,
):
    """
    Scatter plot colored by kerneld density estimation

    Parameters
    ----------

    x : np.ndarray, pd.Series, pd.DataFrame
        x values

    y : np.ndarray, pd.Series, pd.DataFrame
        y values

    axis : plt.Axes, optional
        axis to plot on. If None, the current axis is used

    s : float, default 1
        size of the points

    **kwargs
        additional arguments passed to plt.scatter

    Examples
    --------

    Example using two-dimensional data normal distributed data:

    ```python
    x = np.random.normal(0, 1, 1000)
    y = np.random.normal(0, 1, 1000)

    density_scatter(x, y)
    ```

    """

    if not isinstance(x, np.ndarray):
        x = x.to_numpy()

    if x.ndim > 1:
        raise ValueError("x must be 1-dimensional")

    if not isinstance(y, np.ndarray):
        y = y.to_numpy()

    if y.ndim > 1:
        raise ValueError("y must be 1-dimensional")

    if axis is None:
        axis = plt.gca()

    # Calculate the point density
    xy = np.vstack([x, y])
    z = gaussian_kde(xy, bw_method=bw_method)(xy)

    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    axis.scatter(x, y, c=z, s=s, **kwargs)


def plot_image_collection(
    images: list[np.ndarray], image_width: float = 4, image_height: float = 6
):
    n_images = len(images)
    fig, ax = plt.subplots(1, n_images, figsize=(n_images * image_width, image_height))

    if n_images == 1:
        ax = [ax]

    for i_image, image in enumerate(images):
        ax[i_image].imshow(image)
        ax[i_image].spines[["right", "top", "left", "bottom"]].set_visible(False)
        ax[i_image].set_xticks([])
        ax[i_image].set_yticks([])
    fig.tight_layout()
    plt.show()
