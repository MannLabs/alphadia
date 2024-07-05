# native imports

# alphadia imports
# alpha family imports
# third party imports
import matplotlib.pyplot as plt
import numpy as np

from alphadia import quadrupole
from alphadia.plotting import utils


def plot_fragment_profile(
    template,
    fragment_scan_profile,
    fragment_frame_profile,
    template_frame_profile,
    template_scan_profile,
    has_mobility,
):
    n_fragments = fragment_scan_profile.shape[0]
    n_observations = fragment_scan_profile.shape[1]
    n_scans = fragment_scan_profile.shape[2]
    n_frames = fragment_frame_profile.shape[2]

    scan_indices = np.arange(n_scans)

    if has_mobility:
        figsize = (max(n_frames * 0.2 * 2, 5), n_scans * 0.12 * 2)
        grid_spec = dict(
            height_ratios=[0.2, 0.2, 1],
            width_ratios=[1, 0.5, 0.5],
        )
    else:
        figsize = (max(n_frames * 0.2 * 2, 5), 5)
        grid_spec = dict(
            height_ratios=[1, 1, 0.5],
            width_ratios=[1, 0.5, 0.5],
        )

    images = []

    for i_observation in range(n_observations):
        fig, ax = plt.subplots(
            3, 3, figsize=figsize, sharey="row", sharex="col", gridspec_kw=grid_spec
        )

        ax[2, 0].imshow(template[i_observation])
        ax[2, 1].plot(template_scan_profile[i_observation], scan_indices)
        ax[1, 0].plot(template_frame_profile[i_observation])

        for j in range(n_fragments):
            ax[2, 2].plot(fragment_scan_profile[j, i_observation], scan_indices)
            ax[0, 0].plot(fragment_frame_profile[j, i_observation])

        ax[0, 0].spines[["right", "top"]].set_visible(False)
        ax[0, 0].set_yticks([])

        ax[1, 0].spines[["right", "top"]].set_visible(False)
        ax[1, 0].set_yticks([])

        ax[2, 1].spines[["right", "top"]].set_visible(False)
        ax[2, 1].set_xticks([])

        ax[2, 2].spines[["right", "top"]].set_visible(False)
        ax[2, 2].set_xticks([])

        ax[0, 1].remove()
        ax[0, 2].remove()
        ax[1, 1].remove()
        ax[1, 2].remove()

        ax[0, 0].set_ylabel("Fragments")
        ax[1, 0].set_ylabel("Template")
        ax[2, 0].set_ylabel("Scans")

        ax[2, 0].set_xlabel("Frames")
        ax[2, 1].set_xlabel("Template")
        ax[2, 2].set_xlabel("Fragments")

        fig.suptitle(f"Observation {i_observation}")

        fig.tight_layout()
        fig.set_dpi(200)
        fig.canvas.draw()
        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(
            fig.canvas.get_width_height()[::-1] + (3,)
        )
        images.append(image_from_plot)

        plt.close(fig)

    utils.plot_image_collection(images)


def plot_precursor(dense_precursors):
    v_min = np.min(dense_precursors[0])
    v_max = np.max(dense_precursors[0])

    dpi = 20
    px_width_dense = max(dense_precursors.shape[4], 20)
    px_height_dense = dense_precursors.shape[3]

    n_precursors = dense_precursors.shape[1]
    n_cols = n_precursors

    n_observations = dense_precursors.shape[2]
    n_rows = n_observations * 2

    px_width_figure = px_width_dense * n_cols / dpi
    px_height_figure = px_height_dense * n_rows / dpi

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(px_width_figure, px_height_figure))

    if len(axs.shape) == 1:
        axs = axs.reshape(axs.shape[0], 1)

    for obs in range(n_observations):
        dense_index = obs * 2
        mass_index = obs * 2 + 1

        for frag in range(n_precursors):
            axs[dense_index, frag].imshow(
                dense_precursors[0, frag, obs], vmin=v_min, vmax=v_max
            )
            masked = np.ma.masked_where(
                dense_precursors[1, frag, obs] == 0, dense_precursors[1, frag, obs]
            )
            axs[mass_index, frag].imshow(masked, cmap="RdBu")

    fig.tight_layout()
    plt.show()


def plot_fragments(
    dense_fragments: np.ndarray,
    fragments,
):
    v_min = np.min(dense_fragments)
    v_max = np.max(dense_fragments)

    dpi = 20

    px_width_dense = max(dense_fragments.shape[4], 20)
    px_height_dense = dense_fragments.shape[3]

    n_fragments = dense_fragments.shape[1]
    n_cols = n_fragments

    n_observations = dense_fragments.shape[2]
    n_rows = n_observations * 2

    px_width_figure = px_width_dense * n_cols / dpi
    px_height_figure = px_height_dense * n_rows / dpi + 1

    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(px_width_figure, px_height_figure),
        sharex=True,
        sharey=True,
    )

    if len(axs.shape) == 1:
        axs = axs.reshape(1, -1)

    for obs in range(n_observations):
        dense_index = obs * 2
        mass_index = obs * 2 + 1
        for frag in range(n_fragments):
            frag_type = chr(fragments.type[frag])
            frag_charge = fragments.charge[frag]
            frag_number = fragments.number[frag]

            axs[dense_index, frag].set_title(f"{frag_type}{frag_number} z{frag_charge}")
            axs[dense_index, frag].imshow(
                dense_fragments[0, frag, obs], vmin=v_min, vmax=v_max
            )

            masked = np.ma.masked_where(
                dense_fragments[1, frag, obs] == 0, dense_fragments[1, frag, obs]
            )
            axs[mass_index, frag].imshow(masked, cmap="RdBu")

            axs[mass_index, frag].set_xlabel("frame")
        axs[mass_index, 0].set_ylabel(f"observation {obs}\n scan")
        axs[dense_index, 0].set_ylabel(f"observation {obs}\n scan")

    fig.tight_layout()
    plt.show()


def plot_template(dense_precursors, qtf, template, isotope_intensity):
    n_isotopes = qtf.shape[0]
    n_observations = qtf.shape[1]
    n_scans = qtf.shape[2]

    # figure parameters
    n_cols = n_isotopes * 2 + 1
    n_rows = n_observations
    width_ratios = np.append(np.tile([2, 0.8], n_isotopes), [2])

    scan_range = np.arange(n_scans)
    observation_importance = quadrupole.calculate_observation_importance_single(
        template,
    )

    # iterate over precursors
    # each precursor will be a separate figure
    v_min_dense = np.min(dense_precursors[0])
    v_max_dense = np.max(dense_precursors[0])

    v_min_template = np.min(template)
    v_max_template = np.max(template)

    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * 1, n_rows * 2),
        gridspec_kw={"width_ratios": width_ratios},
        sharey="row",
    )
    # expand axes if there is only one row
    if len(axs.shape) == 1:
        axs = axs.reshape(1, axs.shape[0])

    # iterate over observations, observations will be rows
    for i_observation in range(n_observations):
        # iterate over isotopes, isotopes will be columns
        for i_isotope in range(n_isotopes):
            # each even column will be a dense precursor
            i_dense = 2 * i_isotope
            # each odd column will be a qtf
            i_qtf = 2 * i_isotope + 1

            # plot dense precursor
            axs[i_observation, i_dense].imshow(
                dense_precursors[0, i_isotope, 0],
                vmin=v_min_dense,
                vmax=v_max_dense,
            )
            axs[0, i_dense].set_title(f"isotope {i_isotope}")
            # add text with relative isotope intensity
            axs[i_observation, i_dense].text(
                0.05,
                0.95,
                f"{isotope_intensity[i_isotope]*100:.2f} %",
                horizontalalignment="left",
                verticalalignment="top",
                transform=axs[i_observation, i_dense].transAxes,
                color="white",
            )

            # plot qtf and weighted qtf
            axs[i_observation, i_qtf].plot(qtf[i_isotope, i_observation], scan_range)
            axs[i_observation, i_qtf].plot(
                qtf[i_isotope, i_observation] * isotope_intensity[i_isotope], scan_range
            )
            axs[i_observation, i_qtf].set_xlim(0, 1)
            axs[-1, i_dense].set_xlabel("frame")

        # remove xticks from all but last row
        if i_observation < n_observations - 1:
            for ax in axs[i_observation, :].flat:
                ax.set_xticks([])

        # bold title
        axs[0, -1].set_title("template", fontweight="bold")

        axs[i_observation, -1].imshow(
            template[i_observation],
            vmin=v_min_template,
            vmax=v_max_template,
        )

        axs[i_observation, -1].text(
            0.05,
            0.95,
            f"{observation_importance[i_observation]*100:.2f} %",
            horizontalalignment="left",
            verticalalignment="top",
            transform=axs[i_observation, -1].transAxes,
            color="white",
        )
        axs[i_observation, 0].set_ylabel(f"observation {i_observation}\nscan")

    fig.tight_layout()
    plt.show()
