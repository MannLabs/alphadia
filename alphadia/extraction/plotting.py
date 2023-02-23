# native imports

# alphadia imports
from alphadia.extraction.quadrupole import calculate_observation_importance

# alpha family imports

# third party imports

from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np


def _generate_slice_collection(
    fragment_cycle : np.ndarray, 
    cmap_name : str, 
    start_val : float = 0.4):

    cmap = cm.get_cmap(cmap_name)

    # a slice is a rectangular reagion in the quadrupole, scan space
    slice_collection = []

    for i, frame in enumerate(fragment_cycle):
        current_limit = frame[0]
        scan_start = 0

        for j, slice in enumerate(frame):
            new_limit = slice
            if not np.all(new_limit == current_limit) and new_limit[0] != 0:
            
                slice_collection.append({
                    'scan': np.array([scan_start, j]), 
                    'limits': current_limit, 
                    'color': cmap(start_val + (1-start_val)*i/len(fragment_cycle))
                })
                current_limit = new_limit
                scan_start = j

        slice_collection.append({
            'scan': np.array([scan_start, j]), 
            'limits': current_limit, 
            'color': cmap(start_val + (1-start_val)*i/len(fragment_cycle))
        })
    
    return slice_collection

from matplotlib import patches

def _plot_slice_collection(slice_collection, ax, alpha=0.5, **kwargs):
    
    for element in slice_collection:
        ax.add_patch(
            patches.Rectangle(
                (element['limits'][0], element['scan'][0]),
                element['limits'][1] - element['limits'][0],
                element['scan'][1] - element['scan'][0],
                color=element['color'],
                alpha=alpha,
                **kwargs
            )
        )

def plot_dia_cycle(cycle,ax=None, cmap_name='YlOrRd', **kwargs):

    if ax is None:
        ax = plt.gca()

    # remove pure precursor frames
    fragment_frames = ~np.all(cycle == np.array([-1.,-1.]) , axis=(2,3))
    
    # cycle object with only fragment frames and without empty first dim 
    # (1, 9, 928, 2) => (8, 928, 2)
    fragment_cycle = cycle[fragment_frames]

    slice_collection = _generate_slice_collection(fragment_cycle, cmap_name)
    _plot_slice_collection(slice_collection, ax, **kwargs)

    ax.set_xlim((np.min(fragment_cycle[fragment_cycle > 0]), np.max(fragment_cycle)))
    ax.set_ylim((fragment_cycle.shape[1], 0))
    
    ax.set_xlabel('Quadrupole m/z')
    ax.set_ylabel('Scan')

def plot_all_precursors(
    dense_precursors,
    qtf,
    template,
    isotope_intensity
):


    n_precursors = qtf.shape[0]
    n_isotopes = qtf.shape[1]
    n_observations = qtf.shape[2]
    n_scans = qtf.shape[3]

    # figure parameters
    n_cols = n_isotopes * 2 + 1
    n_rows = n_observations
    width_ratios = np.append(np.tile([2, 0.8], n_isotopes),[2])

    

    scan_range = np.arange(n_scans)
    observation_importance = calculate_observation_importance(
        template,
    )

    # iterate over precursors
    # each precursor will be a separate figure
    for i_precursor in range(n_precursors):

        v_min_dense = np.min(dense_precursors)
        v_max_dense = np.max(dense_precursors)

        v_min_template = np.min(template)
        v_max_template = np.max(template)

        fig, axs = plt.subplots(
            n_rows, 
            n_cols, 
            figsize = (n_cols * 1, n_rows*2), 
            gridspec_kw = {'width_ratios': width_ratios}, 
            sharey='row'
        )
        # expand axes if there is only one row
        if len(axs.shape) == 1:
            axs = axs.reshape(1, axs.shape[0])

        # iterate over observations, observations will be rows
        for i_observation in range(n_observations):

            # iterate over isotopes, isotopes will be columns
            for i_isotope in range(n_isotopes):
                
                # each even column will be a dense precursor
                i_dense = 2*i_isotope
                # each odd column will be a qtf
                i_qtf = 2*i_isotope+1

                # as precursors and isotopes are stored in a flat array, we need to calculate the index
                dense_p_index = i_precursor * n_isotopes + i_isotope

                # plot dense precursor
                axs[i_observation,i_dense].imshow(
                    dense_precursors[0,dense_p_index,0],
                    vmin=v_min_dense,
                    vmax=v_max_dense,
                )
                axs[0,i_dense].set_title(f'isotope {i_isotope}')
                # add text with relative isotope intensity
                axs[i_observation,i_dense].text(
                    0.05, 
                    0.95, 
                    f'{isotope_intensity[i_precursor,i_isotope]*100:.2f} %', 
                    horizontalalignment='left', 
                    verticalalignment='top', 
                    transform=axs[i_observation,i_dense].transAxes, 
                    color='white'
                )

                # plot qtf and weighted qtf
                axs[i_observation,i_qtf].plot(
                    qtf[i_precursor,i_isotope,i_observation],
                    scan_range
                )
                axs[i_observation,i_qtf].plot(
                    qtf[i_precursor,i_isotope,i_observation] * isotope_intensity[i_precursor,i_isotope],
                    scan_range
                )
                axs[i_observation,i_qtf].set_xlim(0, 1)
                axs[-1,i_dense].set_xlabel(f'frame')
 
            # remove xticks from all but last row
            if i_observation < n_observations - 1:
                for ax in axs[i_observation,:].flat:
                    ax.set_xticks([])
                    
            # bold title
            axs[0,-1].set_title(f'template', fontweight='bold')

            axs[i_observation,-1].imshow(
                template[i_precursor,i_observation],
                vmin=v_min_template,
                vmax=v_max_template,
            )

            axs[i_observation,-1].text(
                0.05, 
                0.95, 
                f'{observation_importance[i_precursor,i_observation]*100:.2f} %', 
                horizontalalignment='left', 
                verticalalignment='top', 
                transform=axs[i_observation,-1].transAxes, 
                color='white'
            )
            axs[i_observation,0].set_ylabel(f'observation {i_observation}\nscan')
        
        fig.tight_layout()
        plt.show()

def plot_image_collection(
    images
):
    n_images = len(images)
    fig, ax = plt.subplots(1, n_images, figsize=(n_images*4, 6))

    if n_images == 1:
        ax = [ax]
        
    for i_image, image in enumerate(images):
        ax[i_image].imshow(image)
        ax[i_image].spines[['right', 'top','left','bottom']].set_visible(False)
        ax[i_image].set_xticks([])
        ax[i_image].set_yticks([])
    fig.tight_layout()
    plt.show()


def plot_fragment_profile(
    template,
    fragment_scan_profile,
    fragment_frame_profile,
    template_frame_profile,
    template_scan_profile,
):

    n_fragments = fragment_scan_profile.shape[0]
    n_observations = fragment_scan_profile.shape[1]
    n_scans = fragment_scan_profile.shape[2]
    n_frames = fragment_frame_profile.shape[2]

    scan_indices = np.arange(n_scans)

    grid_spec = dict(
        height_ratios=[0.2, 0.2,1],
        width_ratios=[1,0.5,0.5],
    )

    images = []

    for i_observation in range(n_observations):
        fig, ax = plt.subplots(3, 3, figsize=(n_frames*0.2*1.2, n_scans*0.12*1.2), sharey='row',sharex='col', gridspec_kw=grid_spec)

        ax[2, 0].imshow(template[0,i_observation])
        ax[2, 1].plot(template_scan_profile[0,i_observation],scan_indices)
        ax[1, 0].plot(template_frame_profile[0,i_observation])

        for j in range(n_fragments):
            ax[2, 2].plot(fragment_scan_profile[j,i_observation], scan_indices)
            ax[0, 0].plot(fragment_frame_profile[j,i_observation])

        ax[0, 0].spines[['right', 'top']].set_visible(False)
        ax[0, 0].set_yticks([])

        ax[1, 0].spines[['right', 'top']].set_visible(False)
        ax[1, 0].set_yticks([])

        ax[2, 1].spines[['right', 'top']].set_visible(False)
        ax[2, 1].set_xticks([])

        ax[2, 2].spines[['right', 'top']].set_visible(False)
        ax[2, 2].set_xticks([])

        ax[0, 1].remove()
        ax[0, 2].remove()
        ax[1, 1].remove()
        ax[1, 2].remove()

        ax[0, 0].set_ylabel('Fragments')
        ax[1, 0].set_ylabel('Template')
        ax[2, 0].set_ylabel('Scans')

        ax[2, 0].set_xlabel('Frames')
        ax[2, 1].set_xlabel('Template')
        ax[2, 2].set_xlabel('Fragments')

        fig.suptitle(f'Observation {i_observation}')

        
        #fig.tight_layout()
        fig.set_dpi(200)
        fig.canvas.draw()
        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        images.append(image_from_plot)

        plt.close(fig)

    plot_image_collection(images)


