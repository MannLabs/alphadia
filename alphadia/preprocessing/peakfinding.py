"""Find peaks in dia data."""

import logging

import numpy as np

import alphatims.utils
import alphatims.tempmmap as tm


class PeakFinder:

    def __init__(
        self,
        tof_tolerance=3,
        cycle_tolerance=3,
    ):
        self.tof_tolerance = tof_tolerance
        self.cycle_tolerance = cycle_tolerance

    def set_dia_data(self, dia_data):
        self.dia_data = dia_data

    def set_connector(self, connector):
        self.connector = connector

    def set_smoother(self, smoother):
        self.smoother = smoother

    def find_peaks(self):
        self.assign_internal_points()
        self.find_cluster_paths()
        self.cluster_from_paths()
        self.find_ambiguous_cluster_overlaps()
        self.assemble_clusters()
        self.assign_quantifiable_clusters()
        self.peak_collection = PeakCollection()
        self.peak_collection.set_peak_indptr(
            self.dia_data.push_indptr,
            self.peaks,
        )

    def assign_internal_points(self):
        logging.info("Assigning internal points")
        self.internal_points = tm.empty(
            shape=self.smoother.smooth_intensity_values.shape,
            dtype=np.bool_
        )
        self.valid_neighborhood = np.ones(2**8, dtype=np.bool_)
        for index in range(2**8):
            bin_repr = '{:08b}'.format(index)
            if "00" in bin_repr:
                self.valid_neighborhood[index] = False
            if (index < 2**7) and (index % 2 == 0):
                self.valid_neighborhood[index] = False
        self.internal_points[:] = self.valid_neighborhood[self.smoother.neighbor_types]

    def find_cluster_paths(self):
        logging.info("Finding cluster paths")
        self.cluster_path_pointers = tm.clone(np.arange(len(self.dia_data)))
        cluster_to_max_peaks_(
            range(
                len(self.dia_data.push_indptr) // np.prod(
                    self.connector.cycle.shape[:-1]
                ) + 1
            ),
            self.dia_data.push_indptr,
            self.dia_data.tof_indices,
            self.smoother.smooth_intensity_values,
            self.tof_tolerance,
            self.dia_data.scan_max_index,
            self.dia_data.zeroth_frame,
            self.connector.connection_counts,
            self.connector.connections,
            self.cycle_tolerance,
            self.cluster_path_pointers,
        )

    def cluster_from_paths(self):
        logging.info("Clustering from paths")
        self.cluster_pointers = tm.clone(self.cluster_path_pointers)
        walk_cluster_path(np.arange(10))
        walk_cluster_path(self.cluster_pointers)

    def find_ambiguous_cluster_overlaps(self):
        logging.info("Detecting cluster ambiguities")
        self.nonambiguous_ions = tm.ones(len(self.dia_data), dtype=np.bool_)
        find_unique_peaks_(
            range(
                len(self.dia_data.push_indptr) // np.prod(
                    self.connector.cycle.shape[:-1]
                ) + 1
            ),
            self.dia_data.push_indptr,
            self.dia_data.tof_indices,
            self.smoother.smooth_intensity_values,
            self.tof_tolerance,
            self.dia_data.scan_max_index,
            self.dia_data.zeroth_frame,
            self.connector.connection_counts,
            self.connector.connections,
            self.cycle_tolerance,
            self.cluster_pointers,
            self.nonambiguous_ions,
        )
        logging.info("Removing cluster ambiguities")
        walk_unique_cluster_path(
            np.arange(10),
            np.zeros(10, dtype=np.bool_),
            np.ones(10, dtype=np.bool_),
        )
        to_visit = np.ones_like(self.nonambiguous_ions)
        walk_unique_cluster_path(
            self.cluster_path_pointers,
            self.nonambiguous_ions,
            to_visit,
        )

    def assemble_clusters(self):
        logging.info("Assembling clusters")
        self.cluster_assemblies = tm.clone(self.cluster_pointers)
        assemble_clusters(
            self.cluster_pointers,
            self.nonambiguous_ions,
            self.cluster_assemblies,
        )

    def assign_quantifiable_clusters(self):
        logging.info("Assigning quantifiable clusters")
        unique_peaks = np.unique(self.cluster_pointers)
        self.peaks = unique_peaks[
            (self.nonambiguous_ions & self.internal_points)[unique_peaks]
        ]


class PeakCollection(object):

    def set_peak_indptr(
        self,
        indptr: np.ndarray = None,
        peaks: np.ndarray = None,
    ):
        if peaks is None:
            return
        if peaks.dtype == np.bool_:
            self.indices = tm.clone(np.flatnonzero(peaks))
        else:
            self.indices = tm.clone(peaks)
        self.indptr = tm.empty(indptr.shape, indptr.dtype)
        set_peak_indptr(indptr, self.indptr, self.indices)


@alphatims.utils.njit
def set_peak_indptr(old_indptr, new_indptr, indices):
    count = 0
    offset = 0
    for index in indices:
        while index >= old_indptr[offset]:
            new_indptr[offset] = count
            offset += 1
        count += 1
    while index >= old_indptr[offset]:
        new_indptr[offset] = count
        offset += 1
    new_indptr[offset:] = count




@alphatims.utils.pjit
def cluster_to_max_peaks_(
    cycle_index,
    indptr,
    tof_indices,
    smooth_intensity_values,
    tof_tolerance,
    scan_max_index,
    zeroth_frame,
    connection_counts,
    connections,
    cycle_tolerance,
    clusters,
):
    len_cycle = len(connection_counts) - 1
    push_offset = len_cycle * cycle_index + zeroth_frame * scan_max_index
    for self_connection_index, connection_start in enumerate(
        connection_counts[:-1]
    ):
        connection_end = connection_counts[self_connection_index + 1]
        self_push_index = push_offset + self_connection_index
        if self_push_index > len(indptr):
            break
        self_start = indptr[self_push_index]
        self_end = indptr[self_push_index + 1]
        if self_start == self_end:
            continue
        if True:
            for other_connection_offset in connections[connection_start: connection_end]:
                other_push_index = self_push_index + other_connection_offset
                if other_push_index == self_push_index:
                    continue
                if not (0 <= other_push_index < len(indptr)):
                    continue
        # for cycle_offset in range(-cycle_tolerance, cycle_tolerance + 1):
        #     for other_connection_index in connections[connection_start: connection_end]:
        #         other_push_index = push_offset + other_connection_index + len_cycle * cycle_offset
        #         if other_push_index == self_push_index:
        #             continue
        #         if other_push_index >= len(indptr):
        #             continue
                other_start = indptr[other_push_index]
                other_end = indptr[other_push_index + 1]
                if other_start == other_end:
                    continue
                self_index = self_start
                other_index = other_start
                while (self_index < self_end) and (other_index < other_end):
                    self_tof = tof_indices[self_index]
                    other_tof = tof_indices[other_index]
                    if (self_tof - tof_tolerance) <= other_tof <= (self_tof + tof_tolerance):
                        self_ref = clusters[self_index]
                        max_intensity = smooth_intensity_values[self_ref]
                        other_intensity = smooth_intensity_values[other_index]
                        if max_intensity < other_intensity:
                            clusters[self_index] = other_index
                        elif max_intensity == other_intensity:
                            if self_index <= other_index:
                                clusters[self_index] = other_index
                    if self_tof < other_tof:
                        self_index += 1
                    else:
                        other_index += 1


@alphatims.utils.njit
def walk_cluster_path(
    clusters
):
    for index, pointer in enumerate(clusters):
        initial_index = index
        path_length = 1
        while (pointer >= 0) and (index != pointer):
            index = pointer
            pointer = clusters[index]
            path_length += 1
        if pointer >= 0:
            final_pointer = -(pointer + 1)
        else:
            final_pointer = pointer
        index = initial_index
        for i in range(path_length):
            pointer = clusters[index]
            clusters[index] = final_pointer
            index = pointer
    for index, pointer in enumerate(clusters):
        clusters[index] = -(pointer + 1)


@alphatims.utils.pjit
def find_unique_peaks_(
    cycle_index,
    indptr,
    tof_indices,
    smooth_intensity_values,
    tof_tolerance,
    scan_max_index,
    zeroth_frame,
    connection_counts,
    connections,
    cycle_tolerance,
    clusters,
    unique_peaks,
):
    len_cycle = len(connection_counts) - 1
    push_offset = len_cycle * cycle_index + zeroth_frame * scan_max_index
    for self_connection_index, connection_start in enumerate(
        connection_counts[:-1]
    ):
        connection_end = connection_counts[self_connection_index + 1]
        self_push_index = push_offset + self_connection_index
        if self_push_index > len(indptr):
            break
        self_start = indptr[self_push_index]
        self_end = indptr[self_push_index + 1]
        if self_start == self_end:
            continue
        if True:
            for other_connection_offset in connections[connection_start: connection_end]:
                other_push_index = self_push_index + other_connection_offset
                if other_push_index == self_push_index:
                    continue
                if not (0 <= other_push_index < len(indptr)):
                    continue
        # for cycle_offset in range(-cycle_tolerance, cycle_tolerance + 1):
        #     for other_connection_index in connections[connection_start: connection_end]:
        #         other_push_index = push_offset + other_connection_index + len_cycle * cycle_offset
        #         if other_push_index <= self_push_index:
        #             continue
        #         if other_push_index >= len(indptr):
        #             continue
                other_start = indptr[other_push_index]
                other_end = indptr[other_push_index + 1]
                if other_start == other_end:
                    continue
                self_index = self_start
                other_index = other_start
                while (self_index < self_end) and (other_index < other_end):
                    self_tof = tof_indices[self_index]
                    other_tof = tof_indices[other_index]
                    if (self_tof - tof_tolerance) <= other_tof <= (self_tof + tof_tolerance):
                        self_intensity = smooth_intensity_values[self_index]
                        other_intensity = smooth_intensity_values[other_index]
                        if self_intensity <= other_intensity:
                            if clusters[self_index] != clusters[other_index]:
                                unique_peaks[self_index] = False
                        if self_intensity >= other_intensity:
                            if clusters[self_index] != clusters[other_index]:
                                unique_peaks[other_index] = False
                    if self_tof < other_tof:
                        self_index += 1
                    else:
                        other_index += 1


@alphatims.utils.njit
def walk_unique_cluster_path(
    cluster_pointers,
    nonambiguous_elements,
    to_visit
):
    for index, nonambiguous in enumerate(nonambiguous_elements):
        initial_index = index
        path_length = 0
        while nonambiguous:
            path_length += 1
            if not to_visit[index]:
                break
            else:
                to_visit[index] = False
            pointer = cluster_pointers[index]
            if index == pointer:
                break
            index = pointer
            nonambiguous = nonambiguous_elements[index]
        if not nonambiguous:
            index = initial_index
            for i in range(path_length):
                nonambiguous_elements[index] = False
                index = cluster_pointers[index]


@alphatims.utils.njit
def assemble_clusters(
    cluster_pointers,
    nonambiguous_ions,
    cluster_assemblies,
):
    for index, pointer in enumerate(cluster_pointers):
        if nonambiguous_ions[index]:
            if index != pointer:
                secondary_pointer = cluster_assemblies[pointer]
                cluster_assemblies[index] = secondary_pointer
                cluster_assemblies[pointer] = index
