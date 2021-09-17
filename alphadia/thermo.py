
import alphatims.bruker
import logging
import numpy as np
import pandas as pd
import os


def load_thermo_raw(
    raw_file_name: str,
    profile: bool = False
) -> tuple:
    """Load raw thermo data as a dictionary.
​
    Args:
        raw_file_name (str): The name of a Thermo .raw file.
        n_most_abundant (int): The maximum number of peaks to retain per MS2 spectrum.
        use_profile_ms1 (bool): Use profile data or centroid it beforehand. Defaults to False.
        callback (callable): A function that accepts a float between 0 and 1 as progress. Defaults to None.
​
    Returns:
        tuple: A dictionary with all the raw data and a string with the acquisition_date_time
​
    """
    import alphapept.pyrawfilereader
    import tqdm
    rawfile = alphapept.pyrawfilereader.RawFileReader(raw_file_name)
    _push_indices = []
    mz_values = []
    intensity_values = []
    rt_values = []
    quad_mz_values = []
    precursor_indices = []
    for i in tqdm.tqdm(
        range(
            rawfile.FirstSpectrumNumber,
            rawfile.LastSpectrumNumber + 1
        )
    ):
        if profile:
            masses, intensities = rawfile.GetProfileMassListFromScanNum(i)
        else:
            masses, intensities = rawfile.GetCentroidMassListFromScanNum(i)
        mz_values.append(masses)
        intensity_values.append(intensities)
        _push_indices.append(len(masses))
        rt = rawfile.RTFromScanNum(i)
        rt_values.append(rt)
        ms_order = rawfile.GetMSOrderForScanNum(i)
        if ms_order == 1:
            precursor = 0
            quad_mz_values.append((-1, -1))
        elif ms_order == 2:
            precursor += 1
            isolation_center = rawfile.GetPrecursorMassForScanNum(i)
            DIA_width = rawfile.GetIsolationWidthForScanNum(i)
            quad_mz_values.append(
                (
                    isolation_center - DIA_width / 2,
                    isolation_center + DIA_width / 2,
                )
            )
        precursor_indices.append(precursor)
    rawfile.Close()
    push_indices = np.empty(rawfile.LastSpectrumNumber + 1, np.int64)
    push_indices[0] = 0
    push_indices[1:] = np.cumsum(_push_indices)
    return (
        push_indices,
        np.concatenate(mz_values),
        np.concatenate(intensity_values),
        np.array(rt_values) * 60,
        np.array(quad_mz_values),
        np.array(precursor_indices),
    )


class RawFile(alphatims.bruker.TimsTOF):
    def __init__(
        self,
        thermo_raw_file_name: str,
        slice_as_dataframe: bool = True
    ):
        """Create a Bruker Orbitrap object that contains all data in-memory.
​
        Parameters
        ----------
        thermo_raw_file_name : str
            The full file name to a Bruker .d folder.
            Alternatively, the full file name of an already exported .hdf
            can be provided as well.
        slice_as_dataframe : bool
            If True, slicing returns a pd.DataFrame by default.
            If False, slicing provides a np.int64[:] with raw indices.
            This value can also be modified after creation.
            Default is True.
        """
        self._use_calibrated_mz_values_as_default = False
        self.thermo_raw_file_name = os.path.abspath(thermo_raw_file_name)
        logging.info(f"Importing data from {thermo_raw_file_name}")
        if thermo_raw_file_name.endswith(".raw"):
            self._import_data_from_raw_file(
                thermo_raw_file_name,
            )
        elif thermo_raw_file_name.endswith(".hdf"):
            self._import_data_from_hdf_file(
                thermo_raw_file_name,
            )
            self.thermo_raw_file_name = os.path.abspath(thermo_raw_file_name)
        self.bruker_d_folder_name = self.thermo_raw_file_name
        if not hasattr(self, "version"):
            self._version = "none"
        if self.version != alphatims.__version__:
            logging.info(
                "WARNING: "
                f"AlphaTims version {self.version} was used to initialize "
                f"{thermo_raw_file_name}, while the current version of "
                f"AlphaTims is {alphatims.__version__}."
            )
        logging.info(f"Succesfully imported data from {thermo_raw_file_name}")
        self.slice_as_dataframe = slice_as_dataframe
        # Precompile
        self[0, "raw"]

    def _import_data_from_raw_file(
        self,
        thermo_raw_file_name: str,
    ):
        self._version = alphatims.__version__
        (
            self._push_indptr,
            mz_values,
            self._intensity_values,
            self._rt_values,
            self._quad_mz_values,
            self._precursor_indices,
        ) = load_thermo_raw(thermo_raw_file_name)
        self.thermo_raw_file_name = thermo_raw_file_name
        scan_count = len(self._precursor_indices)
        self._frame_max_index = scan_count
        self._scan_max_index = 1
        self._mobility_max_value = 0
        self._mobility_min_value = 0
        self._mobility_values = np.array([0])
        self._quad_indptr = self._push_indptr
        self._raw_quad_indptr = np.arange(scan_count + 1)
        self._intensity_min_value = float(np.min(self._intensity_values))
        self._intensity_max_value = float(np.max(self._intensity_values))
        self._quad_min_mz_value = float(
            np.min(
                self._quad_mz_values[self._quad_mz_values != -1]
            )
        )
        self._quad_max_mz_value = float(np.max(self._quad_mz_values))
        self._precursor_max_index = int(np.max(self._precursor_indices)) + 1
        self._acquisition_mode = "ddaPASEF" # TODO
        self._mz_min_value = int(np.min(mz_values))
        self._mz_max_value = int(np.max(mz_values)) + 1
        self._decimals = 4
        self._mz_values = np.arange(
            10**self._decimals * self._mz_min_value,
            10**self._decimals * (self._mz_max_value + 1)
        ) / 10**self._decimals
        self._tof_indices = (
            mz_values * 10**self._decimals
        ).astype(np.int32) - 10**self._decimals * self._mz_min_value
        self._tof_max_index = len(self._mz_values)
        self._meta_data = {
            "SampleName": thermo_raw_file_name
        }
        msmstype = np.array(
            [0 if s == -1 else 1 for s, e in self._quad_mz_values]
        )
        summed_intensities_ = np.cumsum(self._intensity_values)
        summed_intensities = -summed_intensities_[self._push_indptr[:-1]]
        summed_intensities[:-1] += summed_intensities_[self._push_indptr[1:-1]]
        summed_intensities[-1] += summed_intensities_[-1]
        self._frames = pd.DataFrame(
            {
                'MsMsType': msmstype,
                'Time': self._rt_values,
                'SummedIntensities': summed_intensities,
                'Id': np.arange(len(self._rt_values)),
            }
        )
        self._zeroth_frame = False
        offset = int(self.zeroth_frame)
        cycle_index = np.searchsorted(
            self.raw_quad_indptr,
            (self.scan_max_index) * (self.precursor_max_index + offset),
            "r"
        ) + 1
        repeats = np.diff(self.raw_quad_indptr[: cycle_index])
        if self.zeroth_frame:
            repeats[0] -= self.scan_max_index
        cycle_length = self.scan_max_index * self.precursor_max_index
        repeat_length = np.sum(repeats)
        if repeat_length != cycle_length:
            repeats[-1] -= repeat_length - cycle_length
        self._dia_mz_cycle = np.empty((cycle_length, 2))
        self._dia_mz_cycle[:, 0] = np.repeat(
            self.quad_mz_values[: cycle_index - 1, 0],
            repeats
        )
        self._dia_mz_cycle[:, 1] = np.repeat(
            self.quad_mz_values[: cycle_index - 1, 1],
            repeats
        )
        self._dia_precursor_cycle = np.repeat(
            self.precursor_indices[: cycle_index - 1],
            repeats
        )
