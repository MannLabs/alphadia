import logging
from functools import reduce

from alphabase.constants.modification import MOD_DF
from alphabase.spectral_library.base import SpecLibBase

from alphadia.libtransform.base import ProcessingStep

logger = logging.getLogger()


class MultiplexLibrary(ProcessingStep):
    def __init__(self, multiplex_mapping: list, input_channel: str | int | None = None):
        """Initialize the MultiplexLibrary step."""
        self._multiplex_mapping = self._create_multiplex_mapping(multiplex_mapping)
        self._input_channel = input_channel

    @staticmethod
    def _create_multiplex_mapping(multiplex_mapping: list) -> dict:
        """Create a dictionary from the multiplex mapping list."""
        mapping = {}
        for list_item in multiplex_mapping:
            mapping[list_item["channel_name"]] = list_item["modifications"]
        return mapping

    def validate(self, input: str) -> bool:
        """Validate the input object. It is expected that the input is a path to a file which exists."""
        valid = True
        valid &= isinstance(input, SpecLibBase)

        # check if all modifications are valid
        for _, channel_multiplex_mapping in self._multiplex_mapping.items():
            for key, value in channel_multiplex_mapping.items():
                for mod in [key, value]:
                    if mod not in MOD_DF.index:
                        logger.error(f"Modification {mod} not found in input library")
                        valid = False

        if "channel" in input.precursor_df.columns:
            channel_unique = input.precursor_df["channel"].unique()
            if self._input_channel not in channel_unique:
                logger.error(
                    f"Input library does not contain channel {self._input_channel}"
                )
                valid = False

            if (len(channel_unique) > 1) and (self._input_channel is None):
                logger.error(
                    f"Input library contains multiple channels {channel_unique}. Please specify a channel."
                )
                valid = False

        return valid

    def forward(self, input: SpecLibBase) -> SpecLibBase:
        """Apply the MultiplexLibrary step to the input object."""
        if "channel" in input.precursor_df.columns:
            input.precursor_df = input.precursor_df[
                input.precursor_df["channel"] == self._input_channel
            ]

        channel_lib_list = []
        for channel, channel_mod_translations in self._multiplex_mapping.items():
            logger.info(f"Multiplexing library for channel {channel}")
            channel_lib = input.copy()
            for original_mod, channel_mod in channel_mod_translations.items():
                channel_lib._precursor_df["mods"] = channel_lib._precursor_df[
                    "mods"
                ].str.replace(original_mod, channel_mod)
                channel_lib._precursor_df["channel"] = channel

            channel_lib.calc_fragment_mz_df()
            channel_lib_list.append(channel_lib)

        def apply_func(x, y):
            x.append(y)
            return x

        speclib = reduce(lambda x, y: apply_func(x, y), channel_lib_list)
        speclib.remove_unused_fragments()
        return speclib
