"""Trace a fixed set of precursors through candidate selection and the final FDR round.

Experiment exp/fc2-gapdump, not for merging: where does alphaDIA lose the peptides DIA-NN 2.7 reports (first pass) on
hela1ft_entr, helaft_5ng_2min_entr and helaft_200ng_2min_entr, by any charge state? The full dumps are too large to
read over the share, so only the traced elution groups are written: every group holding a precursor of `gap_keys.parquet`, plus every SAMPLE_GROUP_MODULUS-th elution group as a
background (false targets and decoys).
"""

from pathlib import Path

import numpy as np
import pandas as pd

GAP_KEYS_PATH = Path(__file__).with_name("gap_keys.parquet")
SAMPLE_GROUP_MODULUS = 50
# alphaDIA modification name -> UniMod id, as in the keys (DIA-NN Modified.Sequence)
UNIMOD = {
    "Carbamidomethyl@C": "4",
    "Oxidation@M": "35",
    "Acetyl@Protein_N-term": "1",
}
LIBRARY_COLUMNS = [
    "precursor_idx",
    "elution_group_idx",
    "decoy",
    "sequence",
    "mods",
    "mod_sites",
    "charge",
    "proteins",
    "rt_library",
    "mz_library",
]


def precursor_key(sequence: str, mods: str, mod_sites: str, charge: int) -> str:
    """sequence|site:unimod,...|charge with the sites sorted as strings, site 0 = N-term."""
    sites = []
    if mods:
        for mod, site in zip(mods.split(";"), mod_sites.split(";"), strict=True):
            sites.append(f"{site}:{UNIMOD.get(mod, mod)}")
    return f"{sequence}|{','.join(sorted(sites))}|{charge}"


class GapTrace:
    """Holds the traced precursor_idx and the frames recorded during the final FDR round."""

    def __init__(self) -> None:
        self.precursor_idx: np.ndarray | None = None
        self.stages: dict[str, pd.DataFrame] = {}

    def init(self, precursor_df: pd.DataFrame) -> pd.DataFrame:
        """Pick the traced precursors and return their library rows, with the key columns of gap_keys.parquet."""
        keys = pd.read_parquet(GAP_KEYS_PATH).set_index("key")
        targets = precursor_df[precursor_df["decoy"] == 0]
        target_keys = pd.Series(
            [
                precursor_key(s, m, ms, c)
                for s, m, ms, c in zip(
                    targets["sequence"],
                    targets["mods"].fillna(""),
                    targets["mod_sites"].fillna(""),
                    targets["charge"],
                    strict=True,
                )
            ],
            index=targets.index,
        )
        matched = target_keys[target_keys.isin(keys.index)]
        groups = set(precursor_df.loc[matched.index, "elution_group_idx"])
        in_trace = precursor_df["elution_group_idx"].isin(groups) | (
            precursor_df["elution_group_idx"] % SAMPLE_GROUP_MODULUS == 0
        )
        library = precursor_df.loc[
            in_trace, [c for c in LIBRARY_COLUMNS if c in precursor_df]
        ].copy()
        library["key"] = target_keys.reindex(library.index)
        library = library.join(keys, on="key")
        self.precursor_idx = library["precursor_idx"].to_numpy()
        self.stages = {}
        return library

    def subset(
        self, df: pd.DataFrame, columns: list[str] | None = None
    ) -> pd.DataFrame:
        sub = df[df["precursor_idx"].isin(self.precursor_idx)]
        if columns is not None:
            sub = sub[[c for c in columns if c in sub]]
        return sub.reset_index(drop=True)

    def record(
        self, stage: str, df: pd.DataFrame, columns: list[str] | None = None
    ) -> None:
        """Keep the traced rows of `df` as `stage`; a later call with the same stage replaces them."""
        if self.precursor_idx is None:
            return
        self.stages[stage] = self.subset(df, columns)

    def write(self, directory: str) -> None:
        if self.precursor_idx is None:
            return
        for stage, df in self.stages.items():
            df.to_parquet(Path(directory) / f"gap_{stage}.parquet", index=False)


TRACE = GapTrace()
