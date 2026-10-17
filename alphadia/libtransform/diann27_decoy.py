"""Decoys built the way DIA-NN 2.7.0 builds them (scheme inferred from its exported decoys, default settings).

The first and the last residue are kept; residues 2 and n-1 always change:
1. Swap residues 2 and n-1, unless their mass difference is below `MIN_SHIFT` or inside a `SHIFT_WINDOWS` band.
2. Otherwise try a few seeded-random generalised swaps (a C-side residue moves to 2, an N-side one to n-1, the old
   residues 2 and n-1 take their places). Accept the first whose interior b/y ions all move by at least `MIN_SHIFT`
   and outside `SHIFT_WINDOWS`.
3. Otherwise put seeded-random residues (never C, weighted by the library's interior amino-acid frequency) at 2 and
   n-1, each and their sum shifting the mass by at least `MIN_SHIFT`.

Modifications move with their residue; a mutated residue loses its modification. The seed is the peptide itself, so
every charge state of a peptide gets the same decoy.

With `keep_outer_ions`, the scheme runs on residues 2 .. n-1 instead, so residues 3 and n-2 change and every decoy keeps
its target's b2 and y2. DIA-NN 2.7's mutation and swap rules leave the decoys' b2 / y2 with rarer residue compositions
than a false target's (fewer L-type residues), so they match fewer co-isolated fragments by chance.
"""

import multiprocessing as mp
import random
import zlib
from collections import Counter

import pandas as pd
from alphabase.constants.aa import AA_ASCII_MASS
from alphabase.constants.modification import MOD_MASS

MIN_SHIFT = 5.0
# DIA-NN never swaps residue pairs whose mass difference falls in these bands (Da); they coincide with common
# modification and neutral-loss masses. Their exact edges are unknown, only the differences seen in data.
SHIFT_WINDOWS = (
    (15.95, 18.04),
    (27.985, 28.05),
    (42.0, 43.025),
    (57.01, 57.07),
    (63.985, 64.005),
)
# Fitted number of generalised-swap attempts before mutating: 2.33 on average.
SHUFFLE_ATTEMPTS = 2
EXTRA_ATTEMPT_PROBABILITY = 0.33
MUTATION_AAS = "ADEFGHIKLMNPQRSTVWY"
MAX_MUTATION_DRAWS = 1000
MP_BATCH_SIZE = 20000


def _in_window(shift: float) -> bool:
    return any(low <= shift <= high for low, high in SHIFT_WINDOWS)


def _allowed(shift: float) -> bool:
    shift = abs(shift)
    return shift >= MIN_SHIFT and not _in_window(shift)


def _site_mods(
    mods: str, mod_sites: str, n: int
) -> tuple[list[list[str]], list[tuple[str, str]]]:
    """Residue modifications per position (0-based) and the terminal ones as (mod, site) pairs."""
    residue_mods = [[] for _ in range(n)]
    terminal = []
    if mods:
        for mod, site in zip(mods.split(";"), mod_sites.split(";"), strict=True):
            position = int(site)
            if 1 <= position <= n:
                residue_mods[position - 1].append(mod)
            else:
                terminal.append((mod, site))
    return residue_mods, terminal


def _interior_shifts_pass(target: list[float], decoy: list[float]) -> bool:
    n = len(target)
    diff = [d - t for t, d in zip(target, decoy, strict=True)]
    b_shift = diff[0]
    for k in range(1, n - 2):
        b_shift += diff[k]
        if not _allowed(b_shift):
            return False
    y_shift = diff[-1]
    for k in range(n - 2, 1, -1):
        y_shift += diff[k]
        if not _allowed(y_shift):
            return False
    return True


class DIANN27DecoyGenerator:
    """Picklable worker that turns (sequence, mods, mod_sites) tuples into decoy tuples."""

    def __init__(
        self, aa_weights: dict[str, float], keep_outer_ions: bool = False
    ) -> None:
        self.keep_outer_ions = keep_outer_ions
        self.mutation_aas = [aa for aa in MUTATION_AAS if aa_weights.get(aa, 0) > 0]
        self.mutation_weights = [aa_weights[aa] for aa in self.mutation_aas]

    def __call__(
        self, peptides: list[tuple[str, str, str]]
    ) -> list[tuple[str, str, str]]:
        return [self.decoy(*peptide) for peptide in peptides]

    def decoy(self, sequence: str, mods: str, mod_sites: str) -> tuple[str, str, str]:
        if not self.keep_outer_ions:
            return self._decoy(sequence, mods, mod_sites)
        n = len(sequence)
        outer, core = [], []
        if mods:
            for mod, site in zip(mods.split(";"), mod_sites.split(";"), strict=True):
                position = int(site)
                if 2 <= position <= n - 1:
                    core.append((mod, str(position - 1)))
                else:
                    outer.append((mod, site))
        core_seq, core_mods, core_sites = self._decoy(
            sequence[1:-1], ";".join(m for m, _ in core), ";".join(s for _, s in core)
        )
        if core_mods:
            outer += [
                (mod, str(int(site) + 1))
                for mod, site in zip(
                    core_mods.split(";"), core_sites.split(";"), strict=True
                )
            ]
        return (
            sequence[0] + core_seq + sequence[-1],
            ";".join(m for m, _ in outer),
            ";".join(s for _, s in outer),
        )

    def _decoy(self, sequence: str, mods: str, mod_sites: str) -> tuple[str, str, str]:
        n = len(sequence)
        if n < 4:
            return sequence, mods, mod_sites
        residues = list(sequence)
        residue_mods, terminal = _site_mods(mods, mod_sites, n)
        masses = [
            AA_ASCII_MASS[ord(aa)] + sum(MOD_MASS[mod] for mod in site)
            for aa, site in zip(residues, residue_mods, strict=True)
        ]
        rng = random.Random(zlib.crc32(f"{sequence}|{mods}|{mod_sites}".encode()))

        order = self._permutation(masses, rng)
        if order is not None:
            residues = [residues[i] for i in order]
            residue_mods = [residue_mods[i] for i in order]
        else:
            for position, aa in zip(
                (1, n - 2), self._mutations(masses, rng), strict=True
            ):
                residues[position] = aa
                residue_mods[position] = []

        site_pairs = terminal + [
            (mod, str(position + 1))
            for position, site in enumerate(residue_mods)
            for mod in site
        ]
        return (
            "".join(residues),
            ";".join(mod for mod, _ in site_pairs),
            ";".join(site for _, site in site_pairs),
        )

    def _permutation(self, masses: list[float], rng: random.Random) -> list[int] | None:
        n = len(masses)
        swap = list(range(n))
        swap[1], swap[n - 2] = n - 2, 1
        if _allowed(masses[1] - masses[n - 2]):
            return swap

        candidates = [
            (i, j, crossed)
            for i in range(2, n - 2)
            for j in range(n - 3, i, -1)
            for crossed in (False, True)
        ]
        attempts = SHUFFLE_ATTEMPTS + (rng.random() < EXTRA_ATTEMPT_PROBABILITY)
        for i, j, crossed in rng.sample(candidates, min(attempts, len(candidates))):
            order = list(range(n))
            order[1], order[n - 2] = j, i
            order[i], order[j] = (n - 2, 1) if crossed else (1, n - 2)
            if _interior_shifts_pass(masses, [masses[k] for k in order]):
                return order
        return None

    def _mutations(self, masses: list[float], rng: random.Random) -> tuple[str, str]:
        first, last = masses[1], masses[-2]
        for _ in range(MAX_MUTATION_DRAWS):
            a, b = rng.choices(self.mutation_aas, self.mutation_weights, k=2)
            shift_a = AA_ASCII_MASS[ord(a)] - first
            shift_b = AA_ASCII_MASS[ord(b)] - last
            if min(abs(shift_a), abs(shift_b), abs(shift_a + shift_b)) >= MIN_SHIFT:
                return a, b
        raise ValueError(f"No mutation found for residue masses {first} and {last}")


def interior_aa_weights(sequences: pd.Series) -> dict[str, float]:
    counts = Counter("".join(sequence[1:-1] for sequence in sequences))
    return {aa: counts.get(aa, 0) for aa in MUTATION_AAS}


def diann27_decoy_precursors(
    precursor_df: pd.DataFrame, mp_process_num: int = 8, keep_outer_ions: bool = False
) -> pd.DataFrame:
    """Return a copy of `precursor_df` with decoy sequences, mods and mod_sites; decoys equal to a target are dropped."""
    keys = ["sequence", "mods", "mod_sites"]
    peptides = precursor_df[keys].drop_duplicates()
    generator = DIANN27DecoyGenerator(
        interior_aa_weights(peptides["sequence"]), keep_outer_ions=keep_outer_ions
    )
    tuples = list(peptides.itertuples(index=False, name=None))

    if mp_process_num <= 1 or len(tuples) < MP_BATCH_SIZE:
        decoys = generator(tuples)
    else:
        batches = [
            tuples[i : i + MP_BATCH_SIZE] for i in range(0, len(tuples), MP_BATCH_SIZE)
        ]
        with mp.get_context("spawn").Pool(mp_process_num) as pool:
            decoys = [
                decoy for batch in pool.imap(generator, batches) for decoy in batch
            ]

    mapping = pd.DataFrame(decoys, columns=[f"decoy_{key}" for key in keys])
    mapping[keys] = peptides.to_numpy()
    decoy_df = precursor_df.merge(mapping, on=keys, how="left")
    decoy_df[keys] = decoy_df[[f"decoy_{key}" for key in keys]].to_numpy()
    decoy_df = decoy_df.drop(columns=[f"decoy_{key}" for key in keys])
    return decoy_df[
        ~decoy_df["sequence"].isin(set(precursor_df["sequence"]))
    ].reset_index(drop=True)
