from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from alphadia.constants.keys import PsmDfCols


def _group_and_parsimony(
    precursor_idx: NDArray[np.int64],
    precursor_ids: NDArray[Any],
    return_parsimony_groups: bool = False,
):
    """Function to group ids based on precursor indices and return groups & master ids as lists

    Parameters
    ----------
    precursor_idx : np.array[int]
        Array containing unique integer indices corresponding to each peptide precursor
    precursor_ids : np.array[str]
        Array of variable length semicolon separated str belonging to a given peptide precursor id

    Returns
    -------
    tuple
        Tuple containing two lists: ids and groups. Each list is ordered by precursor_idx

    """

    # reshape precursor indices and ids into a dictionary of ids linked to sets of precursors
    id_dict = {}
    for precursor, ids in zip(precursor_idx, precursor_ids, strict=True):
        for id in ids.split(";"):
            if id not in id_dict:
                id_dict[id] = set()
            id_dict[id].add(precursor)

    # perform greedy set cover on protein_dict: if two ids share a precursor, the precorsor is deleted from the id with fewer precursors. If this removes all precursors from an id, it is added to the larger id's group.
    id_group = []
    id_master = []
    precursor_set = []

    # loop bounds max iterations
    for _ in range(len(id_dict)):
        # remove longest set from dict as query & remove query peptide from all other sets
        query_id = max(id_dict.keys(), key=lambda x: len(id_dict[x]))
        query_peptides = id_dict.pop(query_id)
        query_group = [query_id]

        # break if query is empty. Sorting step means that all remaining sets are empty
        if len(query_peptides) == 0:
            break

        # update peptides & add to group if depleted
        for subject_protein, subject_peptides in id_dict.items():
            if not subject_peptides:
                continue
            new_subject_set = subject_peptides - query_peptides
            id_dict[subject_protein] = new_subject_set
            # With the following lines commented out, the query will only eliminate peptides from
            # respective subject proteins, but we will not add them to the query group
            if return_parsimony_groups and len(new_subject_set) == 0:
                query_group.append(subject_protein)

        # save query to output lists
        id_group.append(query_group)
        id_master.append(query_id)
        precursor_set.append(query_peptides)

    # convert id_group sublists to semicolon separated strings
    id_group = [";".join(x) for x in id_group]

    # reshape output data and align with precursor dataframe input. Use dictionary for efficient ordering
    # TODO consider iterating over precursor_idx directly
    return_dict = {}
    for i, peptide_set in enumerate(precursor_set):
        for key in peptide_set:
            return_dict[key] = (id_master[i], id_group[i])

    # check that all precursors are found again
    if len(return_dict) != len(precursor_idx):
        raise ValueError(
            f"""Not all precursors were found in the output of the grouping function. {len(return_dict)} precursors were found, but {len(precursor_idx)} were expected."""
        )

    # check that all return_dict keys are unique. Assume same length and unique keys constitutes match to precursor_idx
    if len(return_dict) != len(set(return_dict.keys())):
        raise ValueError(
            """Not all precursors were found in the output of the grouping function.
            Duplicate precursors were found."""
        )

    # order by precursor index and return as lists
    # TODO look above, order by precursor_idx directly?
    return_dict_ordered = {key: return_dict[key] for key in precursor_idx}
    ids, groups = zip(*return_dict_ordered.values(), strict=True)

    return ids, groups


def perform_grouping(
    psm_df: pd.DataFrame,
    genes_or_proteins: str = "proteins",
    decoy_column: str = "decoy",
    group: bool = True,
    return_parsimony_groups: bool = False,
):
    """Highest level function for grouping proteins in precursor table

    Parameters
    ----------
    psm_df : pd.DataFrame
        Precursor table with columns "precursor_idx" and protein & decoy columns.
    genes_or_proteins : str
        Column to group proteins by. Defaults to "proteins".
    decoy_column : str
        Column to use for decoy annotation. Defaults to "decoy".
    group : bool
        Whether to group proteins. Defaults to True.

    Returns
    -------
    pd.DataFrame :
        Precursor table with grouped proteins

    """

    if genes_or_proteins not in ["genes", "proteins"]:
        raise ValueError("Selected column must be 'genes' or 'proteins'")

    # create non-duplicated view of precursor table
    unique_mask = ~psm_df.duplicated(subset=[PsmDfCols.PRECURSOR_IDX], keep="first")

    # make sure column is string and subset to relevant columns
    psm_df[genes_or_proteins] = psm_df[genes_or_proteins].astype(str)
    unique_psm_df = psm_df.loc[
        unique_mask, [PsmDfCols.PRECURSOR_IDX, genes_or_proteins, decoy_column]
    ]

    # greedy set cover on all proteins if there is only one decoy class
    unique_decoys = unique_psm_df[decoy_column].unique()
    if len(unique_decoys) == 1:
        unique_psm_df[decoy_column] = -1
        unique_psm_df["pg_master"], unique_psm_df["pg"] = _group_and_parsimony(
            unique_psm_df[PsmDfCols.PRECURSOR_IDX].values,
            unique_psm_df[genes_or_proteins].values,
            return_parsimony_groups,
        )
        unique_psm_df = unique_psm_df[
            [PsmDfCols.PRECURSOR_IDX, "pg_master", "pg", genes_or_proteins]
        ]
    else:
        # handle case with multiple decoy classes
        target_mask = unique_psm_df[decoy_column] == 0
        decoy_mask = unique_psm_df[decoy_column] == 1

        # greedy set cover on targets
        target_df = unique_psm_df[target_mask].copy()
        target_df["pg_master"], target_df["pg"] = _group_and_parsimony(
            target_df[PsmDfCols.PRECURSOR_IDX].values,
            target_df[genes_or_proteins].values,
            return_parsimony_groups,
        )

        # greedy set cover on decoys
        decoy_df = unique_psm_df[decoy_mask].copy()
        decoy_df["pg_master"], decoy_df["pg"] = _group_and_parsimony(
            decoy_df[PsmDfCols.PRECURSOR_IDX].values,
            decoy_df[genes_or_proteins].values,
            return_parsimony_groups,
        )

        unique_psm_df = pd.concat([target_df, decoy_df])[
            [PsmDfCols.PRECURSOR_IDX, "pg_master", "pg", genes_or_proteins]
        ]

    # heuristic grouping: from each initial precursor's protein ID set, filter out proteins that
    # are never master proteins
    if group:
        # select all master protein groups, which are the first in the semicolon separated list
        allowed_pg = unique_psm_df["pg"].str.split(";", expand=True)[0].unique()
        allowed_set_pg = set(allowed_pg)

        def filter_allowed_pg(pg):
            pg_set = set(pg.split(";")) & allowed_set_pg
            pg_list = list(pg_set)
            pg_list.sort()

            return ";".join(pg_list)

        unique_psm_df["pg"] = unique_psm_df[genes_or_proteins].apply(filter_allowed_pg)

    unique_psm_df.drop(columns=[genes_or_proteins], inplace=True)

    return psm_df.merge(unique_psm_df, on=PsmDfCols.PRECURSOR_IDX, how="left")
