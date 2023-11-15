# native imports

# alphadia imports

# alpha family imports

# third party imports
import numpy as np
import pandas as pd
from typing import List, Any
from numpy.typing import NDArray


def group_and_parsimony(
    precursor_idx: NDArray[np.int64],
    precursor_ids: NDArray[Any],
):
    """
    Function to group ids based on precursor indices and return groups & master ids as lists

    Parameters
    ----------

        precursor_idx : np.array[int]
            array containing unique integer indices corresponding to each peptide precursor

        precursor_ids : np.array[str]
            array of variable length semicolon separated str belonging to a given peptide precursor id

    Returns
    -------

        ids : list[str]
            list of ids linked to a given peptide precursor, such that each precursor only belongs to one id. This list is ordered by precursor_idx.

        groups : list[str]
            list of semicolon separated ids belonging to a given peptide precursor, such that each precursor only belongs to one group. This list is ordered by precursor_idx.

    """

    # reshape precursor indices and ids into a dictionary of ids linked to sets of precursors
    id_dict = {}
    for precursor, ids in zip(precursor_idx, precursor_ids):
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
        # remove longest set from dict as query & remove query peptided from all other sets
        query_id = max(id_dict.keys(), key=lambda x: len(id_dict[x]))
        query_peptides = id_dict.pop(query_id)
        query_group = [query_id]

        if len(query_peptides) == 0:
            break

        # update peptides & add to group if depleted
        for subject_protein, subject_peptides in id_dict.items():
            if not subject_peptides:
                continue
            new_subject_set = subject_peptides - query_peptides
            id_dict[subject_protein] = new_subject_set
            if len(new_subject_set) == 0:
                query_group.append(subject_protein)

        # save query to output lists
        id_group.append(query_group)
        id_master.append(query_id)
        precursor_set.append(query_peptides)

    # convert id_group sublists to semicolon separated strings
    id_group = [";".join(x) for x in id_group]

    # reshape output data and align with precursor dataframe input. Use dictionary for efficient ordering
    return_dict = {}
    for i, peptide_set in enumerate(precursor_set):
        for key in peptide_set:
            return_dict[key] = (id_master[i], id_group[i])

    # check that all precursors are found again
    if len(return_dict) != len(precursor_idx):
        raise ValueError(
            "Not all precursors were found in the output of the grouping function."
        )

    # order by precursor index
    return_dict_ordered = {key: return_dict[key] for key in precursor_idx}
    ids, groups = zip(*return_dict_ordered.values())

    return ids, groups


def perform_grouping(
    psm: pd.DataFrame,
    genes_or_proteins: str = "proteins",
):
    """Highest level function for grouping proteins in precursor table

    Parameters:
        gene_or_protein (str, optional): Column to group proteins by. Defaults to "proteins".

    """

    if genes_or_proteins not in ["genes", "proteins"]:
        raise ValueError("Selected column must be 'genes' or 'proteins'")

    # create non-duplicated view of precursor table
    duplicate_mask = ~psm.duplicated(
        subset=["precursor_idx", genes_or_proteins], keep="first"
    )
    upsm = psm.loc[duplicate_mask, ["precursor_idx", genes_or_proteins, "_decoy"]]

    # handle case with only one decoy class:
    unique_decoys = upsm["_decoy"].unique()
    if len(unique_decoys) == 1:
        upsm["_decoy"] = -1
        upsm["pg_master"], upsm["pg"] = group_and_parsimony(
            upsm.precursor_idx.values, upsm[genes_or_proteins].values
        )
        upsm = upsm[["precursor_idx", "pg_master", "pg"]]
    else:
        target_mask = upsm["_decoy"] == 0
        decoy_mask = upsm["_decoy"] == 1

        t_df = upsm[target_mask].copy()
        new_columns = group_and_parsimony(
            t_df.precursor_idx.values, t_df[genes_or_proteins].values
        )
        t_df["pg_master"], t_df["pg"] = new_columns

        d_df = upsm[decoy_mask].copy()
        new_columns = group_and_parsimony(
            d_df.precursor_idx.values, d_df[genes_or_proteins].values
        )
        d_df["pg_master"], d_df["pg"] = new_columns

        upsm = pd.concat([t_df, d_df])[["precursor_idx", "pg_master", "pg"]]

    return psm.merge(upsm, on="precursor_idx", how="left")
