# native imports

# alphadia imports

# alpha family imports

# third party imports
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def group_and_parsimony(
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
    psm: pd.DataFrame,
    genes_or_proteins: str = "proteins",
    decoy_column: str = "decoy",
    group: bool = True,
    return_parsimony_groups: bool = False,
):
    """Highest level function for grouping proteins in precursor table

    Parameters
    ----------
    psm : pd.DataFrame
        Precursor table with columns "precursor_idx" and protein & decoy columns.
    gene_or_protein : str
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
    duplicate_mask = ~psm.duplicated(subset=["precursor_idx"], keep="first")

    # make sure column is string and subset to relevant columns
    psm[genes_or_proteins] = psm[genes_or_proteins].astype(str)
    upsm = psm.loc[duplicate_mask, ["precursor_idx", genes_or_proteins, decoy_column]]

    # check if duplicate precursors exist
    # TODO: consider removing check for duplicates since duplicate masking is implemented above
    if upsm.duplicated(subset=["precursor_idx"]).any():
        raise ValueError(
            """The same precursor was found annotated to different proteins.
            Please make sure all precursors were searched with the same library."""
        )

    # greedy set cover on all proteins if there is only one decoy class
    unique_decoys = upsm[decoy_column].unique()
    if len(unique_decoys) == 1:
        upsm[decoy_column] = -1
        upsm["pg_master"], upsm["pg"] = group_and_parsimony(
            upsm.precursor_idx.values,
            upsm[genes_or_proteins].values,
            return_parsimony_groups,
        )
        upsm = upsm[["precursor_idx", "pg_master", "pg", genes_or_proteins]]
    else:
        # handle case with multiple decoy classes
        target_mask = upsm[decoy_column] == 0
        decoy_mask = upsm[decoy_column] == 1

        # greedy set cover on targets
        t_df = upsm[target_mask].copy()
        # TODO: consider directly assigning to t_df["pg_master"], t_df["pg"] = group_and_parsimony(...)
        new_columns = group_and_parsimony(
            t_df.precursor_idx.values,
            t_df[genes_or_proteins].values,
            return_parsimony_groups,
        )
        t_df["pg_master"], t_df["pg"] = new_columns

        # greedy set cover on decoys
        d_df = upsm[decoy_mask].copy()
        new_columns = group_and_parsimony(
            d_df.precursor_idx.values,
            d_df[genes_or_proteins].values,
            return_parsimony_groups,
        )
        d_df["pg_master"], d_df["pg"] = new_columns

        upsm = pd.concat([t_df, d_df])[
            ["precursor_idx", "pg_master", "pg", genes_or_proteins]
        ]

    # heuristic grouping: from each initial precursor's protein ID set, filter out proteins that
    # are never master proteins
    if group:
        # select all master protein groups, which are the first in the semicolon separated list
        allowed_pg = upsm["pg"].str.split(";", expand=True)[0].unique()
        allowed_set_pg = set(allowed_pg)

        def filter_allowed_pg(pg):
            pg_set = set(pg.split(";")) & allowed_set_pg
            pg_list = list(pg_set)
            pg_list.sort()

            return ";".join(pg_list)

        upsm["pg"] = upsm[genes_or_proteins].apply(filter_allowed_pg)

    upsm.drop(columns=[genes_or_proteins], inplace=True)

    return psm.merge(upsm, on="precursor_idx", how="left")
