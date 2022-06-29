import math
import numpy as np
import pandas as pd
import typing as t


# TODO: Implement levelutils here - renamed to states to refer to ExoMol states files specifically.


def read_mvl_energies(file: str, qn_cols: t.List[str], energy_cols: t.List[str] = None) -> pd.DataFrame:
    if energy_cols is None:
        energy_cols = ['energy', 'unc', 'degree']
    mvl_energy_cols = qn_cols + energy_cols
    return pd.read_csv(file, sep=r'\s+', names=mvl_energy_cols)


def propagate_error_in_mean(unc_list: list) -> float:
    unc_sq_list = [unc ** 2 for unc in unc_list]
    sum_unc_sq = sum(unc_sq_list)
    sqrt_sum_unc_sq = math.sqrt(sum_unc_sq)
    error = sqrt_sum_unc_sq / len(unc_list)
    return error


def match_levels(
        levels_initial: pd.DataFrame,
        levels_new: pd.DataFrame,
        qn_match_cols: list,
        levels_new_qn_cols: list,
        match_source_tag: str,
        shift_table_qn_cols: list,
        suffixes: t.Tuple = None,
        energy_col_name: str = None,
        is_isotopologue_match: bool = False,
        overwrite_non_match_qn_cols: bool = False
) -> t.Tuple[pd.DataFrame, pd.DataFrame]:
    if suffixes is None:
        suffixes = ('_calc', '_obs')

    if energy_col_name is None:
        energy_col_name = 'energy'

    # Take an inner merge to only get the levels that do have matches.
    levels_matched = levels_initial.merge(levels_new, left_on=qn_match_cols, right_on=qn_match_cols,
                                          suffixes=suffixes, how='inner')
    print('LEVELS MATCHED: \n', levels_matched)
    # TODO: Handling for empty levels_matched DataFrame when no matches between inputs.

    energy_original_col = energy_col_name + suffixes[0]
    energy_marvel_col = energy_col_name + suffixes[1]
    energy_dif_col = energy_col_name + '_dif'
    energy_dif_mag_col = energy_dif_col + '_mag'

    levels_matched[energy_dif_col] = levels_matched.apply(lambda x: x[energy_marvel_col] - x[energy_original_col],
                                                          axis=1)
    levels_matched[energy_dif_mag_col] = levels_matched[energy_dif_col].apply(lambda x: abs(x))

    # This was bad as it changed the structure of the output DataFrame to eb that of the GroupBy frame.
    # levels_matched = levels_matched.sort_values(energy_dif_mag_col).groupby(by=qn_match_cols, as_index=False).first()

    qn_cols_not_match = np.setdiff1d(levels_new_qn_cols, qn_match_cols)
    # If not matching on the full set of quantum numbers that uniquely determine a level, check for duplicates such that
    # no more than one level (defined by the input set of quantum numbers qn_match_cols) is matching on the same
    # levels_new level.
    qn_dupe_cols = qn_match_cols if len(qn_cols_not_match) == 0 else qn_match_cols + [energy_marvel_col]

    levels_matched_dup = levels_matched[levels_matched.duplicated(subset=qn_dupe_cols, keep=False)].sort_values(
        qn_dupe_cols + [energy_dif_mag_col])

    if len(levels_matched_dup) > 0:
        # Get the index of the lowest energy_agreement entry, for each tag which has duplicates
        levels_matched_dup_idx_min = levels_matched_dup.groupby(
            by=qn_dupe_cols, sort=False)[energy_dif_mag_col].transform(min) == levels_matched_dup[energy_dif_mag_col]
        # Take the index of everything other than the lowest energy_agreement entry for each duplicated tag and remove
        # it from the comparison dataframe
        levels_matched = levels_matched.drop(levels_matched_dup[~levels_matched_dup_idx_min].index)

    # TODO: TEST THIS ALTERNATIVE. Explicit ascending?
    # levels_matched = levels_matched.sort_values(by=[qn_dupe_cols + [energy_dif_mag_col]]).drop_duplicates(
    #     subset=qn_dupe_cols, keep='first')

    # Remove the energy difference magnitude column as it is not needed beyond this point.
    del levels_matched[energy_dif_mag_col]

    # Check the 0 energy level.
    zero_energy_level_matches = len(levels_matched.loc[(levels_matched[energy_original_col] == 0)
                                                       & (levels_matched[energy_marvel_col] == 0)])
    if zero_energy_level_matches != 1:
        # TODO: change to raise an error.
        print('0 ENERGY LEVELS DO NOT MATCH ASSIGNMENTS IN BOTH DATASETS.\nORIGINAL:\n',
              levels_matched.loc[levels_matched[energy_original_col] == 0], '\nUPDATE:\n',
              levels_matched.loc[levels_matched[energy_marvel_col] == 0])

    if not is_isotopologue_match:
        # Merge the sets of qn_match_cols in each DataFrame with an indicator to find any rows that are only in
        # levels_new to concat them to levels_matched.
        levels_new_check_matched = levels_new[qn_match_cols].merge(levels_matched[qn_match_cols], how='left',
                                                                   indicator=True)
        # Get the indexes of levels_new where the unique qn_match_cols are not yet in levels_matched.
        levels_new_to_concat_idx = levels_new_check_matched[
            (levels_new_check_matched['_merge'] == 'left_only')].index
        # Isolate only the rows with those indexes to be concatenated.
        levels_new_to_concat = levels_new[levels_new.index.isin(levels_new_to_concat_idx)]
        # Rename energy_col_name in the rows to be concatenated to energy_marvel_col to avoid creating a new column.
        levels_new_to_concat = levels_new_to_concat.rename(columns={energy_col_name: energy_marvel_col})
        if len(qn_cols_not_match) > 0:
            levels_new_to_concat = levels_new_to_concat.rename(
                columns={qn_col: qn_col + suffixes[1] for qn_col in qn_cols_not_match})
        # levels_matched = levels_matched.append(levels_new_to_append)
        levels_matched = pd.concat([levels_matched, levels_new_to_concat])

    levels_matched['source_tag'] = match_source_tag
    levels_matched['energy_final'] = levels_matched[energy_marvel_col]

    # Create table to provide energy shifts and std (for unc estimates) based on a qn grouping.
    shift_table = levels_matched.loc[(~levels_matched[energy_original_col].isna())
                                     & (~levels_matched[energy_marvel_col].isna())].groupby(
        by=shift_table_qn_cols, as_index=False).apply(
        lambda x: pd.Series({
            'energy_dif_mean': np.average(x[energy_dif_col]),
            'energy_dif_unc': propagate_error_in_mean(x['unc'])
        }))
    print('SHIFT TABLE: \n', shift_table)

    # Rename original levels' energy to match the column name in levels_matched
    levels_initial = levels_initial.rename(columns={'energy': energy_original_col})
    # Add missing original levels that are not in the final matching set
    # levels_matched = levels_matched.append(levels_initial.loc[~levels_initial['id'].isin(levels_matched['id'])])
    levels_initial_to_concat = levels_initial.loc[~levels_initial['id'].isin(levels_matched['id'])]
    if len(qn_cols_not_match) > 0:
        levels_initial_to_concat = levels_initial_to_concat.rename(
            columns={qn_col: qn_col + suffixes[0] for qn_col in qn_cols_not_match})

    if 'unc' in levels_initial_to_concat.columns and 'unc' + suffixes[0] in levels_matched.columns:
        levels_initial_to_concat = levels_initial_to_concat.rename(columns={'unc': 'unc' + suffixes[0]})

    levels_matched = pd.concat([levels_matched, levels_initial_to_concat])

    if overwrite_non_match_qn_cols and len(qn_cols_not_match) > 0:
        for qn_col_not_match in qn_cols_not_match:
            levels_matched[qn_col_not_match] = np.where(levels_matched[qn_col_not_match + suffixes[1]].isna(),
                                                        levels_matched[qn_col_not_match + suffixes[0]],
                                                        levels_matched[qn_col_not_match + suffixes[1]])
            del levels_matched[qn_col_not_match + suffixes[0]]
            del levels_matched[qn_col_not_match + suffixes[1]]

    return levels_matched, shift_table
