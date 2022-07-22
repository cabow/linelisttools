import functools
import math
import typing as t
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler

from linelisttools.concurrence import ExecutorType, yield_grouped_data


class SourceTag(Enum):
    CALCULATED = "Ca"
    MARVELISED = "Ma"
    EFFECTIVE_HAMILTONIAN = "EH"
    PARITY_PAIR = "PS_1"
    PREDICTED_SHIFT = "PS_2"
    EXTRAPOLATED_SHIFT = "PS_3"
    PSEUDO_EXPERIMENTAL = "PE"


def read_mvl_energies(
    file: str, qn_cols: t.List[str], energy_cols: t.List[str] = None
) -> pd.DataFrame:
    if energy_cols is None:
        energy_cols = ["energy", "unc", "degree"]
    elif len(energy_cols) != 3:
        raise RuntimeError(
            "energy_cols argument must be  contain three values for the energy, uncertainty and degree columns."
        )
    mvl_energy_cols = qn_cols + energy_cols
    return pd.read_csv(file, sep=r"\s+", names=mvl_energy_cols)


def propagate_error_in_mean(unc_list: t.List[float]) -> float:
    unc_sq_list = [unc**2 for unc in unc_list]
    sum_unc_sq = sum(unc_sq_list)
    sqrt_sum_unc_sq = math.sqrt(sum_unc_sq)
    error = sqrt_sum_unc_sq / len(unc_list)
    return error


def match_levels(
    levels_initial: pd.DataFrame,
    levels_new: pd.DataFrame,
    qn_match_cols: t.List[str],
    match_source_tag: SourceTag,
    shift_table_qn_cols: t.List[str],
    levels_new_qn_cols: t.List[str] = None,
    suffixes: t.Tuple[str, str] = None,
    energy_col: str = "energy",
    unc_col: str = "unc",
    source_tag_col: str = "source_tag",
    id_col: str = "ID",
    is_isotopologue_match: bool = False,
    overwrite_non_match_qn_cols: bool = False,
) -> t.Tuple[pd.DataFrame, pd.DataFrame]:
    """

    Args:
        levels_initial:
        levels_new:
        qn_match_cols:
        match_source_tag:
        shift_table_qn_cols:
        levels_new_qn_cols:
        suffixes:
        energy_col:
        unc_col:
        source_tag_col:
        id_col:
        is_isotopologue_match:
        overwrite_non_match_qn_cols:

    Returns:

    """
    if suffixes is None:
        suffixes = ("_calc", "_obs")

    if levels_new_qn_cols is None:
        levels_new_qn_cols = qn_match_cols

    # if energy_col_name is None:
    #     energy_col_name = 'energy'

    # Take an inner merge to only get the levels that do have matches.
    levels_matched = levels_initial.merge(
        levels_new,
        left_on=qn_match_cols,
        right_on=qn_match_cols,
        suffixes=suffixes,
        how="inner",
    )
    print("LEVELS MATCHED: \n", levels_matched)
    if len(levels_matched) == 0:
        raise RuntimeWarning(
            "No matching levels found. New levels will be appended to existing set."
        )
        # TODO: Change to error, or add input fail_on_no_matches?

    energy_original_col = energy_col + suffixes[0]
    energy_marvel_col = energy_col + suffixes[1]
    energy_dif_col = energy_col + "_dif"
    energy_dif_mag_col = energy_dif_col + "_mag"

    levels_matched[energy_dif_col] = levels_matched.apply(
        lambda x: x[energy_marvel_col] - x[energy_original_col], axis=1
    )
    levels_matched[energy_dif_mag_col] = levels_matched[energy_dif_col].apply(
        lambda x: abs(x)
    )

    # This was bad as it changed the structure of the output DataFrame to be that of the GroupBy frame.
    # levels_matched = levels_matched.sort_values(energy_dif_mag_col).groupby(by=qn_match_cols, as_index=False).first()

    qn_cols_not_match = np.setdiff1d(levels_new_qn_cols, qn_match_cols)
    # If not matching on the full set of quantum numbers that uniquely determine a level, check for duplicates such that
    # no more than one level (defined by the input set of quantum numbers qn_match_cols) is matching on the same
    # levels_new level.
    qn_dupe_cols = (
        qn_match_cols
        if len(qn_cols_not_match) == 0
        else qn_match_cols + [energy_marvel_col]
    )

    levels_matched_dup = levels_matched[
        levels_matched.duplicated(subset=qn_dupe_cols, keep=False)
    ].sort_values(qn_dupe_cols + [energy_dif_mag_col])

    if len(levels_matched_dup) > 0:
        # Get the index of the lowest energy_agreement entry, for each tag which has duplicates
        levels_matched_dup_idx_min = (
            levels_matched_dup.groupby(by=qn_dupe_cols, sort=False)[
                energy_dif_mag_col
            ].transform(min)
            == levels_matched_dup[energy_dif_mag_col]
        )
        # Take the index of everything other than the lowest energy_agreement entry for each duplicated tag and remove
        # it from the comparison dataframe
        levels_matched = levels_matched.drop(
            levels_matched_dup[~levels_matched_dup_idx_min].index
        )

    # TODO: TEST THIS ALTERNATIVE. Explicit ascending?
    # levels_matched = levels_matched.sort_values(by=[qn_dupe_cols + [energy_dif_mag_col]]).drop_duplicates(
    #     subset=qn_dupe_cols, keep='first')

    # Remove the energy difference magnitude column as it is not needed beyond this point.
    del levels_matched[energy_dif_mag_col]

    # Check the 0 energy level.
    zero_energy_level_matches = len(
        levels_matched.loc[
            (levels_matched[energy_original_col] == 0)
            & (levels_matched[energy_marvel_col] == 0)
        ]
    )
    if zero_energy_level_matches != 1:
        raise RuntimeError(
            "0 ENERGY LEVELS DO NOT MATCH ASSIGNMENTS IN BOTH DATASETS.\nORIGINAL:\n",
            levels_matched.loc[levels_matched[energy_original_col] == 0],
            "\nUPDATE:\n",
            levels_matched.loc[levels_matched[energy_marvel_col] == 0],
        )

    if not is_isotopologue_match:
        # Merge the sets of qn_match_cols in each DataFrame with an indicator to find any rows that are only in
        # levels_new to concat them to levels_matched.
        levels_new_check_matched = levels_new[qn_match_cols].merge(
            levels_matched[qn_match_cols], how="left", indicator=True
        )
        # Get the indexes of levels_new where the unique qn_match_cols are not yet in levels_matched.
        levels_new_to_concat_idx = levels_new_check_matched[
            (levels_new_check_matched["_merge"] == "left_only")
        ].index
        # Isolate only the rows with those indexes to be concatenated.
        levels_new_to_concat = levels_new[
            levels_new.index.isin(levels_new_to_concat_idx)
        ]
        # Rename energy_col_name in the rows to be concatenated to energy_marvel_col to avoid creating a new column.
        levels_new_to_concat = levels_new_to_concat.rename(
            columns={energy_col: energy_marvel_col}
        )
        if len(qn_cols_not_match) > 0:
            levels_new_to_concat = levels_new_to_concat.rename(
                columns={qn_col: qn_col + suffixes[1] for qn_col in qn_cols_not_match}
            )
        # levels_matched = levels_matched.append(levels_new_to_append)
        levels_matched = pd.concat([levels_matched, levels_new_to_concat])

    levels_matched[source_tag_col] = match_source_tag.value
    # TODO: Change to rename original column? Or worth keeping both?
    levels_matched["energy_final"] = levels_matched[energy_marvel_col]

    # Create table to provide energy shifts and std (for unc estimates) based on a qn grouping.
    shift_table = (
        levels_matched.loc[
            (~levels_matched[energy_original_col].isna())
            & (~levels_matched[energy_marvel_col].isna())
        ]
        .groupby(by=shift_table_qn_cols, as_index=False)
        .apply(
            lambda x: pd.Series(
                {
                    "energy_dif_mean": np.average(x[energy_dif_col]),
                    "energy_dif_unc": propagate_error_in_mean(x[unc_col]),
                }
            )
        )
    )
    print("SHIFT TABLE: \n", shift_table)

    # Rename original levels' energy to match the column name in levels_matched
    levels_initial = levels_initial.rename(columns={energy_col: energy_original_col})
    # Add missing original levels that are not in the final matching set
    # levels_matched = levels_matched.append(levels_initial.loc[~levels_initial['id'].isin(levels_matched['id'])])
    levels_initial_to_concat = levels_initial.loc[
        ~levels_initial[id_col].isin(levels_matched[id_col])
    ]
    if len(qn_cols_not_match) > 0:
        levels_initial_to_concat = levels_initial_to_concat.rename(
            columns={qn_col: qn_col + suffixes[0] for qn_col in qn_cols_not_match}
        )

    if (
        unc_col in levels_initial_to_concat.columns
        and unc_col + suffixes[0] in levels_matched.columns
    ):
        levels_initial_to_concat = levels_initial_to_concat.rename(
            columns={unc_col: unc_col + suffixes[0]}
        )

    levels_matched = pd.concat([levels_matched, levels_initial_to_concat])

    if overwrite_non_match_qn_cols and len(qn_cols_not_match) > 0:
        for qn_col_not_match in qn_cols_not_match:
            levels_matched[qn_col_not_match] = np.where(
                levels_matched[qn_col_not_match + suffixes[1]].isna(),
                levels_matched[qn_col_not_match + suffixes[0]],
                levels_matched[qn_col_not_match + suffixes[1]],
            )
            del levels_matched[qn_col_not_match + suffixes[0]]
            del levels_matched[qn_col_not_match + suffixes[1]]

    return levels_matched, shift_table


def estimate_uncertainty(
    j_val: float, v_val: float, j_factor: float, v_factor: float
) -> float:
    """
    Provides an uncertainty estimate for a state based on a quadratic J and linear v term.
    Currently, only intended for use with diatomics, due to the singular vibrational quantum number input.

    Args:
        j_val:    The J value of the state.
        v_val:    The v value of the state.
        j_factor: The value to scale the quadratic J term by.
        v_factor: The value to scale the linear v term by.

    Returns:
        A float representing an uncertainty estimate for a state.
    """
    return j_factor * j_val * (j_val * +1) + v_factor * v_val


def set_predicted_unc(
    std: float,
    j_factor: float,
    v_factor: float,
    j_val: float = 0,
    j_max: float = 0,
    v_val: float = 0,
    v_max: float = 0,
) -> float:
    """
    Provides an increased uncertainty estimate for a state based on an initial input uncertainty and a calculated
    increase proportional to the states quantum numbers J and v. This increase is obtained from
    :func:`linelisttools.states.estimate_uncertainty`.

    In cases where uncertainties are to be determined for higher J states in a series (i.e.: a vibronic band) then J_max
    should be set as the highest J present in that series, such that the resulting uncertainty estimate is only
    dependent on how far beyond J_max the user is extrapolating to. Generally, scaling based on v and v_max is for use
    with providing uncertainty estimates for calculated or pseudo-experimentally corrected states in vibrational bands
    above the observational data.

    Currently, this method is only intended for use with diatomics, due to the singular vibrational quantum number
    input.

    Args:
        std:      The base uncertainty of a state to be increased.
        j_factor: The value to scale the quadratic J term by.
        v_factor: The value to scale the linear v term by.
        j_val:    The J value of the state.
        j_max:    The maximum value of J that exists in the series that predictions are being made over. If present then
            the uncertainty estimate J term will only scale off how much greater the current J value is that this.
        v_val:    The v value of the state.
        v_max:    The maximum value of v that exists in the series that predictions are being made over. If present then
            the uncertainty estimate v term will only scale off how much greater the current v value is that this.

    Returns:
        A float representing the updated uncertainty of a state with an applied energy prediction.
    """
    j_dif = max(0, int(j_val - j_max)) if j_val != 0 and j_max != 0 else 0
    v_dif = max(0, int(v_val - v_max)) if v_val != 0 and v_max != 0 else 0
    unc_extrapolated = estimate_uncertainty(j_dif, v_dif, j_factor, v_factor)
    return std + unc_extrapolated


def predict_shifts(
    states_matched: pd.DataFrame,
    shift_table: pd.DataFrame,
    fit_qn_list: t.List[str],
    j_segment_threshold_size: int = 14,
    show_plot: bool = False,
    unc_col: str = "unc",
    j_col: str = "J",
    source_tag_col: str = "source_tag",
    executor_type: ExecutorType = ExecutorType.THREADS,
    n_workers: int = 8,
) -> pd.DataFrame:
    """


    Args:
        states_matched:           The matched Marvel/calculated states from which the obs.-calc. values to fit to are
            derived.
        shift_table:              The shift table derived from the matches states, providing mean obs.-calc. shifts for
            the input states grouped by an arbitrary set of quantum numbers. Generally this grouping should be the same
            as the quantum numbers provided in fit_qn_list.
        fit_qn_list:              The list of arbitrary quantum numbers to group the obs.-calc. trends on for fitting.
            Generally should be the same as those used to generate the shift table. All quantum numbers must exist as
             columns within the shift_table and levels_matched DataFrames.
        j_segment_threshold_size: The minimum number of J data-points that must be present in a given segment to fit to.
            The segments that obs.-calc. predictions are fit to will increase in size if multiple sets of missing data
            exist within an array of sequential J values of length equal to this argument.
        show_plot:                Determines whether plots of the input and fitted data are shown.
        unc_col:                  The string column name for the uncertainty column in states_matched.
        j_col:                    The string column name for the J column in states_matched.
        source_tag_col:           The string column name for the source tag column in states_matched.
        executor_type:            Determines whether the fitting will be carried out with multiple threads or processes.
            Defaults to multithreading.
        n_workers:                The number of threads/processes to concurrently execute for the fitting.

    Returns:
        Outputs the states_matched DataFrame with updated interpolated and extrapolated energy shifts in the series
        defined by fit_qn_list for which Marvel data exists.
    """
    # if fit_qn_list is not None:
    #     shift_table["fit_qn"] = shift_table.apply(
    #         lambda x: "|".join(str(x[qn]) for qn in fit_qn_list), axis=1
    #     )
    shift_predictions = []
    extrapolate_j_shifts = []
    # for fit_qn in shift_table["fit_qn"].unique():
    #     shift_predictions, extrapolate_j_shifts = old_fit_predictions(
    #         shift_table=shift_table,
    #         shift_predictions=shift_predictions,
    #         extrapolate_j_shifts=extrapolate_j_shifts,
    #         colour="#8b4513",
    #         fit_qn=fit_qn,
    #         j_segment_threshold_size=14,
    #         j_col=j_col,
    #         show_plot=show_plot,
    #     )

    worker = functools.partial(
        fit_predictions, "#8b4513", j_segment_threshold_size, j_col, show_plot
    )
    shift_groups = shift_table.groupby(by=fit_qn_list)
    with executor_type.value(max_workers=n_workers) as e:
        for result in tqdm.tqdm(
            e.map(worker, yield_grouped_data(shift_groups)), total=len(shift_groups)
        ):
            shift_predictions.append(result[0])
            extrapolate_j_shifts.append(result[1])
    shift_predictions = [item for items in shift_predictions for item in items]
    extrapolate_j_shifts = [item for items in extrapolate_j_shifts for item in items]

    # Update energies with shift predictions:
    pe_fit_shifts = pd.DataFrame(
        data=shift_predictions,
        columns=fit_qn_list + [j_col, "pe_fit_energy_shift", "pe_fit_unc"],
    )
    # pe_fit_shifts[fit_qn_list] = pe_fit_shifts["fit_qn"].str.split("|", len(fit_qn_list), expand=True)
    # del pe_fit_shifts["fit_qn"]

    qn_merge_cols = fit_qn_list + [j_col]
    states_matched = states_matched.merge(
        pe_fit_shifts, left_on=qn_merge_cols, right_on=qn_merge_cols, how="left"
    )
    states_matched.loc[
        (states_matched["energy_final"].isna())
        & (~states_matched["pe_fit_energy_shift"].isna())
        & (~states_matched["energy_calc"].isna()),
        source_tag_col,
    ] = "PS_2"
    states_matched[unc_col] = np.where(
        states_matched[source_tag_col] == "PS_2",
        states_matched["pe_fit_unc"],
        states_matched[unc_col],
    )
    states_matched["energy_final"] = np.where(
        states_matched[source_tag_col] == "PS_2",
        states_matched["energy_calc"] + states_matched["pe_fit_energy_shift"],
        states_matched["energy_final"],
    )
    del states_matched["pe_fit_energy_shift"]
    del states_matched["pe_fit_unc"]
    print("PS_2: \n", states_matched.loc[states_matched[source_tag_col] == "PS_2"])

    # Update energies with higher-J shift extrapolations:
    j_max_col = j_col + "_max"
    pe_extrapolate_shifts = pd.DataFrame(
        data=extrapolate_j_shifts,
        columns=fit_qn_list
        + [
            j_max_col,
            "pe_extrapolate_energy_shift",
            "pe_extrapolate_energy_shift_std",
        ],
    )
    # pe_extrapolate_shifts[fit_qn_list] = pe_extrapolate_shifts["fit_qn"].str.split("|", len(fit_qn_list), expand=True)
    # del pe_extrapolate_shifts["fit_qn"]

    states_matched = states_matched.merge(
        pe_extrapolate_shifts, left_on=fit_qn_list, right_on=fit_qn_list, how="left"
    )
    states_matched.loc[
        (states_matched["energy_final"].isna())
        & (~states_matched[j_max_col].isna())
        & (~states_matched["energy_calc"].isna())
        & (states_matched[j_col] > states_matched[j_max_col]),
        source_tag_col,
    ] = "PS_3"
    # Scale unc based on j over j_max.
    # states_matched['unc'] = states_matched.apply(
    #     lambda x: scale_uncertainty(std=x['pe_extrapolate_energy_shift_std'], std_scale=2, j_val=x['j'],
    #                                 j_max=x['j_max'], j_scale=0.05)
    #     if math.isnan(x['energy_final']) and not math.isnan(x['energy_calc']) and not math.isnan(x['j_max'])
    #        and x['j'] > x['j_max'] else x['unc'], axis=1)
    states_matched[unc_col] = states_matched.apply(
        lambda x: set_predicted_unc(
            std=x["pe_extrapolate_energy_shift_std"],
            j_factor=0.0001,
            v_factor=0.05,
            j_val=x[j_col],
            j_max=x[j_max_col],
        )
        if x[source_tag_col] == "PS_3"
        else x[unc_col],
        axis=1,
    )
    states_matched["energy_final"] = np.where(
        states_matched[source_tag_col] == "PS_3",
        states_matched["energy_calc"] + states_matched["pe_extrapolate_energy_shift"],
        states_matched["energy_final"],
    )
    del states_matched[j_max_col]
    del states_matched["pe_extrapolate_energy_shift"]
    del states_matched["pe_extrapolate_energy_shift_std"]
    print("PS_3: \n", states_matched.loc[states_matched[source_tag_col] == "PS_3"])

    # UNNECESSARY IF NOT OUTPUTTING THESE DATAFRAMES.
    # Add shift predictions to shift table.
    # What to do when unc = NaN?
    # shift_table_full_j = shift_table.copy()
    # shift_table_full_j = shift_table_full_j.rename(
    #     columns={
    #         "energy_dif_mean": "energy_shift",
    #         "energy_dif_std": "energy_shift_unc",
    #     }
    # )
    # pe_fit_shifts = pe_fit_shifts.rename(
    #     columns={
    #         "pe_fit_energy_shift": "energy_shift",
    #         "pe_fit_unc": "energy_shift_unc",
    #     }
    # )
    # shift_table_full_j = shift_table_full_j.append(pe_fit_shifts)
    # shift_table_full_j = shift_table_full_j.sort_values(
    #     by=["state", "v", "Omega", j_col], ascending=[1, 1, 1, 1]
    # )
    #
    # pe_extrapolate_shifts = pe_extrapolate_shifts.rename(
    #     columns={
    #         "pe_extrapolate_energy_shift": "extrapolation_energy_shift",
    #         "pe_extrapolate_energy_shift_std": "extrapolation_energy_unc",
    #     }
    # )
    return states_matched


def fit_predictions(
    colour: str,
    j_segment_threshold_size: int,
    j_col: str,
    show_plot: bool,
    grouped_data: t.Tuple[t.List[str], pd.DataFrame],
) -> t.Tuple[t.List[tuple], t.List[tuple]]:
    """

    Args:
        colour:
        j_segment_threshold_size:
        j_col:
        show_plot:
        grouped_data:

    Returns:

    """
    # TODO: Clean up internal plotting - does not work well with multithreading.
    shift_predictions = []
    extrapolate_j_shifts = []
    fit_qn_list = tuple(grouped_data[0])
    df_group = grouped_data[1]
    j_max = df_group[j_col].max()
    j_coverage_to_max = [x / 2 for x in range(1, int(j_max * 2), 2)]
    missing_j = np.array(np.setdiff1d(j_coverage_to_max, df_group[j_col]))
    if len(missing_j) > 0:
        delta_missing_j = np.abs(missing_j[1:] - missing_j[:-1])
        split_idx = np.where(delta_missing_j >= j_segment_threshold_size)[0] + 1
        missing_j_segments = np.array_split(missing_j, split_idx)
        if show_plot:
            # Plot the actual data:
            fig = plt.figure()
            ax = fig.gca()
            ax.scatter(
                df_group[j_col],
                df_group["energy_dif_mean"],
                marker="x",
                linewidth=0.5,
                facecolors=colour,
                label=" ".join(str(qn) for qn in fit_qn_list),
                zorder=1,
            )
        for j_segment in missing_j_segments:
            # If the segment is entirely within the slice then the wing size is half the threshold. This
            if (
                min(j_segment) > df_group[j_col].min()
                and max(j_segment) < df_group[j_col].max()
            ):
                segment_wing_size = int(j_segment_threshold_size / 2)
            else:
                segment_wing_size = int(j_segment_threshold_size)

            segment_j_lower_limit = max(0.5, min(j_segment) - segment_wing_size)
            segment_j_upper_limit = max(
                j_segment_threshold_size + 0.5,
                max(j_segment) + segment_wing_size,
            )
            segment_j_coverage = [
                x / 2
                for x in range(
                    int(segment_j_lower_limit * 2),
                    int((segment_j_upper_limit + 1) * 2),
                    2,
                )
            ]
            # Huber regression - resist outliers.
            x_scaler, y_scaler = StandardScaler(), StandardScaler()
            x_train = x_scaler.fit_transform(
                np.array(
                    df_group.loc[
                        (df_group[j_col] >= segment_j_lower_limit)
                        & (df_group[j_col] <= segment_j_upper_limit),
                        j_col,
                    ]
                )[..., None]
            )
            y_train = y_scaler.fit_transform(
                np.array(
                    df_group.loc[
                        (df_group[j_col] >= segment_j_lower_limit)
                        & (df_group[j_col] <= segment_j_upper_limit),
                        "energy_dif_mean",
                    ]
                )[..., None]
            )
            model = HuberRegressor(epsilon=1.35, max_iter=500)
            model.fit(x_train, y_train.ravel())

            model_segment_predictions = model.predict(
                x_scaler.transform(j_segment[..., None])
            ).reshape(-1, 1)
            segment_predictions = y_scaler.inverse_transform(
                model_segment_predictions
            ).ravel()
            # Find the J we are making predictions for that we also have known energy_dif values for.
            segment_j_in_slice = np.array(
                np.intersect1d(segment_j_coverage, df_group[j_col])
            )
            segment_j_outliers = np.array(
                df_group.loc[
                    (df_group[j_col].isin(segment_j_in_slice))
                    & (
                        abs(
                            df_group["energy_dif_mean"]
                            - df_group["energy_dif_mean"].mean()
                        )
                        > (2 * df_group["energy_dif_mean"].std())
                    ),
                    j_col,
                ]
            )
            segment_j_in_slice_no_outliers = np.setdiff1d(
                segment_j_in_slice, segment_j_outliers
            )

            # standard_error_of_estimate = mean_squared_error(
            #     shift_table_slice.loc[shift_table_slice['j'].isin(segment_j_in_slice_no_outliers),
            #                           'energy_dif_mean'],
            #     y_scaler.inverse_transform(model.predict(x_scaler.transform(
            #         segment_j_in_slice_no_outliers[..., None]))),
            #     squared=False)
            real_energy = np.array(
                df_group.loc[
                    df_group[j_col].isin(segment_j_in_slice_no_outliers),
                    "energy_dif_mean",
                ]
            )
            mode_present_predictions = model.predict(
                x_scaler.transform(segment_j_in_slice_no_outliers[..., None])
            ).reshape(-1, 1)
            present_predictions = y_scaler.inverse_transform(
                mode_present_predictions
            ).ravel()

            dif_energy = real_energy - present_predictions
            dif_squared_energy = dif_energy**2
            std_energy = np.sqrt(sum(dif_squared_energy) / len(dif_energy))

            if show_plot:
                ax.scatter(
                    j_segment,
                    segment_predictions,
                    marker="^",
                    linewidth=0.5,
                    edgecolors="#000000",
                    facecolors="none",
                    label=f"{' '.join(str(qn) for qn in fit_qn_list)} FIT",
                    zorder=2,
                )

            for entry in [
                fit_qn_list + (j, prediction, std_energy)
                for j, prediction in zip(j_segment, list(segment_predictions))
            ]:
                shift_predictions.append(entry)

        if show_plot:
            ax.legend(loc="upper left", prop={"size": 10})
            ax.set_xlabel(xlabel=j_col)
            ax.set_ylabel(ylabel=r"Obs.-Calc. (cm-1)")
            # plt.ylim(bottom=-1, top=1)
            plt.sca(ax)
            plt.tight_layout()
            plt.show()
    # Now take the mean of the last 10 points within 2std and take the mean shift there and apply it to
    # all later trans.
    shift_table_final_rows = (
        df_group.loc[
            abs(df_group["energy_dif_mean"] - df_group["energy_dif_mean"].mean())
            < (2 * df_group["energy_dif_mean"].std())
        ]
        .sort_values(by=j_col, ascending=[1])
        .tail(j_segment_threshold_size)
    )
    extrapolate_j_shift_mean = shift_table_final_rows["energy_dif_mean"].mean()
    extrapolate_j_shift_std = shift_table_final_rows["energy_dif_mean"].std()
    extrapolate_j_shifts.append(
        fit_qn_list + (j_max, extrapolate_j_shift_mean, extrapolate_j_shift_std)
    )

    return shift_predictions, extrapolate_j_shifts


# Deprecated - slower and handling of column splitting/retrieving column types an unnecessary hassle.
def old_fit_predictions(
    shift_table: pd.DataFrame,
    shift_predictions: t.List[t.List],
    extrapolate_j_shifts: t.List[t.List],
    colour: str,
    fit_qn: str,
    j_segment_threshold_size: int = 14,
    j_col: str = "J",
    show_plot: bool = False,
):
    shift_table_slice = shift_table.loc[
        (shift_table["fit_qn"] == fit_qn), [j_col, "energy_dif_mean"]
    ]
    j_max = shift_table_slice[j_col].max()
    j_coverage_to_max = [x / 2 for x in range(1, int(j_max * 2), 2)]
    missing_j = np.array(np.setdiff1d(j_coverage_to_max, shift_table_slice[j_col]))
    if len(missing_j) > 0:

        delta_missing_j = np.abs(missing_j[1:] - missing_j[:-1])
        split_idx = np.where(delta_missing_j >= j_segment_threshold_size)[0] + 1
        missing_j_segments = np.array_split(missing_j, split_idx)
        if show_plot:
            # Plot the actual data:
            plt.scatter(
                shift_table_slice[j_col],
                shift_table_slice["energy_dif_mean"],
                marker="x",
                linewidth=0.5,
                facecolors=colour,
                label=fit_qn,
                zorder=1,
            )
        for j_segment in missing_j_segments:
            # If the segment is entirely within the slice then the wing size is half the threshold. This
            if (
                min(j_segment) > shift_table_slice[j_col].min()
                and max(j_segment) < shift_table_slice[j_col].max()
            ):
                segment_wing_size = int(j_segment_threshold_size / 2)
            else:
                segment_wing_size = int(j_segment_threshold_size)

            segment_j_lower_limit = max(0.5, min(j_segment) - segment_wing_size)
            segment_j_upper_limit = max(
                j_segment_threshold_size + 0.5,
                max(j_segment) + segment_wing_size,
            )
            segment_j_coverage = [
                x / 2
                for x in range(
                    int(segment_j_lower_limit * 2),
                    int((segment_j_upper_limit + 1) * 2),
                    2,
                )
            ]
            # Huber regression - resist outliers.
            x_scaler, y_scaler = StandardScaler(), StandardScaler()
            x_train = x_scaler.fit_transform(
                np.array(
                    shift_table_slice.loc[
                        (shift_table_slice[j_col] >= segment_j_lower_limit)
                        & (shift_table_slice[j_col] <= segment_j_upper_limit),
                        j_col,
                    ]
                )[..., None]
            )
            y_train = y_scaler.fit_transform(
                np.array(
                    shift_table_slice.loc[
                        (shift_table_slice[j_col] >= segment_j_lower_limit)
                        & (shift_table_slice[j_col] <= segment_j_upper_limit),
                        "energy_dif_mean",
                    ]
                )[..., None]
            )
            model = HuberRegressor(epsilon=1.35, max_iter=500)
            model.fit(x_train, y_train.ravel())
            model_segment_predictions = model.predict(
                x_scaler.transform(j_segment[..., None])
            ).reshape(-1, 1)
            segment_predictions = y_scaler.inverse_transform(
                model_segment_predictions
            ).ravel()
            # Find the J we are making predictions for that we also have known energy_dif values for.
            segment_j_in_slice = np.array(
                np.intersect1d(segment_j_coverage, shift_table_slice[j_col])
            )
            segment_j_outliers = np.array(
                shift_table_slice.loc[
                    (shift_table_slice[j_col].isin(segment_j_in_slice))
                    & (
                        abs(
                            shift_table_slice["energy_dif_mean"]
                            - shift_table_slice["energy_dif_mean"].mean()
                        )
                        > (2 * shift_table_slice["energy_dif_mean"].std())
                    ),
                    j_col,
                ]
            )
            segment_j_in_slice_no_outliers = np.setdiff1d(
                segment_j_in_slice, segment_j_outliers
            )
            real_energy = np.array(
                shift_table_slice.loc[
                    shift_table_slice[j_col].isin(segment_j_in_slice_no_outliers),
                    "energy_dif_mean",
                ]
            )
            model_present_predictions = model.predict(
                x_scaler.transform(segment_j_in_slice_no_outliers[..., None])
            ).reshape(-1, 1)
            present_predictions = y_scaler.inverse_transform(
                model_present_predictions
            ).ravel()

            dif_energy = real_energy - present_predictions
            dif_squared_energy = dif_energy**2
            std_energy = np.sqrt(sum(dif_squared_energy) / len(dif_energy))

            if show_plot:
                plt.scatter(
                    j_segment,
                    segment_predictions,
                    marker="^",
                    linewidth=0.5,
                    edgecolors="#000000",
                    facecolors="none",
                    label=f"{fit_qn} FIT",
                    zorder=2,
                )

            for entry in [
                [fit_qn, j, prediction, std_energy]
                for j, prediction in zip(j_segment, segment_predictions)
            ]:
                shift_predictions.append(entry)
        if show_plot:
            plt.legend(loc="upper left", prop={"size": 10})
            plt.xlabel(j_col)
            plt.ylabel(r"Obs.-Calc. (cm$^{-1}$)")
            # plt.ylim(bottom=-1, top=1)
            plt.tight_layout()
            plt.show()
    # Now take the mean of the last 10 points within 2std and take the mean shift there and apply it to
    # all later trans.
    shift_table_final_rows = (
        shift_table_slice.loc[
            abs(
                shift_table_slice["energy_dif_mean"]
                - shift_table_slice["energy_dif_mean"].mean()
            )
            < (2 * shift_table_slice["energy_dif_mean"].std())
        ]
        .sort_values(by=j_col, ascending=[1])
        .tail(j_segment_threshold_size)
    )
    extrapolate_j_shift_mean = shift_table_final_rows["energy_dif_mean"].mean()
    extrapolate_j_shift_std = shift_table_final_rows["energy_dif_mean"].std()
    extrapolate_j_shifts.append(
        [
            fit_qn,
            j_max,
            extrapolate_j_shift_mean,
            extrapolate_j_shift_std,
        ]
    )
    return shift_predictions, extrapolate_j_shifts


def set_calc_states(
    states: pd.DataFrame,
    unc_j_factor: float = 0.0001,
    unc_v_factor: float = 0.05,
    source_tag_col: str = "source_tag",
    unc_col: str = "unc",
    j_col: str = "J",
    v_col: str = "v",
    energy_final_col: str = "energy_final",
    energy_calc_col: str = "energy_calc",
) -> pd.DataFrame:
    """
    Updates all states with no assigned source tag to Calculated and estimates their uncertainty with the function
    :func:`linelisttools.states.estimate_uncertainty`. Currently, only works for diatomic state files given the v
    scaling in the uncertainty estimator.

    Args:
        states:           A DataFrame containing all states, those of which without a source_tag set will be updated to
            calculated.
        unc_j_factor:     The uncertainty scale factor for the J term.
        unc_v_factor:     The uncertainty scale factor for the v term.
        source_tag_col:   The string label for the source tag column in states.
        unc_col:          The string label for the uncertainty column in states.
        j_col:            The string label for the J column in states.
        v_col:            The string label for the v column in states.
        energy_final_col: The string label for the final energy column in states.
        energy_calc_col:  The string label for the calculated energy column in states.

    Returns:
        A DataFrame where all input states without a source tag assigned have been set to Calculated and had their
            uncertainty estimated.
    """
    # v_limit_states = states.loc[states[source_tag_col] == source_tag_v_limiter, 'state'].unique()
    # for state in v_limit_states:
    #     update_v_max = states.loc[(states[source_tag_col] == source_tag_v_limiter)
    #                               & (states['state'] == state), v_col].max()
    #     states.loc[(states[source_tag_col].isna()) & (states['state'] == state)
    #                & (states[v_col] > update_v_max), source_tag_col] = SourceTag.CALCULATED.value
    # states.loc[(states[source_tag_col].isna()) & (~states['state'].isin(v_limit_states)),
    #            source_tag_col] = SourceTag.CALCULATED.value
    # states[energy_final_col] = np.where(states[source_tag_col] == SourceTag.CALCULATED.value,
    #                                   states[energy_calc_col], states[energy_final_col])

    # The above previous implementation only set states to Calculated if either the electronic state had no marvel
    # levels, or vibrational bands of electronic states that were beyond the marvel coverage. This was intended to be
    # agnostic to the order in which states are updated (i.e.: are predicted shifts determined before or after
    # calculated levels are set) but I feel it makes more sense to ensure this is done as the last step to catch the
    # remaining states left unchanged.
    states[source_tag_col] = np.where(
        states[source_tag_col].isna(),
        SourceTag.CALCULATED.value,
        states[source_tag_col],
    )
    states[energy_final_col] = np.where(
        states[source_tag_col] == SourceTag.CALCULATED.value,
        states[energy_calc_col],
        states[energy_final_col],
    )
    states[unc_col] = states.apply(
        lambda x: estimate_uncertainty(x[j_col], x[v_col], unc_j_factor, unc_v_factor)
        if x[source_tag_col] == SourceTag.CALCULATED.value
        else x[unc_col],
        axis=1,
    )

    return states


def shift_parity_pairs(
    states: pd.DataFrame,
    shift_table: pd.DataFrame,
    source_tag_col: str = "source_tag",
    unc_col: str = "unc",
    id_col: str = "ID",
    energy_final_col: str = "energy_final",
    energy_calc_col: str = "energy_calc",
) -> pd.DataFrame:
    """
    Updates levels that have not had their source_tag set but have a level with equivalent quantum numbers in the Marvel
    data. This is determined through merging those states without a source_tag on the shift_table, which contains a list
    of the unique combinations of quantum numbers in the Marvel data. This assumes the shift table/level matching was
    performed over a full set of quantum numbers; if matching was done on a partial set, a shift table should be
    manually created for the set of quantum numbers used to determine a parity pair.

    Args:
        states:           A DataFrame containing the states to search for parity pairs in.
        shift_table:      The shift table for the states file, defining the mean energy shift for all states matching a
            set of quantum numbers.
        source_tag_col:   The string label for the source tag column in states.
        unc_col:          The string label for the uncertainty column in states.
        id_col:           The string label for the ID column in states.
        energy_final_col: The string label for the final energy column in states.
        energy_calc_col:  The string label for the calculated energy column in states.

    Returns:
        The states DataFrame updated with the mean energy shift from the shift table applied to any parity pair
            counterparts that were not updated with Marvel data.
    """
    # energy_dif_mean and energy_dif_unc are implicit column names of the shift table: other columns are the quantum
    # numbers it was grouped on.
    energy_dif_mean_col = "energy_dif_mean"
    energy_dif_unc_col = "energy_dif_unc"
    shift_table_qn_cols = [
        qn
        for qn in shift_table.columns
        if qn not in (energy_dif_mean_col, energy_dif_unc_col)
    ]

    # Inner merge here gets us only the states that have matching quantum numbers to an entry in the shift table but has
    # not had its source_tag set, i.e.: those for which we have Marvel data for another level with the same quantum
    # numbers, excluding parity.
    states_missing_parity_pairs = states.loc[states[source_tag_col].isna()].merge(
        shift_table, on=shift_table_qn_cols, how="inner"
    )

    # Apply shift table mean energy difference to calculated energy.
    states_missing_parity_pairs[energy_final_col] = states_missing_parity_pairs.apply(
        lambda x: x[energy_calc_col] + x[energy_dif_mean_col], axis=1
    )

    # Left merge parity pair shifts onto states to keep all states and give shift data where needed.
    states = states.merge(
        states_missing_parity_pairs[[id_col, energy_final_col, energy_dif_unc_col]],
        on=[id_col],
        how="left",
        suffixes=("", "_temp"),
    )
    states.loc[
        states[id_col].isin(states_missing_parity_pairs[id_col]), source_tag_col
    ] = "PS_1"
    # Take the energy_final_temp from the merged DataFrame as energy_final where it exists, and the energy_dif_unc as
    # the unc for these rows.
    energy_final_temp_col = energy_final_col + "_temp"
    # Change to find new merge cols based on "PS_1" source_tag?
    states[energy_final_col] = np.where(
        states[energy_final_temp_col].isna(),
        states[energy_final_col],
        states[energy_final_temp_col],
    )
    states[unc_col] = np.where(
        states[energy_final_temp_col].isna(),
        states[unc_col],
        states[energy_dif_unc_col],
    )
    # Drop any temp columns, including the now useless (as it has been copied over into unc) energy_dif_unc column.
    states = states.drop(
        list(states.filter(regex="_temp")) + [energy_dif_unc_col], axis=1
    )
    # states = states.drop(energy_dif_unc_col, axis=1)
    # Reorder on id for convenience.
    states = states.sort_values(by=[id_col], ascending=True)
    return states
