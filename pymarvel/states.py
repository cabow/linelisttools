import math
import typing as t

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler


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


def propagate_error_in_mean(unc_list: list) -> float:
    unc_sq_list = [unc**2 for unc in unc_list]
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
    energy_col_name: str = "energy",
    unc_col_name: str = "unc",
    is_isotopologue_match: bool = False,
    overwrite_non_match_qn_cols: bool = False,
) -> t.Tuple[pd.DataFrame, pd.DataFrame]:
    if suffixes is None:
        suffixes = ("_calc", "_obs")

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

    energy_original_col = energy_col_name + suffixes[0]
    energy_marvel_col = energy_col_name + suffixes[1]
    energy_dif_col = energy_col_name + "_dif"
    energy_dif_mag_col = energy_dif_col + "_mag"

    levels_matched[energy_dif_col] = levels_matched.apply(
        lambda x: x[energy_marvel_col] - x[energy_original_col], axis=1
    )
    levels_matched[energy_dif_mag_col] = levels_matched[energy_dif_col].apply(
        lambda x: abs(x)
    )

    # This was bad as it changed the structure of the output DataFrame to eb that of the GroupBy frame.
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
            columns={energy_col_name: energy_marvel_col}
        )
        if len(qn_cols_not_match) > 0:
            levels_new_to_concat = levels_new_to_concat.rename(
                columns={qn_col: qn_col + suffixes[1] for qn_col in qn_cols_not_match}
            )
        # levels_matched = levels_matched.append(levels_new_to_append)
        levels_matched = pd.concat([levels_matched, levels_new_to_concat])

    levels_matched["source_tag"] = match_source_tag
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
                    "energy_dif_unc": propagate_error_in_mean(x[unc_col_name]),
                }
            )
        )
    )
    print("SHIFT TABLE: \n", shift_table)

    # Rename original levels' energy to match the column name in levels_matched
    levels_initial = levels_initial.rename(
        columns={energy_col_name: energy_original_col}
    )
    # Add missing original levels that are not in the final matching set
    # levels_matched = levels_matched.append(levels_initial.loc[~levels_initial['id'].isin(levels_matched['id'])])
    levels_initial_to_concat = levels_initial.loc[
        ~levels_initial["id"].isin(levels_matched["id"])
    ]
    if len(qn_cols_not_match) > 0:
        levels_initial_to_concat = levels_initial_to_concat.rename(
            columns={qn_col: qn_col + suffixes[0] for qn_col in qn_cols_not_match}
        )

    if (
        unc_col_name in levels_initial_to_concat.columns
        and unc_col_name + suffixes[0] in levels_matched.columns
    ):
        levels_initial_to_concat = levels_initial_to_concat.rename(
            columns={unc_col_name: unc_col_name + suffixes[0]}
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


def estimate_uncertainty(j_val: float, v_val: float, a: float, b: float) -> float:
    return a * j_val * (j_val * +1) + b * v_val


def set_pseudo_experimental_unc(
    std: float,
    a: float,
    b: float,
    j_val: float = 0,
    j_max: float = 0,
    v_val: float = 0,
    v_max: float = 0,
):
    j_dif = max(0, int(j_val - j_max)) if j_val != 0 and j_max != 0 else 0
    v_dif = max(0, int(v_val - v_max)) if v_val != 0 and v_max != 0 else 0
    unc_extrapolated = estimate_uncertainty(j_dif, v_dif, a, b)
    return std + unc_extrapolated


def predict_shifts(
    levels_matched: pd.DataFrame,
    shift_table: pd.DataFrame,
    vib_qn_list: t.List[str] = None,
    show_plot: bool = False,
) -> pd.DataFrame:
    if vib_qn_list is not None:
        shift_table["v"] = shift_table.apply(
            lambda x: "".join(str(x[vib_qn]) for vib_qn in vib_qn_list), axis=1
        )
    shift_predicitions = []
    extrapolate_j_shifts = []
    # TODO: Move colours, plotting.
    colours = [
        "#8b4513",
        "#006400",
        "#4682b4",
        "#4b0082",
        "#ff0000",
        "#ffff00",
        "#00ff00",
        "#00ffff",
        "#0000ff",
        "#ff69b4",
        "#d1af86",
    ]
    for state in shift_table["state"].unique():
        for v_idx, v in enumerate(
            shift_table.loc[shift_table["state"] == state, "v"].unique()
        ):
            for Omega_idx, Omega in enumerate(
                shift_table.loc[
                    (shift_table["state"] == state) & (shift_table["v"] == v), "Omega"
                ].unique()
            ):
                shift_table_slice = shift_table.loc[
                    (shift_table["state"] == state)
                    & (shift_table["v"] == v)
                    & (shift_table["Omega"] == Omega),
                    ["j", "energy_dif_mean"],
                ]

                # j_min = shift_table_slice['j'].min()
                j_max = shift_table_slice["j"].max()
                j_segment_threshold = 14
                j_coverage_to_max = [x / 2 for x in range(1, int(j_max * 2), 2)]
                missing_j = np.array(
                    np.setdiff1d(j_coverage_to_max, shift_table_slice["j"])
                )
                # if j_min != 0.5:
                if len(missing_j) > 0:
                    # print(f'STATE {state}, v={v}, Omega={Omega}, MIN J={j_min}')
                    # print('MISSING J: ', missing_j)
                    # j_coverage_to_max = [x / 2 for x in range(1, int(j_max * 2), 2)]
                    # missing_j = np.array(np.setdiff1d(j_coverage_to_max, shift_table_slice['j']))

                    delta_missing_j = np.abs(missing_j[1:] - missing_j[:-1])
                    split_idx = np.where(delta_missing_j >= j_segment_threshold)[0] + 1
                    missing_j_segments = np.array_split(missing_j, split_idx)
                    if show_plot:
                        # Plot the actual data:
                        plt.scatter(
                            shift_table_slice["j"],
                            shift_table_slice["energy_dif_mean"],
                            marker="x",
                            linewidth=0.5,
                            facecolors=colours[v_idx],
                            label=f"{state} {v} {Omega}",
                            zorder=1,
                        )
                    for j_segment in missing_j_segments:
                        # If the segment is entirely within the slice then the wing size is half the threshold. This
                        if (
                            min(j_segment) > shift_table_slice["j"].min()
                            and max(j_segment) < shift_table_slice["j"].max()
                        ):
                            segment_wing_size = int(j_segment_threshold / 2)
                        else:
                            segment_wing_size = int(j_segment_threshold)

                        segment_j_lower_limit = max(
                            0.5, min(j_segment) - segment_wing_size
                        )
                        segment_j_upper_limit = max(
                            j_segment_threshold + 0.5,
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
                                    (shift_table_slice["j"] >= segment_j_lower_limit)
                                    & (shift_table_slice["j"] <= segment_j_upper_limit),
                                    "j",
                                ]
                            )[..., None]
                        )
                        y_train = y_scaler.fit_transform(
                            np.array(
                                shift_table_slice.loc[
                                    (shift_table_slice["j"] >= segment_j_lower_limit)
                                    & (shift_table_slice["j"] <= segment_j_upper_limit),
                                    "energy_dif_mean",
                                ]
                            )[..., None]
                        )
                        model = HuberRegressor(epsilon=1.35, max_iter=500)
                        model.fit(x_train, y_train.ravel())
                        segment_predictions = y_scaler.inverse_transform(
                            model.predict(x_scaler.transform(j_segment[..., None]))
                        )
                        # Find the J we are making predictions for that we also have known energy_dif values for.
                        segment_j_in_slice = np.array(
                            np.intersect1d(segment_j_coverage, shift_table_slice["j"])
                        )
                        segment_j_outliers = np.array(
                            shift_table_slice.loc[
                                (shift_table_slice["j"].isin(segment_j_in_slice))
                                & (
                                    abs(
                                        shift_table_slice["energy_dif_mean"]
                                        - shift_table_slice["energy_dif_mean"].mean()
                                    )
                                    > (2 * shift_table_slice["energy_dif_mean"].std())
                                ),
                                "j",
                            ]
                        )
                        segment_j_in_slice_no_outliers = np.setdiff1d(
                            segment_j_in_slice, segment_j_outliers
                        )
                        # print('SEGMENT OUTLIERS: ', segment_j_outliers)

                        # standard_error_of_estimate = mean_squared_error(
                        #     shift_table_slice.loc[shift_table_slice['j'].isin(segment_j_in_slice_no_outliers),
                        #                           'energy_dif_mean'],
                        #     y_scaler.inverse_transform(model.predict(x_scaler.transform(
                        #         segment_j_in_slice_no_outliers[..., None]))),
                        #     squared=False)
                        # print('STANDARD ERROR OF ESTIMATE: ', standard_error_of_estimate)
                        real_energy = np.array(
                            shift_table_slice.loc[
                                shift_table_slice["j"].isin(
                                    segment_j_in_slice_no_outliers
                                ),
                                "energy_dif_mean",
                            ]
                        )
                        predicted_energy = y_scaler.inverse_transform(
                            model.predict(
                                x_scaler.transform(
                                    segment_j_in_slice_no_outliers[..., None]
                                )
                            )
                        )
                        dif_energy = real_energy - predicted_energy
                        dif_squared_energy = dif_energy**2
                        std_energy = np.sqrt(sum(dif_squared_energy) / len(dif_energy))
                        # print('STANDARD ERROR OF ESTIMATE: ', std_energy)

                        if show_plot:
                            # Plot for all predictions:
                            # segment_predictions_all = y_scaler.inverse_transform(
                            #     model.predict(x_scaler.transform(np.array(segment_j_coverage)[..., None]))
                            # )
                            # plt.scatter(segment_j_coverage, segment_predictions_all, marker='^', linewidth=0.5,
                            #             edgecolors='#000000', facecolors='none', label=f'{state} {v} {Omega} FIT',
                            #             zorder=2)
                            # Plot for predictions of missing J:
                            plt.scatter(
                                j_segment,
                                segment_predictions,
                                marker="^",
                                linewidth=0.5,
                                edgecolors="#000000",
                                facecolors="none",
                                label=f"{state} {v} {Omega} FIT",
                                zorder=2,
                            )
                            plt.legend(loc="upper left", prop={"size": 10})
                            plt.xlabel("J")
                            plt.ylabel("Obs.-Calc. (cm$^{-1}$)")

                        for entry in [
                            [state, v, Omega, j, prediction, std_energy]
                            for j, prediction in zip(j_segment, segment_predictions)
                        ]:
                            shift_predicitions.append(entry)

                    # plt.ylim(bottom=-1, top=1)
                    if show_plot:
                        plt.tight_layout()
                        # plt.savefig(r'D:\PhD\AlO\AlO_PS_A2PI_6_0.5_FIT.jpg', dpi=800)
                        plt.show()

                    # print(shift_predicitions)
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
                    .sort_values(by="j", ascending=[1])
                    .tail(j_segment_threshold)
                )
                extrapolate_j_shift_mean = shift_table_final_rows[
                    "energy_dif_mean"
                ].mean()
                extrapolate_j_shift_std = shift_table_final_rows[
                    "energy_dif_mean"
                ].std()
                extrapolate_j_shifts.append(
                    [
                        state,
                        v,
                        Omega,
                        j_max,
                        extrapolate_j_shift_mean,
                        extrapolate_j_shift_std,
                    ]
                )

    if show_plot:
        plt.show()

    # Update energies with shift predictions:
    pe_fit_shifts = pd.DataFrame(
        data=shift_predicitions,
        columns=["state", "v", "Omega", "j", "pe_fit_energy_shift", "pe_fit_unc"],
    )

    levels_matched = levels_matched.merge(
        pe_fit_shifts,
        left_on=["state", "v", "Omega", "j"],
        right_on=["state", "v", "Omega", "j"],
        how="left",
    )

    # print(levels_matched.loc[~levels_matched['pe_fit_energy_shift'].isna()])

    levels_matched.loc[
        (levels_matched["energy_final"].isna())
        & (~levels_matched["pe_fit_energy_shift"].isna())
        & (~levels_matched["energy_calc"].isna()),
        "source_tag",
    ] = "PS_2"

    levels_matched.loc[
        (levels_matched["energy_final"].isna())
        & (~levels_matched["pe_fit_energy_shift"].isna())
        & (~levels_matched["energy_calc"].isna()),
        "unc",
    ] = levels_matched.loc[
        (levels_matched["energy_final"].isna())
        & (~levels_matched["pe_fit_energy_shift"].isna())
        & (~levels_matched["energy_calc"].isna()),
        "pe_fit_unc",
    ]

    levels_matched.loc[
        (levels_matched["energy_final"].isna())
        & (~levels_matched["pe_fit_energy_shift"].isna())
        & (~levels_matched["energy_calc"].isna()),
        "energy_final",
    ] = levels_matched.loc[
        (levels_matched["energy_final"].isna())
        & (~levels_matched["pe_fit_energy_shift"].isna())
        & (~levels_matched["energy_calc"].isna()),
        ["energy_calc", "pe_fit_energy_shift"],
    ].apply(
        lambda x: x["energy_calc"] + x["pe_fit_energy_shift"], axis=1
    )

    del levels_matched["pe_fit_energy_shift"]
    del levels_matched["pe_fit_unc"]

    # Update energies with higher-J shift extrapolations:
    pe_extrapolate_shifts = pd.DataFrame(
        data=extrapolate_j_shifts,
        columns=[
            "state",
            "v",
            "Omega",
            "j_max",
            "pe_extrapolate_energy_shift",
            "pe_extrapolate_energy_shift_std",
        ],
    )
    # print('EXTRAPOLATION SHIFTS: \n', pe_extrapolate_shifts)

    levels_matched = levels_matched.merge(
        pe_extrapolate_shifts,
        left_on=["state", "v", "Omega"],
        right_on=["state", "v", "Omega"],
        how="left",
    )

    levels_matched.loc[
        (levels_matched["energy_final"].isna())
        & (~levels_matched["j_max"].isna())
        & (~levels_matched["energy_calc"].isna())
        & (levels_matched["j"] > levels_matched["j_max"]),
        "source_tag",
    ] = "PS_3"
    # print('PS_3 LEVELS: \n', levels_matched.loc[levels_matched['source_tag'] == 'PS_3'])

    # Scale unc based on j over j_max.
    # levels_matched['unc'] = levels_matched.apply(
    #     lambda x: scale_uncertainty(std=x['pe_extrapolate_energy_shift_std'], std_scale=2, j_val=x['j'],
    #                                 j_max=x['j_max'], j_scale=0.05)
    #     if math.isnan(x['energy_final']) and not math.isnan(x['energy_calc']) and not math.isnan(x['j_max'])
    #        and x['j'] > x['j_max'] else x['unc'], axis=1)

    levels_matched["unc"] = levels_matched.apply(
        lambda x: scale_uncertainty2(
            std=x["pe_extrapolate_energy_shift_std"],
            a=0.0001,
            b=0.05,
            j_val=x["j"],
            j_max=x["j_max"],
        )
        if math.isnan(x["energy_final"])
        and not math.isnan(x["energy_calc"])
        and not math.isnan(x["j_max"])
        and x["j"] > x["j_max"]
        else x["unc"],
        axis=1,
    )

    # levels_matched['unc3'] = levels_matched.apply(
    #     lambda x: estimate_uncertainty(j_val=x['j'], v_val=x['v'], a=0.0001, b=0.05)
    #     if math.isnan(x['energy_final']) and not math.isnan(x['energy_calc']) and not math.isnan(x['j_max'])
    #        and x['j'] > x['j_max'] else x['unc'], axis=1)

    # print(levels_matched.loc[levels_matched['source_tag'] == 'PS_3',
    #                          ['energy_final', 'j', 'v', 'unc', 'unc2', 'unc3']].sort_values(['v', 'j'],
    #                                                                                         ascending=[1, 1]))
    #
    # plt.scatter(levels_matched.loc[levels_matched['source_tag'] == 'PS_3', 'j'],
    #             levels_matched.loc[levels_matched['source_tag'] == 'PS_3', 'unc'], color='r', label='Linear-J Unc.')
    # plt.scatter(levels_matched.loc[levels_matched['source_tag'] == 'PS_3', 'j'],
    #             levels_matched.loc[levels_matched['source_tag'] == 'PS_3', 'unc2'], color='b', label='Quadratic-J Unc.')
    # plt.scatter(levels_matched.loc[levels_matched['source_tag'] == 'PS_3', 'j'],
    #             levels_matched.loc[levels_matched['source_tag'] == 'PS_3', 'unc3'], color='g',
    #             label='Quadratic J-J$_{max}$ Unc')
    # plt.scatter(levels_matched.loc[levels_matched['source_tag'] == sfmt.SOURCE_TAG_MARVELISED, 'j'],
    #             levels_matched.loc[levels_matched['source_tag'] == sfmt.SOURCE_TAG_MARVELISED, 'unc'], color='k', label='Marvel Unc.')
    # plt.ylabel(ylabel='Uncertainty (cm$^{-1}$)')
    # plt.xlabel(xlabel='J')
    # # plt.ylim(bottom=-0.3, top=0.1)
    # plt.legend(loc='upper left', prop={'size': 7})
    # plt.savefig(r'D:\PhD\AlO\Marvel+PE_unc_plot.jpg', dpi=700)
    # plt.show()
    #
    # del levels_matched['unc2']
    # del levels_matched['unc3']

    print("NEW METHOD: \n", levels_matched.loc[levels_matched["source_tag"] == "PS_3"])

    levels_matched["energy_final"] = np.where(
        (levels_matched["energy_final"].isna())
        & (~levels_matched["j_max"].isna())
        & (~levels_matched["energy_calc"].isna())
        & (levels_matched["j"] > levels_matched["j_max"]),
        levels_matched["energy_calc"] + levels_matched["pe_extrapolate_energy_shift"],
        levels_matched["energy_final"],
    )

    del levels_matched["j_max"]
    del levels_matched["pe_extrapolate_energy_shift"]
    del levels_matched["pe_extrapolate_energy_shift_std"]

    # UNNECESSARY IF NOT OUTPUTTING THESE DATAFRAMES.
    # Add shift predictions to shift table.
    # What to do when unc = NaN?
    shift_table_full_j = shift_table.copy()
    shift_table_full_j = shift_table_full_j.rename(
        columns={
            "energy_dif_mean": "energy_shift",
            "energy_dif_std": "energy_shift_unc",
        }
    )
    pe_fit_shifts = pe_fit_shifts.rename(
        columns={
            "pe_fit_energy_shift": "energy_shift",
            "pe_fit_unc": "energy_shift_unc",
        }
    )
    shift_table_full_j = shift_table_full_j.append(pe_fit_shifts)
    shift_table_full_j = shift_table_full_j.sort_values(
        by=["state", "v", "Omega", "j"], ascending=[1, 1, 1, 1]
    )

    pe_extrapolate_shifts = pe_extrapolate_shifts.rename(
        columns={
            "pe_extrapolate_energy_shift": "extrapolation_energy_shift",
            "pe_extrapolate_energy_shift_std": "extrapolation_energy_unc",
        }
    )

    return levels_matched
