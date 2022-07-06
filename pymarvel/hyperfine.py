import functools
import math
import typing as t
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import tqdm


def calculate_weighted_mean_energy(
    energy_list: t.List[float], weight_list: t.List[float]
) -> float:
    if sum(weight_list) != 0:
        return float(np.average(energy_list, weights=weight_list))
    else:
        return np.nan


def propagate_weighted_mean_unc(
    unc_list: t.List[float], weight_list: t.List[float]
) -> float:
    unc_list = list(unc_list)
    weight_list = list(weight_list)
    # print('Propagating errors:')
    # print(f'unc_list: {unc_list}')
    # print(f'weight_list: {weight_list}')
    weight_total = sum(weight_list)
    if weight_total != 0:
        # print(f'weight_total: {weight_total}')
        # norm_weight_list = weight_list/weight_total
        norm_weight_list = [w / weight_total for w in weight_list]
        # print(f'norm_weight_list: {norm_weight_list}')
        # norm_sq_weight_list = norm_weight_list ** 2
        norm_sq_weight_list = [n**2 for n in norm_weight_list]
        # print(f'norm_sq_weight_list: {norm_sq_weight_list}')

        # unc_sq_list = unc_list ** 2
        unc_sq_list = [u**2 for u in unc_list]
        # print(f'unc_sq_list: {unc_sq_list}')

        # prod_list = norm_sq_weight_list * unc_sq_list
        prod_list = [w * u for w, u in zip(norm_sq_weight_list, unc_sq_list)]
        # print(f'prod_list: {prod_list}')
        norm_sq_error_sum = sum(prod_list)
        # print(f'norm_sq_error_sum: {norm_sq_error_sum}')
        error = math.sqrt(norm_sq_error_sum)
        # print(f'error: {error}')
        return error
    else:
        return np.nan


def calculate_num_possible_hf_transitions(
    j_u: float, j_l: float, f_u_list: t.List[float], f_l_list: t.List[float]
) -> int:
    # Determined present DeltaF values:
    present_f_pair_list = list(zip(f_u_list, f_l_list))
    # print('PRESENT F PAIRS: ', present_f_pair_list)
    present_delta_f_list = list(
        set([int(f_u - f_l) for f_u, f_l in present_f_pair_list])
    )
    # print('PRESENT DELTA F VALUES: ', present_delta_f_list)

    # 0 indexing necessary/check if list first?
    j_u = j_u[0]
    j_l = j_l[0]
    # print(f'J_U={j_u}, J_L={j_l}')
    min_j = min((j_u, j_l))
    # TODO: Change to use same method as calculate_hfs_skew()?
    # If min_j is >= I(=3.5) then this value is 8(=2I+1). Decreases by 2 for each J below this.
    if min_j == 0.5:
        max_delta_f_trans = 2
    elif min_j == 1.5:
        max_delta_f_trans = 4
    elif min_j == 2.5:
        max_delta_f_trans = 6
    else:
        max_delta_f_trans = 8

    max_j = max((j_u, j_l))

    # If the maximum number of f trans is in [2,4,6], i.e.: is defined by min_j, and min_j is not equal to max_j, then
    # this is the limiting factor and we can set the number of trans for each branch as this value.
    if max_delta_f_trans in [2, 4, 6] and max_j > min_j:
        num_delta_f_trans = np.repeat(max_delta_f_trans, len(present_delta_f_list))
        # print(f'(if) num_Delta_f_trans = {num_delta_f_trans}')
    else:
        delta_j = j_u - j_l
        # print(f'DELTA_J={delta_j}')
        num_delta_f_trans = [
            int(max_delta_f_trans - abs(delta_j - delta_f))
            for delta_f in present_delta_f_list
        ]
        # print(f'(else) num_Delta_f_trans = {num_delta_f_trans}')

    num_trans = sum(num_delta_f_trans)
    return num_trans


def calculate_hfs_presence(
    possible_hf_trans: int,
    present_hf_trans: int,
    hfs_presence_scale_factor: float = 4.0,
) -> float:
    # When the number of trans in the data frame is equal to the number of possible transitions, the scale factor
    # returned is 1. When only 1 of the possible transitions is present (provided more than 1 are possible, the scale
    # factor is 10. This relationship is linear relative to the fraction of possible transitions available.
    # if possible_hf_trans == 1:
    #     return 1
    # else:
    return 1 + (
        hfs_presence_scale_factor
        * (possible_hf_trans - present_hf_trans)
        / (possible_hf_trans - 1)
    )


def calculate_hfs_skew(
    nuclear_spin: float,
    j_u: float,
    j_l: float,
    f_u_list: t.List[float],
    f_l_list: t.List[float],
    hfs_skew_scale_factor: float = 4.0,
) -> float:
    # 1: Determine present DeltaF values in input f_u/f_l lists.
    present_f_pair_list = list(zip(f_u_list, f_l_list))
    # print('PRESENT F PAIRS: ', present_f_pair_list)
    present_delta_f_list = list(
        set([int(f_u - f_l) for f_u, f_l in present_f_pair_list])
    )
    # print('PRESENT DELTA F VALUES: ', present_delta_f_list)
    # if len(present_delta_f_list) > 1:
    #     print(colored(np.repeat('*******************', 1000), 'green'))

    # 2: Calculate full mu_DeltaF for the j_u->j_l passed.
    j_u = j_u[0]
    j_l = j_l[0]
    # print(f'J_L: {j_l}, J_U: {j_u}')

    min_f_l = abs(j_l - nuclear_spin)
    f_is_half_integral = False
    if min_f_l % 1 == 0.5:
        f_is_half_integral = True
    max_f_l = j_l + nuclear_spin
    # print(f'MIN F_L: {min_f_l}, MAX F_L: {max_f_l}')
    if f_is_half_integral:
        possible_f_l = [
            x / 2 for x in range(int(min_f_l * 2), int((max_f_l + 1) * 2), 2)
        ]
    else:
        possible_f_l = list(range(int(min_f_l), int(max_f_l) + 1))
    # print('POSSIBLE F_L: ', possible_f_l)

    min_f_u = abs(j_u - nuclear_spin)
    max_f_u = j_u + nuclear_spin
    # print(f'MIN F_U: {min_f_u}, MAX F_U: {max_f_u}')
    if f_is_half_integral:
        possible_f_u = [
            x / 2 for x in range(int(min_f_u * 2), int((max_f_u + 1) * 2), 2)
        ]
    else:
        possible_f_u = list(range(int(min_f_u), int(max_f_u) + 1))
    # print('POSSIBLE F_U: ', possible_f_u)

    possible_f_pair_list = [
        [
            [f_u, f_l]
            for f_l in possible_f_l
            for f_u in possible_f_u
            if (f_u - f_l) == delta_f
        ]
        for delta_f in present_delta_f_list
    ]
    # print('POSSIBLE F PAIRS: ', possible_f_pair_list)
    possible_f_centres = [
        np.mean([pair[0], pair[1]])
        for pair_list in possible_f_pair_list
        for pair in pair_list
    ]
    # print('POSSIBLE F CENTRES: ', possible_f_centres)
    mean_possible_f_centre = sum(possible_f_centres) / len(possible_f_centres)
    # print(f'MEAN POSSIBLE F CENTRE: {mean_possible_f_centre}')

    # 3: Calculate current mu_DeltaF for the j_u->j_l passed.
    present_f_centres = [np.mean([pair[0], pair[1]]) for pair in present_f_pair_list]
    mean_present_f_centre = np.mean(present_f_centres)
    # print(f'MEAN PRESENT F CENTRE: {mean_present_f_centre}')
    abs_f_centre_offset = abs(mean_present_f_centre - mean_possible_f_centre)

    return 1 + (hfs_skew_scale_factor * abs_f_centre_offset)


def calculate_deperturbation(
    nuclear_spin: float,
    energy_col: str,
    unc_col: str,
    j_col_u: str,
    j_col_l: str,
    f_col_u: str,
    f_col_l: str,
    intensity_col: str,
    hfs_presence_scale_factor: float,
    hfs_skew_scale_factor: float,
    grouped_data,
) -> t.List:
    df_group = grouped_data[1]
    energy_wm = calculate_weighted_mean_energy(
        energy_list=df_group[energy_col], weight_list=df_group[intensity_col]
    )
    unc_wm = propagate_weighted_mean_unc(df_group[unc_col], df_group[intensity_col])
    present_hf_trans = len(df_group.index)
    possible_hf_trans = calculate_num_possible_hf_transitions(
        df_group[j_col_u].values,
        df_group[j_col_l].values,
        df_group[f_col_u].values,
        df_group[f_col_l].values,
    )
    hfs_presence = calculate_hfs_presence(
        possible_hf_trans,
        present_hf_trans,
        hfs_presence_scale_factor=hfs_presence_scale_factor,
    )
    hfs_skew = calculate_hfs_skew(
        nuclear_spin,
        df_group[j_col_u].values,
        df_group[j_col_l].values,
        df_group[f_col_u].values,
        df_group[f_col_l].values,
        hfs_skew_scale_factor=hfs_skew_scale_factor,
    )

    out_list = list(grouped_data[0])
    out_list.extend(
        [energy_wm, unc_wm, present_hf_trans, possible_hf_trans, hfs_presence, hfs_skew]
    )
    return out_list


def yield_transition_group(grouped_data):
    for group_name, df_group in grouped_data:
        yield group_name, df_group


def deperturb_hyperfine(
    transitions: pd.DataFrame,
    qn_list: t.List[str],
    nuclear_spin: float,
    energy_col: str = "energy",
    unc_col: str = "unc",
    j_col: str = "J",
    f_col: str = "F",
    source_col: str = "source",
    intensity_col: str = "intensity",
    suffixes: t.Tuple[str, str] = ("_u", "_l"),
    hfs_presence_scale_factor: float = None,
    hfs_skew_scale_factor: float = None,
    n_workers: int = 8,
) -> pd.DataFrame:
    # TODO: Error handling for if {J, F} cols not in qn_list.
    if f_col in qn_list:
        qn_list.remove(f_col)
    transitions["source"] = transitions["source"].map(lambda x: x.split(".")[0])

    group_by_cols = [qn + state_label for state_label in suffixes for qn in qn_list] + [
        source_col
    ]
    transitions_grouped = transitions.loc[transitions[energy_col] >= 0].groupby(
        by=group_by_cols, as_index=False
    )

    j_col_u = j_col + suffixes[0]
    j_col_l = j_col + suffixes[1]
    f_col_u = f_col + suffixes[0]
    f_col_l = f_col + suffixes[1]

    output_cols = group_by_cols + [
        energy_col + "_wm",
        unc_col + "_wm",
        "present_hf_trans",
        "possible_hf_trans",
        "hfs_presence",
        "hfs_skew",
    ]

    worker = functools.partial(
        calculate_deperturbation,
        nuclear_spin,
        energy_col,
        unc_col,
        j_col_u,
        j_col_l,
        f_col_u,
        f_col_l,
        intensity_col,
        hfs_presence_scale_factor,
        hfs_skew_scale_factor,
    )

    deperturbed_list = []
    with ThreadPoolExecutor(max_workers=n_workers) as e:
        for result in tqdm.tqdm(
            e.map(worker, yield_transition_group(transitions_grouped))
        ):
            deperturbed_list.append(result)

    # TODO: Create arg to determine if present/possible hf trans are kept?

    return pd.DataFrame(data=deperturbed_list, columns=output_cols)
