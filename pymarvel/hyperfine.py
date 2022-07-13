import functools
import math
import typing as t
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import tqdm


def calc_weighted_mean_energy(
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


def calc_possible_f_values(
    nuclear_spin: float, j_value: float
) -> t.List[t.Union[int, float]]:
    min_f = abs(j_value - nuclear_spin)
    f_is_half_integral = False
    if min_f % 1 == 0.5:
        f_is_half_integral = True
    max_f = j_value + nuclear_spin
    if f_is_half_integral:
        possible_f = [x / 2 for x in range(int(min_f * 2), int((max_f + 1) * 2), 2)]
    else:
        possible_f = list(range(int(min_f), int(max_f) + 1))
    return possible_f


def calc_possible_hf_trans(
    nuclear_spin: float, j_u: float, j_l: float, delta_f_list: t.List[float]
) -> t.List[t.Tuple[t.Union[int, float], t.Union[int, float]]]:
    possible_f_l = calc_possible_f_values(nuclear_spin=nuclear_spin, j_value=j_l)
    possible_f_u = calc_possible_f_values(nuclear_spin=nuclear_spin, j_value=j_u)

    possible_f_pair_list = [
        [
            (f_u, f_l)
            for f_l in possible_f_l
            for f_u in possible_f_u
            if (f_u - f_l) == delta_f
        ]
        for delta_f in delta_f_list
    ]
    possible_f_pair_list = [
        pair for delta_f_pair_list in possible_f_pair_list for pair in delta_f_pair_list
    ]
    return possible_f_pair_list


def calc_num_possible_hf_trans(
    nuclear_spin: float,
    j_u: float,
    j_l: float,
    f_u_list: t.List[float],
    f_l_list: t.List[float],
) -> int:
    present_delta_f_list = list(
        set([int(f_u - f_l) for f_u, f_l in list(zip(f_u_list, f_l_list))])
    )
    return len(calc_possible_hf_trans(nuclear_spin, j_u, j_l, present_delta_f_list))


def calc_hf_presence(
    possible_hf_trans: int,
    present_hf_trans: int,
    hf_presence_scale_factor: float = None,
) -> float:
    if hf_presence_scale_factor is None:
        hf_presence_scale_factor = 4.0
    # When the number of trans in the data frame is equal to the number of possible transitions, the scale factor
    # returned is 1. When only 1 of the possible transitions is present (provided more than 1 are possible, the scale
    # factor is 10. This relationship is linear relative to the fraction of possible transitions available.
    # if possible_hf_trans == 1:
    #     return 1
    # else:
    print(hf_presence_scale_factor, possible_hf_trans, present_hf_trans)
    return 1 + (
        hf_presence_scale_factor
        * (possible_hf_trans - present_hf_trans)
        / (possible_hf_trans - 1)
    )


def calc_hf_skew(
    nuclear_spin: float,
    j_u: float,
    j_l: float,
    f_u_list: t.List[float],
    f_l_list: t.List[float],
    hf_skew_scale_factor: float = None,
) -> float:
    if hf_skew_scale_factor is None:
        hf_skew_scale_factor = 4.0
    present_f_pair_list = list(zip(f_u_list, f_l_list))
    present_delta_f_list = list(
        set([int(f_u - f_l) for f_u, f_l in present_f_pair_list])
    )
    possible_f_pair_list = calc_possible_hf_trans(
        nuclear_spin, j_u, j_l, present_delta_f_list
    )
    possible_f_centres = [
        np.mean([f_pair[0], f_pair[1]]) for f_pair in possible_f_pair_list
    ]
    mean_possible_f_centre = np.mean(possible_f_centres)

    present_f_centres = [np.mean([pair[0], pair[1]]) for pair in present_f_pair_list]
    mean_present_f_centre = np.mean(present_f_centres)
    abs_f_centre_offset = abs(mean_present_f_centre - mean_possible_f_centre)
    print(abs_f_centre_offset)

    return 1 + (hf_skew_scale_factor * abs_f_centre_offset)


def calc_deperturbation(
    nuclear_spin: float,
    energy_col: str,
    unc_col: str,
    j_col_u: str,
    j_col_l: str,
    f_col_u: str,
    f_col_l: str,
    intensity_col: str,
    hf_presence_scale_factor: float,
    hf_skew_scale_factor: float,
    grouped_data,
) -> t.List:
    df_group = grouped_data[1]
    energy_wm = calc_weighted_mean_energy(
        energy_list=df_group[energy_col], weight_list=df_group[intensity_col]
    )
    unc_wm = propagate_weighted_mean_unc(df_group[unc_col], df_group[intensity_col])
    present_hf_trans = len(df_group.index)
    possible_hf_trans = calc_num_possible_hf_trans(
        nuclear_spin,
        df_group[j_col_u].iloc[0],
        df_group[j_col_l].iloc[0],
        df_group[f_col_u].values,
        df_group[f_col_l].values,
    )
    hfs_presence = calc_hf_presence(
        possible_hf_trans,
        present_hf_trans,
        hf_presence_scale_factor=hf_presence_scale_factor,
    )
    hfs_skew = calc_hf_skew(
        nuclear_spin,
        df_group[j_col_u].iloc[0],
        df_group[j_col_l].iloc[0],
        df_group[f_col_u].values,
        df_group[f_col_l].values,
        hf_skew_scale_factor=hf_skew_scale_factor,
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
    hf_presence_scale_factor: float = None,
    hf_skew_scale_factor: float = None,
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
        calc_deperturbation,
        nuclear_spin,
        energy_col,
        unc_col,
        j_col_u,
        j_col_l,
        f_col_u,
        f_col_l,
        intensity_col,
        hf_presence_scale_factor,
        hf_skew_scale_factor,
    )

    deperturbed_list = []
    with ThreadPoolExecutor(max_workers=n_workers) as e:
        for result in tqdm.tqdm(
            e.map(worker, yield_transition_group(transitions_grouped))
        ):
            deperturbed_list.append(result)

    # TODO: Create arg to determine if present/possible hf trans are kept?

    return pd.DataFrame(data=deperturbed_list, columns=output_cols)
