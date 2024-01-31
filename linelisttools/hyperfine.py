import functools
import math
import typing as t
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import tqdm
from sympy.physics.wigner import wigner_6j

from .concurrence import yield_grouped_data
from .format import create_tag

__DEPERTURB_SCALE_FACTOR_DEFAULT = 15.0
__HF_PRESENCE_SCALE_FACTOR_DEFAULT = 4.0
__HF_SKEW_SCALE_FACTOR_DEFAULT = 4.0


def calc_relative_intensity(
    f_u: float,
    f_l: float,
    k: float,
    j_l: float,
    j_u: float,
    nuclear_spin: float,
    source: str,
) -> float:
    # try:
    w6j = wigner_6j(f_u, f_l, k, j_l, j_u, nuclear_spin)
    # except ValueError:
    #     print(
    #         f"Wigner 6j: arguments not integre/half-integer or do not satisfy triangle relation for {source}: {f_u, f_l,
    #         k, j_l, j_u, nuclear_spin}.)
    return ((2 * f_u) + 1) * ((2 * f_l) + 1) * w6j**2


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
    """
    Determines the present values of DeltaF from the input F_u -> F_l transitions and then calculates the number of
    possible hyperfine transitions with those values of DeltaF between J_u and J_l.

    The list of upper and lower state F quantum numbers should be the same length, such that values of F at each index
    in both lists represent the upper and lower state values of F for a measured hyperfine transition.

    Args:
        nuclear_spin: The nuclear spin of the molecule.
        j_u:          The upper state J quantum number.
        j_l:          The lower state J quantum number.
        f_u_list:     The upper state values of F for which measured hyperfine transitions exist.
        f_l_list:     The lower state values of F for which measured hyperfine transitions exist.

    Returns:
        The integer number of possible hyperfine transitions allowed between the upper and lower states for a given set
            of observed DeltaF values.
    """
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
        hf_presence_scale_factor = __HF_PRESENCE_SCALE_FACTOR_DEFAULT
    # When the number of trans in the data frame is equal to the number of possible transitions, the scale factor
    # returned is 1. When only 1 of the possible transitions is present (provided more than 1 are possible, the scale
    # factor is 10. This relationship is linear relative to the fraction of possible transitions available.
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
        hf_skew_scale_factor = __HF_SKEW_SCALE_FACTOR_DEFAULT
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

    present_f_centres = [
        np.mean([f_pair[0], f_pair[1]]) for f_pair in present_f_pair_list
    ]
    mean_present_f_centre = np.mean(present_f_centres)
    abs_f_centre_offset = abs(mean_present_f_centre - mean_possible_f_centre)
    # print(abs_f_centre_offset)

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
    grouped_data: t.Tuple[t.List[str], pd.DataFrame],
) -> t.List[t.Union[str, float, int]]:
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


# def yield_transition_group(grouped_data):
#     for group_name, df_group in grouped_data:
#         yield group_name, df_group


def deperturb_hyperfine(
    transitions: pd.DataFrame,
    qn_list: t.List[str],
    nuclear_spin: float,
    energy_col: str = "energy",
    unc_col: str = "unc",
    j_col: str = "J",
    f_col: str = "F",
    source_col: str = "source",
    intensity_col: str = "relative_intensity",
    suffixes: t.Tuple[str, str] = ("_u", "_l"),
    hf_presence_scale_factor: float = None,
    hf_skew_scale_factor: float = None,
    deperturb_scale_factor: float = None,
    n_workers: int = 8,
) -> pd.DataFrame:
    if len({energy_col, unc_col, source_col}.difference(transitions.columns)) > 0:
        raise RuntimeError(
            "The following columns were not found in the transition DataFrame: ",
            {energy_col, unc_col, source_col}.difference(transitions.columns),
        )

    if f_col in qn_list:
        qn_list.remove(f_col)

    transitions[intensity_col] = transitions.apply(
        lambda x: calc_relative_intensity(
            x[f_col + suffixes[0]],
            x[f_col + suffixes[1]],
            1,
            x[j_col + suffixes[1]],
            x[j_col + suffixes[0]],
            nuclear_spin,
            x[source_col],
        ),
        axis=1,
    )
    transitions[source_col] = transitions[source_col].map(lambda x: x.split(".")[0])
    transitions = transitions.drop_duplicates(keep="first")

    group_by_cols = [qn + state_label for state_label in suffixes for qn in qn_list] + [
        source_col
    ]
    if len(set(group_by_cols).difference(transitions.columns)) > 0:
        raise RuntimeError(
            "The following columns were not found in the transition DataFrame: ",
            set(group_by_cols).difference(transitions.columns),
        )

    transitions_grouped = transitions.loc[transitions[energy_col] >= 0].groupby(
        by=group_by_cols, as_index=False
    )

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
        j_col + suffixes[0],
        j_col + suffixes[1],
        f_col + suffixes[0],
        f_col + suffixes[1],
        intensity_col,
        hf_presence_scale_factor,
        hf_skew_scale_factor,
    )

    deperturbed_list = []
    with ThreadPoolExecutor(max_workers=n_workers) as e:
        for result in tqdm.tqdm(e.map(worker, yield_grouped_data(transitions_grouped))):
            deperturbed_list.append(result)

    deperturbed_data = pd.DataFrame(data=deperturbed_list, columns=output_cols)
    if deperturb_scale_factor is None:
        deperturb_scale_factor = __DEPERTURB_SCALE_FACTOR_DEFAULT
    deperturbed_data[unc_col] = (
        deperturbed_data[unc_col + "_wm"]
        * deperturbed_data["hfs_presence"]
        * deperturbed_data["hfs_skew"]
        * deperturb_scale_factor
    )
    # Remove any energies with nan results, i.e.: |DeltaJ| > 1 transitions that cannot have intensities approximated
    # through this method.
    deperturbed_data = deperturbed_data.loc[
        ~deperturbed_data[energy_col + "_wm"].isna()
    ]

    for source in deperturbed_data[source_col].unique():
        source_count = len(deperturbed_data.loc[deperturbed_data[source_col] == source])
        tag_length = int(math.log10(source_count)) + 1
        deperturbed_data.loc[deperturbed_data[source_col] == source, source_col] = [
            create_tag(source, tag_num, tag_length)
            for tag_num in np.arange(1, source_count + 1)
        ]

    return deperturbed_data


def perturb_hyperfine(
    states_hfr: pd.DataFrame,
    states_hfu: pd.DataFrame,
    qn_list: t.List[str],
    splitting_dependent_qn_list: t.List[str],
    nuclear_spin: float,
    f_col: str = "F",
) -> pd.DataFrame:
    """
    WIP

    Takes in a set of calculated hyperfine-resolved states and empirical hyperfine-unresolved states. The calculated
    hyperfine-resolved states are deperturbed and compared to the empirical hyperfine-unresolved states in order to
    identify an obs.-calc. shift, which is then applied to the calculated hyperfine-resolved states such that their
    deperturbed values would be consistent with observational data.

    # TODO: Should all hyperfine components be shifted equally?
    # TODO: How to include comparison to any empirical, hyperfine-resolved data?

    Args:
        states_hfr:
        states_hfu:
        qn_list:
        splitting_dependent_qn_list:
        nuclear_spin:
        f_col:

    Returns:

    """
    # Approaches that fail:
    #
    # Approach 1, does not seem good as ambiguity in whether shift arises from upper or lower level of trans.
    # 1) Determine calculated transition frequencies from hyperfine-resolved calculated states that correspond to
    # hyperfine-unresolved observed transitions.
    # 2) Deperturb these derived calculated frequencies.
    # 3) Measure shift in each transition.
    # 4) Apply shift back to original set of hyperfine data.
    #
    # Approach 2
    # 1) Determine calculated transition frequencies from hyperfine-resolved calculated states that correspond to
    # hyperfine-unresolved observed transitions.
    # 2) Calculate residual obs.-calc. between transitions.
    # 3) Pass into minimiser, include a floated shift array that shifts each matching relevant hyperfine resolved level
    # in the original such that it shifts all states that are relevant to determining matching transitions frequencies
    # 4) Determine minimum when obs.-calc. residual between observed hyperfine-unresolved and calculated, deperturbed
    # hyperfine is a minimum, finding optimal shift for each state.
    # ISSUES: Many hundreds or more parameter problem, does not guarantee the hyperfine components of a level to be
    # deperturbed are shifted together in a physical way/potential loss of physical splitting in hyperfine levels.

    # DISCUSSION WITH JT:
    # If hyperfine independent of v - copy splitting in v=0 to same assignment at higher v.
    # For doublets where no hf splitting observed - have no hf splitting so all hf components of given level are on top
    # of each other but set unc based on the resolution limit for experiments where hf splitting was not observed.
    # Czech results may imply that hyperfine is v dependent. Doesn't seem to be!
    # Cheaty way to do splittings: EH constants.

    # 26/10: Assume input hfr levels are the full set of levels, where the obs MARVEL levels have been matched to the
    # computed set. This means that splittings where we have MARVEL data will be "corrected", and we use computed
    # splittings elsewhere.

    # TODO: 01/11: Potentially add C and B to model; splitting likely smaller than X.
    #  Fix X state vibrational band origins.
    #  Add other states to Duo with no hyperfine; fix T_0 where known better.
    #  Then add marvel on top, prioritising hf marvel then take hfu where absent.

    # Step -1: Calculate all F values needed
    # j_hf_splitting = [
    #     [j_val, calc_possible_f_values(nuclear_spin=nuclear_spin, j_value=j_val)]
    #     for j_val in states_hfu["J"].unique()
    # ]
    # j_hf_splitting = [
    #     [j_val, f_val] for j_val, f_list in j_hf_splitting for f_val in f_list
    # ]
    j_hf_splitting = [
        [j_val, f_val]
        for j_val in states_hfu["J"].unique()
        for f_val in calc_possible_f_values(nuclear_spin=nuclear_spin, j_value=j_val)
    ]
    j_hf_splitting = pd.DataFrame(j_hf_splitting, columns=["J", "F"])
    states_hfu = states_hfu.merge(j_hf_splitting, on=["J"], how="inner")

    # Step 0: Measure each F component shift from center.
    splitting_dependent_hfu_qn_list = list(
        set(splitting_dependent_qn_list) - set(f_col)
    )
    states_hrf_mean = states_hfr.groupby(
        by=splitting_dependent_hfu_qn_list, as_index=False
    ).agg({"energy": "mean"})
    states_hrf_mean.columns = splitting_dependent_hfu_qn_list + ["energy_mean"]

    # Step 1: Get splitting by state/fs/parity/J/F
    states_hfr = states_hfr.merge(
        states_hrf_mean, on=splitting_dependent_hfu_qn_list, how="inner"
    )
    states_hfr["splitting"] = states_hfr.apply(
        lambda x: x["energy"] - x["energy_mean"], axis=1
    )
    # Take the mean of the splitting for a given hf qn set on the off chance we do have multiple v (or other splitting
    # independent qn) for each instance of those qn.
    states_hfr = states_hfr.groupby(by=splitting_dependent_qn_list, as_index=False).agg(
        {"splitting": "mean"}
    )

    # Step 2: Determine which states to apply no splitting to (i.e.: Doublets)

    # Step 3: Apply splitting to hfu states where equivalent exist in states_hfr
    states_perturbed = states_hfu.merge(states_hfr, on=qn_list, how="left")
    # Where there is no splitting merged from the hfr calculations, set the splitting to 0.
    states_perturbed["splitting"] = states_perturbed["splitting"].fillna(0)
    states_perturbed["energy_hfr"] = (
        states_perturbed["energy"] + states_perturbed["splitting"]
    )

    # Step 4: Apply uncertainties; estimate based opn experimental resolution for states with no splitting where the
    # size of hf splitting must be below experimental resolution.

    return states_perturbed
