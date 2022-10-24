from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from linelisttools.concurrence import ExecutorType
from linelisttools.format import SourceTag
from linelisttools.states import (  # match_levels,
    ExoMolStatesHeader,
    match_states,
    predict_shifts,
    read_exomol_states,
    read_mvl_energies,
    set_calc_states,
    shift_parity_pairs,
)


@pytest.fixture(scope="session")
def alo_marvel_energies_file():
    return (
        Path(__file__).parent / r"./inputs/AlO_MARVEL_Energies_1.9-clean.txt"
    ).resolve()


@pytest.fixture(scope="session")
def alo_states_file():
    return (Path(__file__).parent / r"./inputs/27Al-16O__ATP_Modified.states").resolve()


@pytest.mark.parametrize(
    "marvel_qn_cols, qn_match_cols, match_source_tag, shift_table_qn_cols, levels_new_qn_cols, suffixes, "
    "energy_col, unc_col, j_col, parity_col, v_col, source_tag_col, id_col, is_isotopologue_match, "
    "overwrite_non_match_qn_cols, j_segment_threshold_size",
    [
        (
            [
                "state",
                "v",
                "J",
                "N",
                "fs",
                ExoMolStatesHeader.StatesParity.TOTAL_PARITY.value,
            ],
            [
                "state",
                "v",
                "J",
                ExoMolStatesHeader.StatesParity.TOTAL_PARITY.value,
                "Omega",
            ],
            SourceTag.MARVELISED,
            ["state", "v", "Omega", "J"],
            None,
            ("_calc", "_obs"),
            "energy",
            "unc",
            "J",
            ExoMolStatesHeader.StatesParity.TOTAL_PARITY.value,
            "v",
            "source_tag",
            "id",
            False,
            False,
            14,
        )
    ],
)
def test_alo_states(
    alo_marvel_energies_file,
    marvel_qn_cols,
    alo_states_file,
    qn_match_cols,
    match_source_tag,
    shift_table_qn_cols,
    levels_new_qn_cols,
    suffixes,
    energy_col,
    unc_col,
    j_col,
    parity_col,
    v_col,
    source_tag_col,
    id_col,
    is_isotopologue_match,
    overwrite_non_match_qn_cols,
    j_segment_threshold_size,
):
    pd.set_option("display.max_columns", None)
    levels_new = read_mvl_energies(
        alo_marvel_energies_file,
        marvel_qn_cols,
        energy_cols=["energy", "unc", "degree"],
    )
    print(levels_new)

    def temp_set_omega(state: str, fine_struct: str) -> float:
        series = state[0]
        if series in ["X", "B", "D", "F"]:
            # For AlO these series are all Sigma states.
            return 0.5
        elif series == "A":
            if fine_struct == "F1":
                return 1.5
            elif fine_struct == "F2":
                return 0.5
        elif series == "C":
            if fine_struct == "F1":
                return 0.5
            elif fine_struct == "F2":
                return 1.5
        elif series == "E":
            if fine_struct == "F1":
                return 2.5
            elif fine_struct == "F2":
                return 1.5
        else:
            return np.nan

    levels_new["Omega"] = levels_new.apply(
        lambda x: temp_set_omega(x["state"], x["fs"]), axis=1
    )
    # levels_initial = pd.read_csv(
    #     alo_states_file,
    #     delim_whitespace=True,
    #     names=[
    #         id_col,
    #         energy_col,
    #         "degeneracy",
    #         j_col,
    #         "lifetime",
    #         parity_col,
    #         ExoMolStatesHeader.StatesParity.ROTATIONLESS_PARITY.value,
    #         "state",
    #         v_col,
    #         "Lambda",
    #         "Sigma",
    #         "Omega",
    #     ],
    # )
    states_header = ExoMolStatesHeader(
        unc=None,
        parity=[
            ExoMolStatesHeader.StatesParity.TOTAL_PARITY,
            ExoMolStatesHeader.StatesParity.ROTATIONLESS_PARITY,
        ],
        symmetry="state",
        vibrational_qn=v_col,
        other_qn=["Lambda", "Sigma", "Omega"],
        source_tag=None,
    )
    states_calc = read_exomol_states(alo_states_file, states_header)
    print(states_calc)
    states_matched = match_states(
        states_calc=states_calc,
        states_obs=levels_new,
        qn_match_cols=qn_match_cols,
        match_source_tag=match_source_tag,
        states_header=states_header,
        states_new_qn_cols=levels_new_qn_cols,
        suffixes=suffixes,
        is_isotopologue_match=is_isotopologue_match,
        overwrite_non_match_qn_cols=overwrite_non_match_qn_cols,
    )
    # matched_states = match_levels(
    #     levels_initial=levels_initial,
    #     levels_new=levels_new,
    #     qn_match_cols=qn_match_cols,
    #     match_source_tag=match_source_tag,
    #     levels_new_qn_cols=levels_new_qn_cols,
    #     suffixes=suffixes,
    #     energy_col=energy_col,
    #     unc_col=unc_col,
    #     source_tag_col=source_tag_col,
    #     id_col=id_col,
    #     is_isotopologue_match=is_isotopologue_match,
    #     overwrite_non_match_qn_cols=overwrite_non_match_qn_cols,
    # )
    # print(states_matched)
    # print(matched_states)
    # assert matched_states.equals(states_matched)
    shift_table_qn_cols.remove(states_header.rigorous_qn)
    states_matched = predict_shifts(
        states_matched=states_matched,
        fit_qn_list=shift_table_qn_cols,
        states_header=states_header,
        j_segment_threshold_size=j_segment_threshold_size,
        show_plot=True,
        executor_type=ExecutorType.THREADS,
        n_workers=8,
    )
    print(states_matched)
    energy_calc_col = states_header.energy + "_calc"
    energy_obs_col = states_header.energy + "_obs"
    energy_dif_col = states_header.energy + "_dif"
    energy_final_col = states_header.energy + "_final"
    states_matched = shift_parity_pairs(
        states=states_matched,
        states_header=states_header,
        shift_table_qn_cols=shift_table_qn_cols,
        energy_calc_col=energy_calc_col,
        energy_obs_col=energy_obs_col,
        energy_dif_col=energy_dif_col,
        energy_final_col=energy_final_col,
    )
    states_matched = set_calc_states(
        states=states_matched,
        states_header=states_header,
        unc_j_factor=0.0001,
        unc_v_factor=0.05,
        energy_final_col=energy_final_col,
        energy_calc_col=energy_calc_col,
    )
    states_matched = states_matched.sort_values(
        by=[states_header.rigorous_qn, parity_col, energy_final_col]
    )
    states_matched[states_header.state_id] = np.arange(1, len(states_matched) + 1)
    print(states_matched)


def test_exomolstatesheader_formatting():
    header = ExoMolStatesHeader(
        rigorous_qn="F",
        unc=None,
        parity=ExoMolStatesHeader.StatesParity.ROTATIONLESS_PARITY,
        symmetry="sym",
        isomer=["ns_iso", "iso"],
        vibrational_qn=["v1", "v2"],
        other_qn=["J", "Omega"],
        source_tag="BEANS",
    )
    header_out = header.get_header()
    print(header_out)
