from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pymarvel.concurrence import ExecutorType
from pymarvel.format import SourceTag
from pymarvel.states import match_levels, predict_shifts, read_mvl_energies


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
    "energy_col, unc_col, j_col, source_tag_col,is_isotopologue_match, overwrite_non_match_qn_cols",
    [
        (
            ["state", "v", "J", "N", "fs", "parity"],
            ["state", "v", "J", "parity", "Omega"],
            SourceTag.MARVELISED,
            ["state", "v", "Omega", "J"],
            None,
            ("_calc", "_obs"),
            "energy",
            "unc",
            "J",
            "source_tag",
            False,
            False,
        )
    ],
)
def test_alo_match_levels(
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
    source_tag_col,
    is_isotopologue_match,
    overwrite_non_match_qn_cols,
):
    pd.set_option("display.max_columns", None)
    levels_new = read_mvl_energies(alo_marvel_energies_file, marvel_qn_cols)
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
    levels_initial = pd.read_csv(
        alo_states_file,
        delim_whitespace=True,
        names=[
            "id",
            "energy",
            "degeneracy",
            "J",
            "lifetime",
            "parity",
            "parity_norot",
            "state",
            "v",
            "Lambda",
            "Sigma",
            "Omega",
        ],
    )
    matched_levels, shift_table = match_levels(
        levels_initial=levels_initial,
        levels_new=levels_new,
        qn_match_cols=qn_match_cols,
        match_source_tag=match_source_tag,
        shift_table_qn_cols=shift_table_qn_cols,
        levels_new_qn_cols=levels_new_qn_cols,
        suffixes=suffixes,
        energy_col_name=energy_col,
        unc_col_name=unc_col,
        source_tag_col=source_tag_col,
        is_isotopologue_match=is_isotopologue_match,
        overwrite_non_match_qn_cols=overwrite_non_match_qn_cols,
    )
    print(matched_levels)
    shift_table_qn_cols.remove(j_col)
    matched_levels = predict_shifts(
        levels_matched=matched_levels,
        shift_table=shift_table,
        fit_qn_list=shift_table_qn_cols,
        j_segment_threshold_size=14,
        show_plot=False,
        unc_col=unc_col,
        j_col=j_col,
        source_tag_col=source_tag_col,
        executor_type=ExecutorType.THREADS,
        n_workers=8,
    )
    print(matched_levels)
