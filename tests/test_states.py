from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from linelisttools.concurrence import ExecutorType
from linelisttools.format import SourceTag
from linelisttools.marvel import read_marvel_energies
from linelisttools.states import (
    ExoMolStatesHeader,
    match_states,
    predict_shifts,
    read_exomol_states,
    set_calc_states,
    shift_parity_pairs,
)

# @pytest.fixture(scope="session")
# def alo_states_file():
#     return (Path(__file__).parent / r"./inputs/27Al-16O__ATP_Modified.states").resolve()
#
#
# @pytest.fixture(scope="session")
# def alo_marvel_energies_file():
#     return (
#         Path(__file__).parent / r"./inputs/AlO_MARVEL_Energies_1.9-clean.txt"
#     ).resolve()


@pytest.fixture(scope="session")
def so_states_file():
    return (Path(__file__).parent / r"./inputs/SO.states").resolve()


@pytest.fixture(scope="session")
def so_marvel_energies_file():
    return (Path(__file__).parent / r"./inputs/SO_MarvelEnergies.txt").resolve()


def test_so_states(
    so_states_file,
    so_marvel_energies_file,
):
    header = ExoMolStatesHeader(
        unc=None,
        lifetime=None,
        parity=["parity_tot", "parity_rotless"],
        symmetry="state",
        vibrational_qn="v",
        other_qn=["Lambda", "Sigma", "Omega"],
        source_tag=None,
    )
    df_duo = read_exomol_states(so_states_file, exomol_states_header=header)
    print(f"Number of Duo states: {len(df_duo)}")
    print("Duo states: \n", df_duo)

    df_marvel = pd.read_csv(
        so_marvel_energies_file,
        delim_whitespace=True,
        names=[
            "i",
            "energy",
            "state",
            "degeneracy",
            "J",
            "parity_tot",
            "parity_rotless",
            "v",
            "Lambda",
            "Sigma",
            "Omega",
            "unc",
        ],
    )
    df_marvel = df_marvel[["state", "v", "Omega", "parity_tot", "J", "energy", "unc"]]
    print("Marvel levels: \n", df_marvel)

    df_match = match_states(
        states_calc=df_duo,
        states_obs=df_marvel,
        qn_match_cols=["J", "parity_tot", "state", "v", "Omega"],
        match_source_tag=SourceTag.MARVELISED,
        states_header=header,
        # states_new_qn_cols=[
        #     # "state", "v",
        #     # "Omega",
        #     # "Sigma",
        #     # "Lambda",
        # ],
        # overwrite_non_match_qn_cols=True,
    )
    print(df_match)

    df_match_check = df_match.loc[df_match["source_tag"] == SourceTag.MARVELISED]
    print(f"Number of Marvel states: {len(df_match_check)}")
    print(
        f"Absolute mean Obs.-Calc.: {df_match_check['energy_dif'].abs().mean()},"
        f" Std.Dev.: {df_match_check['energy_dif'].abs().std()}"
    )
    rms_check = df_match_check.loc[~df_match_check["energy_dif"].isna(), "energy_dif"]
    print("RMS: ", np.sqrt(sum(rms_check**2) / len(rms_check)))
    del rms_check

    df_match["Omega_fit"] = df_match.apply(
        lambda x: x["Omega"] * -1 if x["parity_tot"] == "-" else x["Omega"], axis=1
    )

    df_match = shift_parity_pairs(
        states=df_match,
        states_header=header,
        shift_table_qn_cols=["state", "v", "Omega_fit", "J"],
    )
    print(
        "States with PS_1 parity pair Predicted Shifts: \n",
        df_match.loc[df_match["source_tag"] == SourceTag.PS_PARITY_PAIR],
    )

    df_match = predict_shifts(
        states_matched=df_match,
        fit_qn_list=["state", "v", "Omega_fit"],
        states_header=header,
        j_segment_threshold_size=10,
        show_plot=True,
        plot_states=["A3Pi"],
        executor_type=ExecutorType.THREADS,
        n_workers=8,
    )
    print(
        "States with linear regression (PS_2) & extrapolation (PS_3) Predicted Shifts: \n",
        df_match,
    )

    df_match = set_calc_states(
        states=df_match,
        states_header=header,
        unc_j_factor=0.0001,
        unc_v_factor=0.05,
    )
    print("States with predicted Calculated (Ca) Unc.: \n", df_match)


def test_exomolstatesheader_formatting():
    header = ExoMolStatesHeader(
        is_hyperfine=False,
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


# TODO: Add parity_total_to_norot and parity_norot_to_total tests.
