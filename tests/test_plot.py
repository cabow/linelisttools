from pathlib import Path

import pandas as pd
import pytest

from linelisttools.marvel import read_marvel_energies
from linelisttools.plot import PlotType, plot_state_coverage, plot_states_by_source_tag
from linelisttools.states import SourceTag


@pytest.fixture(scope="session")
def diatomic_energy_file():
    return (
        Path(__file__).parent / r"./inputs/VO_MARVEL_Energies-1.8.1-final.txt"
    ).resolve()


@pytest.fixture(scope="session")
def diatomic_states_file():
    return (Path(__file__).parent / r"./inputs/27Al-16O_Marvel-2.0.states").resolve()


@pytest.fixture(scope="session")
def temp_plot_output():
    return (Path(__file__).parent / r"./outputs/temp-plot-output.png").resolve()


@pytest.mark.parametrize(
    "qn_cols,state_configuration_dict,qn_group_cols,electron_configurations,energy_col",
    [
        (
            ["state", "fs", "Omega", "parity", "v", "J"],
            {
                "X_4Sigma-": "$\\sigma\\delta^2$",
                "A'_4Phi": "$\\sigma\\delta\\pi$",
                "A_4Pi": "$\\sigma\\delta\\pi$",
                "A_4Pi_P": "$\\sigma\\delta\\pi$",
                "1_2Delta": "$\\sigma^2\\delta$",
                "B_4Pi": "$\\delta^2\\pi$",
                "B_4Pi_P": "$\\delta^2\\pi$",
                "1_2Sigma+": "$\\sigma\\delta^2$",
                "1_2Phi": "$\\sigma\\delta\\pi$",
                "1_2Pi": "$\\sigma^2\\pi$",
                "C_4Sigma-": "$\\delta^2\\sigma^*$",
                "C_4Sigma-_P": "$\\delta^2\\sigma^*$",
                "2_2Pi": "$\\sigma\\delta\\pi$",
                "D_4Delta": "$\\sigma\\delta\\sigma^*$",
                "D_4Delta_P": "$\\sigma\\delta\\sigma^*$",
                "2_2Delta": "$\\sigma\\delta\\sigma^*$",
                "3_2Delta": "$\\sigma\\delta\\sigma^*$",
            },
            ["state", "Omega"],
            [
                "$\\sigma^2\\sigma^*$",
                "$\\sigma^2\\pi$",
                "$\\sigma^2\\delta$",
                "$\\sigma\\delta\\pi$",
                "$\\sigma\\delta\\sigma^*$",
                "$\\sigma\\delta^2$",
                "$\\delta^2\\sigma^*$",
                "$\\delta^2\\pi$",
                "$\\delta^3$",
            ],
            "energy",
        )
    ],
)
def test_plot_state_coverage(
    diatomic_energy_file,
    qn_cols,
    state_configuration_dict,
    qn_group_cols,
    temp_plot_output,
    electron_configurations,
    energy_col,
):
    diatomic_energies = read_marvel_energies(
        marvel_energy_file=diatomic_energy_file, qn_list=qn_cols
    )
    plot_state_coverage(
        energies=diatomic_energies,
        # state_list=["X_4Sigma-", "1_2Phi", "1_2Sigma+"],
        state_configuration_dict=state_configuration_dict,
        show=False,
        out_file=temp_plot_output,
        plot_type=PlotType.VIOLIN,
        # electron_configurations=electron_configurations,
        energy_col=energy_col,
    )


# @pytest.mark.parametrize(
#     "georgi_marvel_energy_file,qn_list,state_list,outfile",
#     [
#         (
#             r"\path\to\georgi's\marvel\energies.txt",  # Update this!
#             [
#                 "state",
#                 "fs",
#                 "Omega",
#                 "parity",
#                 "v",
#                 "J",
#             ],  # The quantum numbers in your marvel energies
#             [
#                 "X_4Sigma-",
#                 "A_4Pi",
#             ],  # List the states in your molecule you want to plot in left-to-right order
#             r"\path\to\where\to\save\plot.png",  # Update this!
#         )
#     ],
# )
# def test_georgi(
#     georgi_marvel_energy_file,
#     qn_list,
#     state_list,
#     outfile,
# ):
#     diatomic_energies = read_marvel_energies(
#         marvel_energy_file=georgi_marvel_energy_file, qn_list=qn_list
#     )
#     plot_state_coverage(
#         energies=diatomic_energies,
#         state_list=state_list,
#         show=False,
#         out_file=outfile,
#         plot_type=PlotType.EVENT,  # Try changing to PlotType.VIOLIN for
#     )


@pytest.mark.parametrize(
    "show,id_col,energy_col,j_col,source_tag_col,plot_state_list,plot_source_list",
    [
        (
            True,
            "ID",
            "energy",
            "J",
            "source_tag",
            ["X2SIGMA+", "A2PI", "B2SIGMA+"],
            ["Ma", "PS_1", "PS_2", "EH"],
        )
    ],
)
def test_plot_states_by_source_tag(
    diatomic_states_file,
    show,
    temp_plot_output,
    id_col,
    energy_col,
    j_col,
    source_tag_col,
    plot_state_list,
    plot_source_list,
):
    states = pd.read_csv(
        diatomic_states_file,
        delim_whitespace=True,
        names=[
            id_col,
            energy_col,
            "degeneracy",
            j_col,
            "uncertainty",
            "lifetime",
            "parity_tot",
            "parity_norot",
            "state",
            "v",
            "Lambda",
            "Sigma",
            "Omega",
            source_tag_col,
        ],
    )
    parity_pair_merge_cols = ["state", "v", j_col, "Omega"]
    states_parity_pairs = states.loc[states[source_tag_col] == "PS"].merge(
        states.loc[states[source_tag_col] == SourceTag.MARVELISED],
        left_on=parity_pair_merge_cols,
        right_on=parity_pair_merge_cols,
        how="inner",
        suffixes=["", "_pair"],
    )
    states_parity_pairs_idxs = states_parity_pairs[id_col].unique()
    states.loc[
        states[id_col].isin(states_parity_pairs_idxs), source_tag_col
    ] = SourceTag.PS_PARITY_PAIR.value

    interp_bands_cols = ["state", "v", "Omega"]
    states_marvel_bands_max_j = (
        states.loc[states[source_tag_col] == SourceTag.MARVELISED.value]
        .groupby(by=interp_bands_cols, as_index=False)
        .agg({j_col: "max"})
    )
    j_max_col = j_col + "_max"
    states_marvel_bands_max_j.columns = interp_bands_cols + [j_max_col]
    df_states_interp = states.loc[states[source_tag_col] == "PS"].merge(
        states_marvel_bands_max_j, on=interp_bands_cols, how="inner"
    )
    df_states_interp_idxs = df_states_interp.loc[
        df_states_interp[j_col] < df_states_interp[j_max_col], id_col
    ].unique()
    states.loc[
        states[id_col].isin(df_states_interp_idxs), source_tag_col
    ] = SourceTag.PS_LINEAR_REGRESSION.value

    df_states_extrap_idxs = df_states_interp.loc[
        df_states_interp[j_col] >= df_states_interp[j_max_col], id_col
    ].unique()
    states.loc[
        states[id_col].isin(df_states_extrap_idxs), source_tag_col
    ] = SourceTag.PS_EXTRAPOLATION.value
    plot_states_by_source_tag(
        states=states,
        show=show,
        out_file=temp_plot_output,
        j_col=j_col,
        energy_col=energy_col,
        source_tag_col=source_tag_col,
        plot_state_list=plot_state_list,
        plot_source_list=plot_source_list,
    )
