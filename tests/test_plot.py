from pathlib import Path

import pandas as pd
import pytest

from pymarvel.plot import PlotType, plot_state_coverage
from pymarvel.states import read_mvl_energies


@pytest.fixture(scope="session")
def diatomic_energy_file():
    return (
        Path(__file__).parent / r"./inputs/VO_MARVEL_Energies-1.8.1-final.txt"
    ).resolve()


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
    colours = [
        "#EE7733",
        "#0077BB",
        "#EE3377",
        "#33BBEE",
        "#CC3311",
        "#009988",
        "#BBBBBB",
    ]
    diatomic_energies = read_mvl_energies(file=diatomic_energy_file, qn_cols=qn_cols)
    plot_state_coverage(
        energies=diatomic_energies,
        state_configuration_dict=state_configuration_dict,
        colours=colours,
        show=False,
        out_file=temp_plot_output,
        # plot_type=PlotType.VIOLIN,
        electron_configurations=electron_configurations,
        energy_col=energy_col,
    )
