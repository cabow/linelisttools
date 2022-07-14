import re
import typing as t
from enum import Enum
from itertools import cycle, islice

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class PlotType(Enum):
    VIOLIN = "Ca"
    EVENT = "Ma"


def get_state_tex(states: t.List[str]) -> t.Dict:
    """

    Args:
        states: List of raw state labels to be converted to tex.
    Returns:
        A dictionary containing a mapping from the raw state label to its tex markup.
    """
    states = set(states)
    state_tex_dict = {}
    state_regex = re.compile(r"(\w)(['`]?)_?(\d)(\w*)([-\+]?)")
    for state in states:
        state_match = state_regex.match(state)
        prime = "\\prime" if state_match.group(2) != "" else ""
        symmetry = state_match.group(5)
        symmetry_str = "^" + symmetry if symmetry != "" else ""
        state_tex = f"{state_match.group(1)}$^{{{prime}{state_match.group(3)}}}\\{state_match.group(4).capitalize()}{symmetry_str}$"
        state_tex_dict[state] = state_tex
    return state_tex_dict


def plot_state_coverage(
    energies: pd.DataFrame,
    state_configuration_dict: t.Dict,
    colours: t.List[str],
    show: bool = True,
    out_file: str = None,
    plot_type: PlotType = PlotType.EVENT,
    electron_configurations: t.List[str] = None,
    energy_col: str = "energy",
):
    if out_file is None and not show:
        raise RuntimeError(
            "No out_file specified and show set to False - nothing to do."
        )
    if plot_type not in (PlotType.EVENT, PlotType.VIOLIN):
        raise RuntimeError(
            f"State coverage plot only accepts PlotTypes of {PlotType.EVENT} (default) or {PlotType.VIOLIN}."
        )
    if electron_configurations is None:
        plot_config_list = list(set(state_configuration_dict.values()))
    else:
        plot_electron_configs = list(set(state_configuration_dict.values()))
        plot_config_list = list(
            sorted(
                set(electron_configurations).intersection(plot_electron_configs),
                key=lambda x: electron_configurations.index(x),
            )
        )

    print("Electron config order: ", plot_config_list)

    plot_config_states = [
        state
        for plot_config in plot_config_list
        for state, config in state_configuration_dict.items()
        if config == plot_config and not state.endswith("_P")
    ]
    print("Energy ordered states by electron config order", plot_config_states)

    plot_config_ticks = [
        state_configuration_dict.get(state) for state in plot_config_states
    ]
    print(plot_config_ticks)
    plot_config_tick_min_dict = {
        plot_config: min(
            idx for idx, val in enumerate(plot_config_ticks) if val == plot_config
        )
        for plot_config in plot_config_list
    }
    print(plot_config_tick_min_dict)
    plot_config_tick_max_dict = {
        plot_config: max(
            idx for idx, val in enumerate(plot_config_ticks) if val == plot_config
        )
        for plot_config in plot_config_list
    }
    print(plot_config_tick_max_dict)
    plot_config_tick_mid_dict = {
        plot_config: np.mean(
            [
                plot_config_tick_min_dict.get(plot_config),
                plot_config_tick_max_dict.get(plot_config),
            ]
        )
        for plot_config in plot_config_list
    }
    print(plot_config_tick_mid_dict)

    plot_colours = list(islice(cycle(colours), len(plot_config_states)))

    config_state_energies = [
        energies.loc[energies["state"] == state, energy_col]
        for state in plot_config_states
    ]

    plt.figure(num=None, dpi=800, figsize=(8, 7))

    if plot_type is PlotType.EVENT:
        config_state_widths = [
            0.02 if len(energies) > 1000 else 0.05 for energies in config_state_energies
        ]

        plt.eventplot(
            config_state_energies,
            linelengths=0.95,
            linewidths=config_state_widths,
            colors=plot_colours,
            orientation="vertical",
        )
    elif plot_type is PlotType.VIOLIN:
        violin_parts = plt.violinplot(
            config_state_energies,
            positions=range(0, len(config_state_energies)),
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for part_idx, part in enumerate(violin_parts["bodies"]):
            part.set_facecolor(plot_colours[part_idx])

    state_tex_dict = get_state_tex(states=plot_config_states)

    label_font_size = 10
    _, y_max = plt.gca().get_ylim()
    text_offset = y_max / 30
    for state_idx, state in enumerate(plot_config_states):
        text_height = (
            energies.loc[energies["state"] == state, energy_col]
            .sort_values(ascending=True)
            .iloc[0]
            - text_offset
        )
        plt.text(
            state_idx,
            text_height,
            state_tex_dict.get(state),
            horizontalalignment="center",
            fontsize=label_font_size,
        )

    for plot_config in plot_config_list:
        plt.text(
            plot_config_tick_mid_dict.get(plot_config),
            -3000,
            plot_config,
            horizontalalignment="center",
            fontsize=label_font_size,
        )

    plt.tick_params(axis="x", which="both", bottom=True, top=True, labelbottom=False)
    plt.xticks(
        [
            plot_config_tick_max_dict.get(plot_config) + 0.5
            for plot_config in plot_config_list
        ][:-1]
    )
    plt.xlim(left=-0.6, right=len(plot_config_states) - 0.4)
    plt.xlabel("Electronic Configuration", labelpad=25)

    plt.tick_params(axis="y", which="both", labelsize=6)
    plt.ylabel("Energy (cm$^{-1}$)")

    plt.tight_layout()
    if out_file is not None:
        plt.savefig(out_file)
    if show:
        plt.show()
