import re
import typing as t
from itertools import cycle, islice

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_state_tex(
    grouped_states: pd.DataFrame, energy_sort_col: str = "energy_min"
) -> t.Dict:
    """

    Args:
        grouped_states:     pd.DataFrame containing the grouped states to determine markup strings for.
        energy_sort_col:    The column for the energy to sort the states on, generally either 'energy_min' or 't_0'

    Returns:
        A dictionary containing a mapping from the raw state label to its tex markup.
    """
    markup_states = list(
        grouped_states.sort_values(by=[energy_sort_col], ascending=[1])[
            "state"
        ].unique()
    )
    state_tex_dict = {}
    state_regex = re.compile(r"(\w)(['`]?)_?(\d)(\w*)([-\+]?)")
    for state in markup_states:
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
    qn_group_cols: t.List[str],
    colours: t.List[str],
    out_file: str,
    electron_configurations: t.List[str] = None,
    energy_col: str = "energy",
):
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

    config_state_widths = [
        0.02 if len(energies) > 1000 else 0.05 for energies in config_state_energies
    ]

    plt.figure(num=None, dpi=800, figsize=(8, 7))

    plt.eventplot(
        config_state_energies,
        linelengths=0.95,
        linewidths=config_state_widths,
        colors=plot_colours,
        orientation="vertical",
    )

    # TODO: Add support for T_0 values.
    t0_label = "energy_min"
    # t0_label = 't0'
    df_energies_group = energies.groupby(by=qn_group_cols, as_index=False).agg(
        {energy_col: ["min"]}
    )
    energy_min_col = energy_col + "_min"
    df_energies_group.columns = qn_group_cols + [energy_min_col]

    state_tex_dict = get_state_tex(
        grouped_states=df_energies_group, energy_sort_col=energy_min_col
    )

    label_font_size = 10
    _, y_max = plt.gca().get_ylim()
    text_offset = y_max / 30
    for state_idx, state in enumerate(plot_config_states):
        text_height = (
            list(
                df_energies_group.loc[
                    df_energies_group["state"] == state, t0_label
                ].sort_values(ascending=True)
            )[0]
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
    plt.savefig(out_file)
    plt.show()
