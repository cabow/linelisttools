import re
import typing as t
from enum import Enum
from itertools import cycle, islice

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from linelisttools.states import SourceTag

# TODO: Create colourblind friendly colour utility.


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
    electron_configurations: t.Set[str] = None,
    energy_col: str = "energy",
):
    # TODO: Test inbuilt sizes/shifts (i.e.: labelpad) or different scale plots. Allow for plotting without electronic
    #  configuration.
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
                electron_configurations.intersection(plot_electron_configs),
                key=lambda x: electron_configurations.index(x),
            )
        )
    # print("Electron config order: ", plot_config_list)

    plot_config_states = [
        state
        for plot_config in plot_config_list
        for state, config in state_configuration_dict.items()
        if config == plot_config and not state.endswith("_P")
    ]
    # print("Energy ordered states by electron config order", plot_config_states)

    plot_config_ticks = [
        state_configuration_dict.get(state) for state in plot_config_states
    ]
    # print(plot_config_ticks)
    plot_config_tick_min_dict = {
        plot_config: min(
            idx for idx, val in enumerate(plot_config_ticks) if val == plot_config
        )
        for plot_config in plot_config_list
    }
    # print(plot_config_tick_min_dict)
    plot_config_tick_max_dict = {
        plot_config: max(
            idx for idx, val in enumerate(plot_config_ticks) if val == plot_config
        )
        for plot_config in plot_config_list
    }
    # print(plot_config_tick_max_dict)
    plot_config_tick_mid_dict = {
        plot_config: np.mean(
            [
                plot_config_tick_min_dict.get(plot_config),
                plot_config_tick_max_dict.get(plot_config),
            ]
        )
        for plot_config in plot_config_list
    }
    # print(plot_config_tick_mid_dict)

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


def plot_states_by_source_tag(
    states: pd.DataFrame,
    show: bool = True,
    out_file: str = None,
    j_col: str = "J",
    energy_col: str = "energy",
    source_tag_col: str = "source_tag",
    plot_state_list: t.List[str] = None,
    plot_source_list: t.List[t.Union[str, SourceTag]] = None,
):
    if out_file is None and not show:
        raise RuntimeError(
            "No out_file specified and show set to False - nothing to do."
        )

    if plot_state_list is None:
        plot_state_list = states["state"].unique()

    if plot_source_list is None:
        plot_source_list = [st.value for st in SourceTag]

    state_tex_dict = get_state_tex(states=plot_state_list)

    source_tag_legend_dict = {
        SourceTag.CALCULATED.value: "Calculated Energy Level",
        SourceTag.MARVELISED.value: "Marvelised Energy Level",
        SourceTag.EFFECTIVE_HAMILTONIAN.value: "Effective Hamiltonian",
        SourceTag.PARITY_PAIR.value: "Predicted shift (parity pair)",
        SourceTag.PREDICTED_SHIFT.value: "Predicted shift (linear regression)",
        SourceTag.EXTRAPOLATED_SHIFT.value: "Predicted shift (extrapolation)",
        SourceTag.PSEUDO_EXPERIMENTAL.value: "Pseudo-experimental correction",
    }
    plot_colour_list = [
        "#EE7733",
        "#0077BB",
        "#EE3377",
        "#33BBEE",
        "#CC3311",
        "#009988",
        "#BBBBBB",
    ]

    fig, axs = plt.subplots(len(plot_state_list), 1, figsize=(10, 20))
    for state_idx, state in enumerate(plot_state_list):
        state_ax = axs[state_idx]
        data_sets = []
        label_list = []
        for source_idx, source in enumerate(plot_source_list):
            states_source_slice = states.loc[
                (states["state"] == state) & (states[source_tag_col] == source),
                [energy_col, j_col, "parity_norot"],
            ]
            marker_size = 80
            marker_line_width = 0.8
            state_e = state_ax.scatter(
                states_source_slice.loc[
                    states_source_slice["parity_norot"] == "e", j_col
                ],
                states_source_slice.loc[
                    states_source_slice["parity_norot"] == "e", energy_col
                ],
                marker="x",
                s=marker_size,
                linewidth=marker_line_width,
                facecolors=plot_colour_list[source_idx],
                label=f"{source} e",
            )
            state_f = state_ax.scatter(
                states_source_slice.loc[
                    states_source_slice["parity_norot"] == "f", j_col
                ],
                states_source_slice.loc[
                    states_source_slice["parity_norot"] == "f", energy_col
                ],
                marker="+",
                s=marker_size,
                linewidth=marker_line_width,
                facecolors=plot_colour_list[source_idx],
                label=f"{source} f",
            )

            data_sets.append([state_e, state_f])
            label_list.append(["", f"{source_tag_legend_dict.get(source)}"])
        state_tex = state_tex_dict.get(state)
        state_ax.text(
            0.05,
            0.9,
            f"{state_tex}",
            size=20,
            ha="left",
            va="center",
            transform=state_ax.transAxes,
        )
        state_ax.tick_params(axis="x", labelsize=18)
        state_ax.tick_params(axis="y", labelsize=18)
        if state_idx == 1:
            # Set y-axis label for middle plot only
            state_ax.set_ylabel(ylabel="Energy (cm$^{-1}$)", fontsize=18)
        if state_idx == 2:
            # Create legend with both + and x markers for both parity in each source
            state_ax.set_xlabel(xlabel=j_col, fontsize=26)
            data_sets = list(np.array(data_sets).T.flatten())
            label_list = list(np.array(label_list).T.flatten())
            state_ax.legend(
                handles=data_sets,
                labels=label_list,
                loc="lower right",
                prop={"size": 14},
                ncol=2,
                handlelength=0.1,
                columnspacing=0,
            )
    #     Uncomment below if you wish to fiddle with the axes ranges.
    #     # We only need to set the max J we want to plot up to and then auto-scale the y based on this - can need a bit of
    #     fiddling to look good.
    #     x_axis_max_j = plot_coverage_max_j_dict.get(state)
    #     state_ax.set_xlim(left=-1, right=x_axis_max_j)
    #     # Round to the nearest 100 above the maximum energy in the J range.
    #     y_axis_max_energy = int(math.ceil(df_states.loc[(df_states['state'] == state)
    #                                                     & (df_states['source_tag'].isin(plot_source_list))
    #                                                     & (df_states['J'] <= x_axis_max_j), 'energy'].max() / 1000)) * 1000
    #     state_ax.set_ylim(top=y_axis_max_energy)

    plt.tight_layout()
    if out_file is not None:
        plt.savefig(out_file)
    if show:
        plt.show()
