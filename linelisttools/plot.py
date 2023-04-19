import re
import typing as t
from enum import IntEnum
from itertools import cycle, islice

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .format import SourceTag


class PlotType(IntEnum):
    VIOLIN = 0
    EVENT = 1


def get_vibrant_colors(n_colors: int) -> t.List[str]:
    vibrant_color_list = [
        "#0077BB",
        "#33BBEE",
        "#009988",
        "#44EE66",
        "#FFCC11",
        "#EE7733",
        "#CC3311",
        "#EE3377",
        "#BB33BB",
        "#8833EE",
    ]
    if n_colors > len(vibrant_color_list):
        return list(islice(cycle(vibrant_color_list), n_colors))
    else:
        colors_list_idx = np.linspace(
            0, len(vibrant_color_list) - 1, n_colors, dtype=int
        ).tolist()
        return list([vibrant_color_list[idx] for idx in colors_list_idx])


def get_qualitative_colors(n_colors: int) -> t.List[str]:
    qualitative_color_dict = {
        1: ["#4477aa"],
        2: ["#4477aa", "#cc6677"],
        3: ["#4477aa", "#ddcc77", "#cc6677"],
        4: ["#4477aa", "#117733", "#ddcc77", "#cc6677"],
        5: ["#332288", "#88ccee", "#117733", "#ddcc77", "#cc6677"],
        6: ["#332288", "#88ccee", "#117733", "#ddcc77", "#cc6677", "#aa4499"],
        7: [
            "#332288",
            "#88ccee",
            "#44aa99",
            "#117733",
            "#ddcc77",
            "#cc6677",
            "#aa4499",
        ],
        8: [
            "#332288",
            "#88ccee",
            "#44aa99",
            "#117733",
            "#999933",
            "#ddcc77",
            "#cc6677",
            "#aa4499",
        ],
        9: [
            "#332288",
            "#88ccee",
            "#44aa99",
            "#117733",
            "#999933",
            "#ddcc77",
            "#cc6677",
            "#882255",
            "#aa4499",
        ],
        10: [
            "#332288",
            "#88ccee",
            "#44aa99",
            "#117733",
            "#999933",
            "#ddcc77",
            "#661100",
            "#cc6677",
            "#882255",
            "#aa4499",
        ],
        11: [
            "#332288",
            "#6699cc",
            "#88ccee",
            "#44aa99",
            "#117733",
            "#999933",
            "#ddcc77",
            "#661100",
            "#cc6677",
            "#882255",
            "#aa4499",
        ],
        12: [
            "#332288",
            "#6699cc",
            "#88ccee",
            "#44aa99",
            "#117733",
            "#999933",
            "#ddcc77",
            "#661100",
            "#cc6677",
            "#aa4466",
            "#882255",
            "#aa4499",
        ],
    }
    if n_colors > 12:
        return list(islice(cycle(qualitative_color_dict.get(12)), n_colors))
    else:
        return qualitative_color_dict.get(n_colors)


def get_state_tex(states: t.List[str]) -> t.Dict:
    """
    Maps the input list of string state labels to a TeX markup version, assuming the state label consist of (in order):
        - A single letter (upper- or lower-case) or number.
        - Any sequential apostrophes immediately after.
        - Potentially a single underscore to separate the state label or symbol.
        - A single digit (denoting the spin multiplicity).
        - At least one word character representing the type of state (i.e.: "Sigma").
        - A potential single instance of a minus or plus character.

    Do we need support for garade/ungarade notation?

    Args:
        states: List of raw state labels to be converted to tex.
    Returns:
        A dictionary containing a mapping from the raw state label to its tex markup.
    """
    states = set(states)
    state_tex_dict = {}
    state_regex = re.compile(r"([^\W_])([p'`]*?)_?(\d)(\w+)([-\+]?)")
    electronic_shorthand_dict = {"Sig": "Sigma", "Del": "Delta", "Gam": "Gamma"}
    for state in states:
        state_match = state_regex.match(state)
        prime = "\\prime" if state_match.group(2) != "" else ""
        symmetry = state_match.group(5)
        electronic_state = state_match.group(4).capitalize()
        if electronic_state in electronic_shorthand_dict.keys():
            electronic_state = electronic_shorthand_dict.get(electronic_state)

        symmetry_str = "^" + symmetry if symmetry != "" else ""
        state_tex = f"{state_match.group(1)}$^{{{prime}{state_match.group(3)}}}\\{electronic_state}{symmetry_str}$"
        state_tex_dict[state] = state_tex
    return state_tex_dict


def plot_state_coverage(
    energies: pd.DataFrame,
    state_list: t.List[str] = None,
    state_configuration_dict: t.Dict = None,
    show: bool = True,
    out_file: str = None,
    plot_type: PlotType = PlotType.EVENT,
    electron_configurations: t.List[str] = None,
    energy_col: str = "energy",
) -> None:
    """
    Plots the level coverage of the input marvel energies by electronic state as a function of their energies. Orders
    the states by electron configuration when a mapping is provided, otherwise the order in which they were provided is
    used.

    A valid out_file path must be provided or show set to true.

    Args:
        energies:                 A DataFrame containing the Marvel energies for a molecule.
        state_list:               A list of the states in the energies DataFrame to plot.
        state_configuration_dict: A dictionary mapping each state to be plotted to its electron configuration.
        show:                     A boolean to determine whether the generated figure is shown.
        out_file:                 The filepath to save the figure to.
        plot_type:                An enum to determine whether the state coverage is plotted as an event or violin plot.
        electron_configurations:  A list of the possible electron configurations to plot.
        energy_col:               The string column name for the energy column in the energies DataFrame.
    """
    # TODO: Clarify that electron_configurations is intended to provide the left-to-right plotting order for configs.
    if state_list is None and state_configuration_dict is None:
        raise RuntimeError(
            "No values passed for state_list or state_configuration_dict. Either specify which states to plot by "
            "passing a string list to state_list or a dictionary mapping them to their electron configuration to "
            "state_configuration_dict."
        )

    if out_file is None and not show:
        raise RuntimeError(
            "No out_file specified and show set to False - nothing to do."
        )

    if plot_type not in (PlotType.EVENT, PlotType.VIOLIN):
        raise RuntimeError(
            f"State coverage plot only accepts PlotTypes of {PlotType.EVENT} (default) or {PlotType.VIOLIN}."
        )

    plt.figure(num=None, dpi=800, figsize=(8, 7))
    label_font_size = 10

    if state_list is None:
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

        state_list = [
            state
            for plot_config in plot_config_list
            for state, config in state_configuration_dict.items()
            if config == plot_config and not state.endswith("_P")
        ]

        plot_config_ticks = [
            state_configuration_dict.get(state) for state in state_list
        ]
        plot_config_tick_min_dict = {
            plot_config: min(
                idx for idx, val in enumerate(plot_config_ticks) if val == plot_config
            )
            for plot_config in plot_config_list
        }
        plot_config_tick_max_dict = {
            plot_config: max(
                idx for idx, val in enumerate(plot_config_ticks) if val == plot_config
            )
            for plot_config in plot_config_list
        }
        plot_config_tick_mid_dict = {
            plot_config: np.mean(
                [
                    plot_config_tick_min_dict.get(plot_config),
                    plot_config_tick_max_dict.get(plot_config),
                ]
            )
            for plot_config in plot_config_list
        }

        for plot_config in plot_config_list:
            plt.text(
                plot_config_tick_mid_dict.get(plot_config),
                -3000,
                plot_config,
                horizontalalignment="center",
                fontsize=label_font_size,
            )

        plt.tick_params(
            axis="x", which="both", bottom=True, top=True, labelbottom=False
        )
        plt.xticks(
            [
                plot_config_tick_max_dict.get(plot_config) + 0.5
                for plot_config in plot_config_list
            ][:-1]
        )
    else:
        plt.tick_params(
            axis="x", which="both", bottom=False, top=False, labelbottom=False
        )

    # TODO: Allow for switch between vibrant and generic qualitative progression?
    # plot_colour_list = get_qualitative_colors(len(plot_config_states))
    plot_colour_list = get_vibrant_colors(len(state_list))

    state_energies = [
        energies.loc[energies["state"] == state, energy_col] for state in state_list
    ]

    if plot_type is PlotType.EVENT:
        config_state_widths = [
            0.02 if len(energies) > 1000 else 0.05 for energies in state_energies
        ]

        plt.eventplot(
            state_energies,
            linelengths=0.95,
            linewidths=config_state_widths,
            colors=plot_colour_list,
            orientation="vertical",
        )
    elif plot_type is PlotType.VIOLIN:
        violin_plot = plt.violinplot(
            state_energies,
            positions=range(0, len(state_energies)),
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for body_idx, body in enumerate(violin_plot["bodies"]):
            body.set_facecolor(plot_colour_list[body_idx])
            body.set_alpha(1.0)

    state_tex_dict = get_state_tex(states=state_list)

    _, y_max = plt.gca().get_ylim()
    text_offset = y_max / 30
    for state_idx, state in enumerate(state_list):
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

    plt.xlim(left=-0.6, right=len(state_list) - 0.4)
    plt.xlabel(
        "Electronic Configuration",
        labelpad=6 if state_configuration_dict is None else 25,
    )

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
    energy_cutoff: float = None,
    j_cutoff: float = None,
    j_col: str = "J",
    energy_col: str = "energy",
    source_tag_col: str = "source_tag",
    plot_state_list: t.List[str] = None,
    plot_source_list: t.List[t.Union[str, SourceTag]] = None,
) -> None:
    """
    Plots the contents of the states file, colour coded by their source_tag. Can be restricted to a subset of the
    possible source_tag values and to a subset of the present electronic states.

    Assumes the states file contains a total parity column.

    Args:
        states:
        show:
        out_file:
        energy_cutoff:
        j_col:
        energy_col:
        source_tag_col:
        plot_state_list:
        plot_source_list:

    Returns:

    """
    if out_file is None and not show:
        raise RuntimeError(
            "No out_file specified and show set to False - nothing to do."
        )

    if plot_state_list is None:
        plot_state_list = states["state"].unique()
    elif (
        len(missing_states := np.setdiff1d(plot_state_list, states["state"].unique()))
        != 0
    ):
        raise RuntimeWarning(
            f"Some states specified in plot_state_list do not exist in the states file: {missing_states}"
        )

    if plot_source_list is None:
        plot_source_list = [st.value for st in SourceTag]
    else:
        plot_source_list = [
            st.value if isinstance(st, SourceTag) else st for st in plot_source_list
        ]

    if energy_cutoff is None:
        plot_states = states.copy()
    else:
        plot_states = states.loc[states["energy"] <= energy_cutoff]

    if j_cutoff is not None:
        plot_states = plot_states.loc[plot_states["J"] <= j_cutoff]

    if len(plot_states) == 0:
        raise RuntimeError("No states to plot after applying energy and j cutoffs.")

    for plot_state in plot_state_list:
        if (
            len(
                plot_states.loc[
                    (plot_states["state"] == plot_state)
                    & (
                        plot_states[source_tag_col]
                        .map(lambda x: x.value)
                        .isin(plot_source_list)
                    )
                ]
            )
            == 0
        ):
            raise RuntimeError(
                f"The state {plot_state} has no levels with matching source tags {plot_source_list} to plot"
                f"{' below the energy/J cutoffs' if energy_cutoff is not None or j_cutoff is not None else ''}."
            )

    state_tex_dict = get_state_tex(states=plot_state_list)

    source_tag_legend_dict = {
        SourceTag.CALCULATED.value: "Calculated Energy Level",
        SourceTag.MARVELISED.value: "Marvelised Energy Level",
        SourceTag.EFFECTIVE_HAMILTONIAN.value: "Effective Hamiltonian",
        SourceTag.PS_PARITY_PAIR.value: "Predicted shift (parity pair)",
        SourceTag.PS_LINEAR_REGRESSION.value: "Predicted shift (linear regression)",
        SourceTag.PS_EXTRAPOLATION.value: "Predicted shift (extrapolation)",
        SourceTag.PSEUDO_EXPERIMENTAL.value: "Pseudo-experimental correction",
    }
    source_tag_color_dict = {
        SourceTag.CALCULATED.value: "#EE7733",
        SourceTag.MARVELISED.value: "#EE3377",
        SourceTag.EFFECTIVE_HAMILTONIAN.value: "#009988",
        SourceTag.PS_PARITY_PAIR.value: "#33BBEE",
        SourceTag.PS_LINEAR_REGRESSION.value: "#FFCC11",  # EECC33 is maybe more in keeping with saturation/brightness.
        SourceTag.PS_EXTRAPOLATION.value: "#0077BB",
        SourceTag.PSEUDO_EXPERIMENTAL.value: "#CC3311",
    }
    # plot_colour_list = get_qualitative_colors(len(plot_source_list))

    fig, axs = plt.subplots(len(plot_state_list), 1, figsize=(10, 20))
    marker_size = 80
    marker_line_width = 0.8
    for state_idx, state in enumerate(plot_state_list):
        if len(plot_state_list) == 1:
            state_ax = axs
        else:
            state_ax = axs[state_idx]
        data_sets = []
        label_list = []
        for source_idx, source in enumerate(plot_source_list):
            states_source_slice = plot_states.loc[
                (plot_states["state"] == state)
                & (plot_states[source_tag_col].map(lambda x: x.value) == source),
                [energy_col, j_col, "parity_tot"],
            ]
            state_e = state_ax.scatter(
                states_source_slice.loc[
                    states_source_slice["parity_tot"] == "-", j_col
                ],
                states_source_slice.loc[
                    states_source_slice["parity_tot"] == "-", energy_col
                ],
                marker="x",
                s=marker_size,
                linewidth=marker_line_width,
                # facecolors=plot_colour_list[source_idx],
                facecolors=source_tag_color_dict.get(source),
                label=f"{source} e",
            )
            state_f = state_ax.scatter(
                states_source_slice.loc[
                    states_source_slice["parity_tot"] == "+", j_col
                ],
                states_source_slice.loc[
                    states_source_slice["parity_tot"] == "+", energy_col
                ],
                marker="+",
                s=marker_size,
                linewidth=marker_line_width,
                # facecolors=plot_colour_list[source_idx],
                facecolors=source_tag_color_dict.get(source),
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
        state_ax.set_ylabel(ylabel="Energy (cm$^{-1}$)", fontsize=18)
        if state_idx == len(plot_state_list) - 1:
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
