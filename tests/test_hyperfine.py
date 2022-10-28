from io import StringIO
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import pytest

from linelisttools.format import SourceTag
from linelisttools.hyperfine import (
    calc_hf_skew,
    calc_num_possible_hf_trans,
    calc_possible_hf_trans,
    deperturb_hyperfine,
    perturb_hyperfine,
)
from linelisttools.states import (
    ExoMolStatesHeader,
    match_states,
    parity_norot_to_total,
    read_exomol_states,
    read_mvl_energies,
)


@pytest.fixture(scope="session")
def df_transitions():
    test_transitions = StringIO(
        """ 100.0000000  0.005   a  5     5   b   4     4   test.1  0.45
            100.1622648  0.005   a  5     6   b   4     5   test.2  0.65
            143.4885845  0.040   a  7     7   b   6     6   test.3  0.51
        """
    )
    return pd.read_csv(
        test_transitions,
        delim_whitespace=True,
        header=None,
        names=[
            "energy",
            "unc",
            "state_u",
            "J_u",
            "F_u",
            "state_l",
            "J_l",
            "F_l",
            "source",
            "intensity",
        ],
    )


@pytest.mark.parametrize(
    "test_nuclear_spin,test_j_u,test_j_l,test_delta_f_list",
    [(3.5, 5.5, 4.5, [-1, 0, 1])],
)
def test_calculate_possible_hf_trans(
    test_nuclear_spin, test_j_u, test_j_l, test_delta_f_list
):
    possible_hf_trans = calc_possible_hf_trans(
        test_nuclear_spin, test_j_u, test_j_l, test_delta_f_list
    )
    print(possible_hf_trans)
    assert len(possible_hf_trans) == 21


@pytest.mark.parametrize(
    "test_nuclear_spin,test_j_u,test_j_l,test_f_u_list,test_f_l_list",
    [(3.5, 5.5, 4.5, [4, 3, 5, 6], [4, 4, 4, 5])],
)
def test_new_calculate_num_possible_hf_transitions(
    test_nuclear_spin, test_j_u, test_j_l, test_f_u_list, test_f_l_list
):
    new_num_poss_hf_trans = calc_num_possible_hf_trans(
        test_nuclear_spin, test_j_u, test_j_l, test_f_u_list, test_f_l_list
    )
    assert new_num_poss_hf_trans == 21


@pytest.mark.parametrize(
    "test_nuclear_spin,test_j_u,test_j_l,test_f_u_list,test_f_l_list",
    [(3.5, 5.5, 4.5, [4, 3, 5, 6], [4, 4, 4, 5])],
)
def test_calc_possible_hf_skew(
    test_nuclear_spin, test_j_u, test_j_l, test_f_u_list, test_f_l_list
):
    hf_skew = calc_hf_skew(
        test_nuclear_spin,
        test_j_u,
        test_j_l,
        test_f_u_list,
        test_f_l_list,
        hf_skew_scale_factor=4,
    )
    assert hf_skew == 3.5


def test_deperturb_hyperfine(df_transitions):
    deperturbed_transitions = deperturb_hyperfine(
        df_transitions,
        qn_list=["state", "J", "F"],
        nuclear_spin=2,
    )
    assert len(deperturbed_transitions) == 2
    assert round(deperturbed_transitions["energy_wm"].loc[0], 6) == 100.095884
    assert deperturbed_transitions["present_hf_trans"].loc[0] == 2
    assert deperturbed_transitions["possible_hf_trans"].loc[1] == 5
    assert deperturbed_transitions["hfs_presence"].loc[1] == 5
    assert deperturbed_transitions["hfs_skew"].loc[0] == 3


def test_deperturb_hyperfine_column_errors(df_transitions):
    with pytest.raises(RuntimeError):
        deperturb_hyperfine(
            df_transitions,
            qn_list=["state", "J", "F"],
            nuclear_spin=2,
            intensity_col="wrong",
        )
    with pytest.raises(RuntimeError):
        deperturb_hyperfine(
            df_transitions,
            qn_list=["state", "J", "F", "WRONG"],
            nuclear_spin=2,
        )


@pytest.fixture(scope="session")
def vo_4hfs_marvel_energies_file():
    return (
        Path(__file__).parent / r"./inputs/VO_4hfs_MARVEL_Energies-2.0.txt"
    ).resolve()


@pytest.fixture(scope="session")
def vo_4hfs_duo_states_file():
    return (Path(__file__).parent / r"./inputs/VO_hfs.states").resolve()


@pytest.mark.parametrize(
    "qn_list,splitting_dependent_qn_list,nuclear_spin,f_col",
    [
        (
            ["state", "omega", "parity_tot", "v", "J", "F"],
            ["state", "omega", "parity_tot", "J", "F"],
            3.5,
            "F",
        )
    ],
)
def test_perturb_hyperfine(
    vo_4hfs_marvel_energies_file,
    vo_4hfs_duo_states_file,
    # states_hfu,
    qn_list,
    splitting_dependent_qn_list,
    nuclear_spin,
    f_col,
):
    states_hfr_mvl = read_mvl_energies(
        file=vo_4hfs_marvel_energies_file,
        qn_cols=["state", "fs", "omega", "parity_norot", "v", "J", "F"],
        energy_cols=["energy", "unc", "degree"],
    )
    states_hfr_mvl["parity_tot"] = states_hfr_mvl.apply(
        lambda x: parity_norot_to_total(parity_norot=x["parity_norot"], j_val=x["J"]),
        axis=1,
    )
    print(states_hfr_mvl)
    states_hfr_duo = pd.read_csv(
        vo_4hfs_duo_states_file,
        delim_whitespace=True,
        names=[
            "id",
            "energy",
            "g",
            "F",
            "I",
            "parity_tot",
            "J",
            "state",
            "v",
            "lambda",
            "sigma",
            "omega",
        ],
    )
    states_hfr_duo["energy"] = states_hfr_duo["energy"] - states_hfr_duo["energy"].min()
    states_hfr_duo["state"] = states_hfr_duo["state"].map(
        lambda x: str.replace(x, ",", "_")
    )
    states_hfr_duo["omega"] = states_hfr_duo.apply(
        lambda x: abs(x["omega"]) if x["state"] == "X_4Sigma-" else x["omega"], axis=1
    )
    states_hfr_duo = states_hfr_duo[
        [
            "id",
            "energy",
            "g",
            "F",
            "parity_tot",
            "state",
            "v",
            "J",
            "lambda",
            "sigma",
            "omega",
        ]
    ]
    statesheader_hfr_duo = ExoMolStatesHeader(
        degeneracy="g",
        rigorous_qn="F",
        unc=None,
        parity=ExoMolStatesHeader.StatesParity.TOTAL_PARITY,
        symmetry="state",
        vibrational_qn="v",
        other_qn=["J", "lambda", "sigma", "omega"],
    )
    states_hfr_match = match_states(
        states_calc=states_hfr_duo,
        states_obs=states_hfr_mvl,
        qn_match_cols=qn_list,
        match_source_tag=SourceTag.MARVELISED,
        states_header=statesheader_hfr_duo,
    )
    states_hfr_match_mvl = states_hfr_match.loc[
        states_hfr_match["source_tag"] == SourceTag.MARVELISED.value
    ]
    print(states_hfr_match_mvl)
    pio.renderers.default = "browser"
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=states_hfr_match_mvl["J"],
            y=states_hfr_match_mvl["energy_obs"],
            customdata=states_hfr_match_mvl["state"],
            marker=dict(
                color="#33BBEE",
                line=dict(color="#33BBEE", width=3),
                symbol="cross-thin",
                size=20,
            ),
            mode="markers",
            hovertemplate="<i>J:</i>%{x}<br><i>Energy (obs):</i>%{y}<br><i>State:</i>%{customdata}",
            showlegend=True,
            name="MARVEL",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=states_hfr_match_mvl["J"],
            y=states_hfr_match_mvl["energy_calc"],
            customdata=states_hfr_match_mvl["state"],
            marker=dict(
                color="#CC3311",
                line=dict(color="#CC3311", width=3),
                symbol="x-thin",
                size=20,
            ),
            mode="markers",
            hovertemplate="<i>J:</i>%{x}<br><i>Energy (calc):</i>%{y}<br><i>State:</i>%{customdata}",
            showlegend=True,
            name="Duo",
        )
    )
    fig.update_layout(
        xaxis=go.layout.XAxis(title="J"),
        yaxis=go.layout.YAxis(title="Energy (cm<sup>-1</sup>)", showticklabels=False),
    )
    fig.show()

    # Match the hyperfine MARVEL states with the Duo states, then use the shifts to perturb the nhf MARVEL states.
    # TODO: What hfu states to use?
    # perturb_hyperfine(
    #     states_hfr=states_hfr_match,
    #     states_hfu=states_hfu,
    #     qn_list=qn_list,
    #     splitting_dependent_qn_list=splitting_dependent_qn_list,
    #     nuclear_spin=nuclear_spin,
    #     f_col=f_col,
    # )
