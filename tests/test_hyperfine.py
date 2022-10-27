from io import StringIO

import pandas as pd
import pytest

from linelisttools.hyperfine import (
    calc_hf_skew,
    calc_num_possible_hf_trans,
    calc_possible_hf_trans,
    deperturb_hyperfine,
    perturb_hyperfine,
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


def test_perturb_hyperfine(
    states_hfr, states_hfu, qn_list, splitting_dependent_qn_list, nuclear_spin, f_col
):
    # Match the hyperfine MARVEL states with the Duo states, then use the shifts to perturb the nhf MARVEL states.
    perturb_hyperfine(
        states_hfr=states_hfr,
        states_hfu=states_hfu,
        qn_list=qn_list,
        splitting_dependent_qn_list=splitting_dependent_qn_list,
        nuclear_spin=nuclear_spin,
        f_col=f_col,
    )
