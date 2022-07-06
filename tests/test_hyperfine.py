import os
from io import StringIO
from pathlib import Path

import pandas as pd
import pytest

from pymarvel.hyperfine import deperturb_hyperfine


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


def test_deperturb_hyperfine(df_transitions):
    print(df_transitions)
    print("BEGIN TEST")
    deperturbed_transitions = deperturb_hyperfine(
        df_transitions,
        qn_list=["state", "J", "F"],
        nuclear_spin=2,
    )
    print(deperturbed_transitions)
