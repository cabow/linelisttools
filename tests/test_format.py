import os
from io import StringIO
from pathlib import Path

import pandas as pd
import pytest

from pymarvel.format import output_data


@pytest.fixture(scope="session")
def df_state():
    test_states = StringIO(
        """            1     0.0000000     12  0     0.0000001          inf +   A1     1   o   0   0   0   0   0 Ma
            2  1968.1622648     12  0     0.0050000   1.9317e-02 +   A1     2   o   0   1   0   0   0 Ma
            3  2736.9754969     12  0     0.0000040   1.1633e-02 +   A1     3   o   1   0   0   0   0 Ma
            4  3821.2081670     12  0     0.1000000   1.2227e-02 +   A1     4   o nan nan nan nan nan Ca
            5  4042.7721648     12  0     0.0009000   1.2305e-02 +   A1     5   o   0   0   2   0   0 Ma
            6  4648.7587400     12  0     0.2000000   6.4298e-03 +   A1     6   o nan nan nan nan nan Ca
            7  5385.3522270     12  0     0.2000000   6.0840e-03 +   A1     7   o nan nan nan nan nan Ca
            8  5579.1928220     12  0     0.2000000   8.0766e-03 +   A1     8   o nan nan nan nan nan Ca
            9  6008.5163100     12  0     0.3000000   4.8418e-03 +   A1     9   o nan nan nan nan nan Ca
           10  6432.5596700     12  0     0.3000000   6.6492e-03 +   A1    10   o nan nan nan nan nan Ca
        """
    )
    return pd.read_csv(test_states, delim_whitespace=True, header=None)


@pytest.fixture(scope="session")
def temp_state_file():
    return (Path(__file__).parent / r"./inputs/state_format_test.txt").resolve()


@pytest.mark.parametrize(
    "fortran_format_list",
    [
        [
            "i10",
            "f14.7",
            "i4",
            "i2",
            "f14.8",
            "e12.4",
            "a1",
            "a4",
            "i5",
            "a3",
            "i3",
            "i3",
            "i3",
            "i3",
            "i3",
            "a6",
        ]
    ],
)
def test_output_data(df_state, temp_state_file, fortran_format_list):
    output_data(df_state, temp_state_file, fortran_format_list)
    # TODO: Figure out useful asserts, cleanup generated file.
