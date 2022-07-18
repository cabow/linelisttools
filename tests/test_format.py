import os
from io import StringIO
from pathlib import Path

import pandas as pd
import pytest

from linelisttools.format import fortran_format, output_data


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


@pytest.mark.parametrize(
    "val_fmt_list",
    [
        [
            ["12", "i12.3"],
            ["12", "i11"],
            ["12.8483", "f10.2"],
            ["12.34534534", "f10.9"],
            ["beans", "a"],
            ["beans", "a19"],
            ["1.389478939845", "e10.3"],
            ["5000000.7771389478939845", "g20.5"],
            ["5000000.7771389478939845", "G20.5"],
            ["50000.77", "G10.8"],
            ["502345700.7771389478939845", "e12.5"],
            ["23485723.4562735234", "E12.7"],
            ["inf", "e12.5"],
            ["nan", "e12.5"],
        ]
    ],
)
def test_fortran_format(val_fmt_list):
    for item in val_fmt_list:
        item.extend([fortran_format(val=item[0], fmt=item[1])])
    assert val_fmt_list[0][2] == "         012"
    assert val_fmt_list[1][2] == "         12"
    assert val_fmt_list[2][2] == "     12.85"
    assert val_fmt_list[3][2] == "12.345345340"
    assert val_fmt_list[4][2] == "beans"
    assert val_fmt_list[5][2] == "              beans"
    assert val_fmt_list[6][2] == " 1.389e+00"
    assert val_fmt_list[7][2] == "          5.0000e+06"
    assert val_fmt_list[8][2] == "          5.0000E+06"
    assert val_fmt_list[9][2] == " 50000.770"
    assert val_fmt_list[10][2] == " 5.02346e+08"
    assert val_fmt_list[11][2] == "2.3485723E+07"
    assert val_fmt_list[12][2] == "         inf"
    assert val_fmt_list[13][2] == "         nan"
