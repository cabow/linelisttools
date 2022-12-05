import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def trans_file():
    return (Path(__file__).parent / r"./inputs/trans19May22.txt").resolve()


@pytest.fixture(scope="session")
def segment_file():
    return (Path(__file__).parent / r"./inputs/segment_N2O_0414.txt").resolve()


@pytest.fixture(scope="session")
def check_trans_file():
    return (Path(__file__).parent / r"./inputs/CheckTransitions.txt").resolve()


@pytest.mark.parametrize("test_nqn", [5])
def test_run_marvel(trans_file, test_nqn, segment_file):
    from linelisttools.marvel import run_marvel

    run_marvel(marvel_trans_file=trans_file, nqn=test_nqn, segment_file=segment_file)
    clean_outputs(trans_file)


@pytest.mark.parametrize("test_qn_list", [["v1", "v2", "v3", "parity", "J"]])
def test_generate_marvel_energies(trans_file, test_qn_list, segment_file):
    from linelisttools.marvel import generate_marvel_energies

    df_energies = generate_marvel_energies(
        transitions_file=trans_file, qn_list=test_qn_list, segment_file=segment_file
    )
    for qn in test_qn_list:
        assert qn in df_energies.columns

    clean_outputs(trans_file)


@pytest.mark.parametrize("test_nqn", [5])
@pytest.mark.parametrize("bad_segmentfile", [r"C:\BAD.txt"])
def test_run_marvel_wrong_nqn(trans_file, test_nqn, bad_segmentfile):
    from linelisttools.marvel import run_marvel

    with pytest.raises(RuntimeError):
        run_marvel(
            marvel_trans_file=trans_file, nqn=test_nqn, segment_file=bad_segmentfile
        )

    clean_outputs(trans_file)


def clean_outputs(trans_file):
    clean_files = [
        "EnergyLevels.txt",
        "reviveTRs.txt",
        "CheckTransitions.txt",
        "Components.txt",
        "BadLines.txt",
        "AVTable.txt",
    ]
    for clean_file in clean_files:
        clean_path = (Path(trans_file).parent / clean_file).resolve()
        try:
            os.remove(clean_path)
        except OSError:
            pass


@pytest.mark.parametrize("qn_list", [["state", "fs", "omega", "parity_tot", "v", "J"]])
def test_check_trans_regex(check_trans_file, qn_list):
    from linelisttools.marvel import parse_check_trans

    parse_check_trans(check_trans_file=check_trans_file, qn_list=qn_list)


@pytest.mark.parametrize(
    "marvel_path, marvel_trans_file, segment_file, qn_list, marvel_trans_fortran_format_list",
    [
        (
            r"C:\PhD\MARVEL\MARVEL4.1\MARVEL4.1.exe",
            (Path(__file__).parent / r"./inputs/VO_transitions-2.0.txt").resolve(),
            (Path(__file__).parent / r"./inputs/VO_segment-2.0.txt").resolve(),
            ["state", "fs", "omega", "parity_tot", "v", "J"],
            [
                "f14.8",
                "e14.8",
                "e14.8",
                "a12",
                "i1",
                "f4.1",
                "a1",
                "i2",
                "f6.1",
                "i3",
                "a12",
                "i1",
                "f4.1",
                "a1",
                "i2",
                "f6.1",
                "i3",
                "a17",
            ],
        )
    ],
)
def test_optimise_bad_transitions(
    marvel_path,
    marvel_trans_file,
    segment_file,
    qn_list,
    marvel_trans_fortran_format_list,
):
    from linelisttools.marvel import optimise_bad_transitions

    optimise_bad_transitions(
        marvel_path=marvel_path,
        marvel_trans_file=marvel_trans_file,
        segment_file=segment_file,
        qn_list=qn_list,
        marvel_trans_fortran_format_list=marvel_trans_fortran_format_list,
    )
