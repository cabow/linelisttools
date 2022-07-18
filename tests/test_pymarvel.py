import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def transfile():
    return (Path(__file__).parent / r"./inputs/trans19May22.txt").resolve()


@pytest.fixture(scope="session")
def segmentfile():
    return (Path(__file__).parent / r"./inputs/segment_N2O_0414.txt").resolve()


@pytest.mark.parametrize("test_nqn", [5])
def test_run_marvel(transfile, test_nqn, segmentfile):
    from linelisttools.pymarvel import run_marvel

    run_marvel(transitions_file=transfile, nqn=test_nqn, segment_file=segmentfile)
    clean_outputs(transfile)


@pytest.mark.parametrize("test_qn_list", [["v1", "v2", "v3", "parity", "J"]])
def test_generate_marvel_energies(transfile, test_qn_list, segmentfile):
    from linelisttools.pymarvel import generate_marvel_energies

    df_energies = generate_marvel_energies(
        transitions_file=transfile, qn_list=test_qn_list, segment_file=segmentfile
    )
    for qn in test_qn_list:
        assert qn in df_energies.columns

    clean_outputs(transfile)


@pytest.mark.parametrize("test_nqn", [5])
@pytest.mark.parametrize("bad_segmentfile", [r"C:\BAD.txt"])
def test_run_marvel_wrong_nqn(transfile, test_nqn, bad_segmentfile):
    from linelisttools.pymarvel import run_marvel

    with pytest.raises(RuntimeError):
        run_marvel(transitions_file=transfile, nqn=test_nqn, segment_file=segmentfile)

    clean_outputs(transfile)


def clean_outputs(transfile):
    clean_files = ["EnergyLevels.txt", "reviveTRs.txt", "CheckTransitions.txt"]
    for clean_file in clean_files:
        clean_path = (Path(transfile).parent / clean_file).resolve()
        try:
            os.remove(clean_path)
        except OSError:
            pass
