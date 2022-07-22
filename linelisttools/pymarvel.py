import subprocess
from pathlib import Path

import pandas as pd

# TODO: This needs to be changed to run on Linux - could be passed in as argument to allow for local Linux compilations.
#  Would Cython make this easier? Probably. Add argument as stop gap before cython is implemented.
MARVEL_PATH = (Path(__file__).parent / r"../resources/MARVEL3.1.exe").resolve()


def generate_marvel_energies(
    transitions_file: str,
    qn_list: list[str],
    segment_file: str = None,
    old_format: bool = False,
):
    run_marvel(
        transitions_file=transitions_file,
        nqn=len(qn_list),
        segment_file=segment_file,
        old_format=old_format,
    )
    df_energies = load_energies(
        Path(transitions_file).parent.absolute(), qn_list=qn_list
    )
    return df_energies


def run_marvel(
    transitions_file: str,
    nqn: int,
    segment_file: str = None,
    old_format: bool = False,
    min_size: int = None,
) -> None:
    transitions_path = Path(transitions_file)
    transitions_folder = transitions_path.parent.absolute()
    run_command = f'{MARVEL_PATH} -t {transitions_file} {"-s " + str(segment_file) if segment_file else ""} -n {nqn}{" --old" if old_format else ""}{" --minsize " + str(min_size) if min_size is not None and min_size >= 0 else ""}'
    marvel_process = subprocess.Popen(
        run_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=transitions_folder,
    )

    communicate_res = marvel_process.communicate()
    # std_out_val, std_err_val = communicate_res
    # print(communicate_res)
    # print(marvel_process.returncode)
    return_code = marvel_process.returncode
    if return_code is not None and return_code != 0:
        raise RuntimeError(
            f"Marvel failed to execute correctly: Return code = {return_code}"
        )


def load_energies(source_folder: Path, qn_list: list[str]) -> pd.DataFrame:
    mvl_energies_cols = qn_list + ["energy", "unc", "stddev", "degree"]
    return pd.read_csv(
        (source_folder / "EnergyLevels.txt").resolve(),
        sep=r"\s+",
        names=mvl_energies_cols,
    )
