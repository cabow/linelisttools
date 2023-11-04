import math
import os
import re
import subprocess
import typing as t
from os import getcwd
from pathlib import Path
from re import Pattern

import numpy as np
import pandas as pd

from .format import output_data


def run_marvel(
    marvel_path: t.Union[str, Path],
    marvel_trans_file: t.Union[str, Path],
    nqn: int,
    segment_file: str = None,
    min_size: int = None,
    bootstrap_iterations: int = None,
) -> None:
    transitions_path = Path(marvel_trans_file)
    transitions_folder = transitions_path.parent.absolute()
    run_command = f"{marvel_path} -t {marvel_trans_file} {'-s ' + str(segment_file) if segment_file else ''} -n {nqn}{' --minsize ' + str(min_size) if min_size is not None and min_size >= 0 else ''}{' --bootiter ' + str(bootstrap_iterations) if bootstrap_iterations is not None and bootstrap_iterations >= 0 else ''}"
    marvel_process = subprocess.Popen(
        run_command,
        # stdout=subprocess.PIPE,
        # stderr=subprocess.PIPE,
        # cwd=transitions_folder,
        env=dict(os.environ),
        shell=True,
    )

    marvel_process.communicate()
    # std_out_val, std_err_val = communicate_res
    # print(communicate_res)
    # print(marvel_process.returncode)
    return_code = marvel_process.returncode
    if return_code is not None and return_code != 0:
        raise RuntimeError(
            f"Marvel failed to execute correctly: Return code = {return_code}"
        )


def read_marvel_energies(
    marvel_energy_file: t.Union[str, Path], qn_list: t.List[str]
) -> pd.DataFrame:
    mvl_energy_cols = qn_list + ["energy", "unc", "degree"]
    return pd.read_csv(marvel_energy_file, delim_whitespace=True, names=mvl_energy_cols)


def generate_marvel_energies(
    marvel_path: t.Union[str, Path],
    marvel_trans_file: t.Union[str, Path],
    qn_list: list[str],
    segment_file: str = None,
    min_size: int = None,
    bootstrap_iterations: int = None,
) -> pd.DataFrame:
    run_marvel(
        marvel_path=marvel_path,
        marvel_trans_file=marvel_trans_file,
        nqn=len(qn_list),
        segment_file=segment_file,
        min_size=min_size,
        bootstrap_iterations=bootstrap_iterations,
    )
    print("TEST", Path(getcwd()))
    print("TEST", (Path(getcwd()) / f"./EnergyLevels.txt").resolve())
    marvel_energies = read_marvel_energies(
        marvel_energy_file=(Path(getcwd()) / f"./EnergyLevels.txt").resolve(),
        qn_list=qn_list,
    )
    return marvel_energies


def get_check_trans_regex(nqn: int, only_bad: bool) -> Pattern:
    check_trans_regex = (
        r"^([\d\w_\-]+\.\d+)[^\S\r\n]+(\d+\.\d+)[^\S\r\n]+(-?\d+\.\d+)[^\S\r\n]+(\d\.\d+[eE][\-\+]\d+)[^\S\r\n]+"
        r"(-?\d\.\d+[eE][\-\+]\d+)[^\S\r\n]+"
    )
    check_trans_regex += r"([^\s]+)[^\S\r\n]+" * nqn
    check_trans_regex += r"(\d+\.\d+)[^\S\r\n]+(\d\.\d+[eE][\-\+]\d+)"
    if only_bad:
        check_trans_regex += r"[^\S\r\n]+(WRONG|VERY BAD|VERY BAD_100|VERY BAD_1000)"
    else:
        check_trans_regex += (
            r"(?:[^\S\r\n]+(WRONG|VERY BAD|VERY BAD_100|VERY BAD_1000))?"
        )
    # Group order: Tag, Transition (cm-1,) Offset (float), Unc, Offset (exponent), QN(s), Upper Energy, Upper Level Unc.
    return re.compile(check_trans_regex)


def parse_bad_transitions(
    check_trans_file: t.Union[str, Path], qn_list: t.List[str]
) -> pd.DataFrame:
    """
    Read the CheckTransitions.txt file generated by MARVEL and returns the unique set of transitions tags flagged as
    "BAD", "VERY BAD", "VAY BAD_100" or "VARY BAD_1000". Includes the input uncertainty for each bad transition and the
    current absolute energy offset from the energy level.

    Args:
        check_trans_file: The string path or Path object representing the CheckTransitions.txt file.
        qn_list:          The list of quantum numbers identifying an energy level within the MARVEL network.

    Returns:
        A DataFrame containing the tags identifying the bad transitions in the MARVEL network, their uncertainties and
            their offset from the energy levels they connect to.
    """
    check_trans_lines = []
    # check_trans_regex = (
    #     r"^([\d\w_\-]+\.\d+)[^\S\r\n]+(\d+\.\d+)[^\S\r\n]+(\d+\.\d+)[^\S\r\n]+(\d\.\d+[eE][\-\+]\d+)[^\S\r\n]+"
    #     r"(-?\d\.\d+[eE][\-\+]\d+)[^\S\r\n]+"
    # )
    # check_trans_regex += r"([^\s]+)[^\S\r\n]+" * len(qn_list)
    # check_trans_regex += r"(\d+\.\d+)[^\S\r\n]+(\d\.\d+[eE][\-\+]\d+)[^\S\r\n]+(WRONG|VERY BAD|VERY BAD_100|VERY BAD_1000)"
    # print(check_trans_regex)
    check_trans_regex = get_check_trans_regex(nqn=len(qn_list), only_bad=True)
    with open(check_trans_file, "r") as file:
        for line in list(file):
            line_match = check_trans_regex.match(line)
            if line_match is not None:
                check_trans_lines += [
                    [
                        line_match.group(1),  # Transition Tag
                        float(line_match.group(4)),  # Uncertainty
                        float(line_match.group(5)),  # Offset
                    ]
                ]

    bad_trans = pd.DataFrame(check_trans_lines, columns=["tag", "unc", "offset"])
    bad_trans["offset"] = bad_trans["offset"].abs()
    bad_trans = bad_trans.drop_duplicates(keep="first")
    bad_trans["offset_factor"] = bad_trans["offset"] / bad_trans["unc"]
    return bad_trans


def parse_check_transitions(
    check_trans_file: t.Union[str, Path],
    nqn: int,
    bad_trans_tag_list: t.List[str] = None,
) -> pd.DataFrame:
    only_bad = True if bad_trans_tag_list is None else False
    wrong_regex_group_num = 8 + nqn
    check_trans_regex = get_check_trans_regex(nqn=nqn, only_bad=only_bad)
    check_trans_lines = []
    with open(check_trans_file, "r") as file:
        for line in list(file):
            line_match = check_trans_regex.match(line)
            if line_match is not None:
                if (
                    only_bad
                    or line_match.group(1) in bad_trans_tag_list
                    or line_match.group(wrong_regex_group_num) is not None
                ):
                    check_trans_lines += [
                        [
                            line_match.group(1),  # Transition Tag
                            float(line_match.group(4)),  # Uncertainty
                            float(line_match.group(5)),  # Offset
                            line_match.group(wrong_regex_group_num)
                            is not None,  # WRONG/BAD Flag
                        ]
                    ]
    bad_trans = pd.DataFrame(
        check_trans_lines, columns=["tag", "unc", "offset", "is_bad"]
    )
    bad_trans["offset"] = bad_trans["offset"].abs()
    bad_trans = bad_trans.sort_values(by=["tag", "offset"], ascending=[1, 0])
    bad_trans = bad_trans.drop_duplicates(subset=["tag"], keep="first")
    bad_trans["offset_factor"] = bad_trans["offset"] / bad_trans["unc"]
    return bad_trans


def read_marvel_transitions(
    marvel_trans_file: t.Union[str, Path], qn_list: t.List[str]
) -> pd.DataFrame:
    trans_qn_columns = [qn + label for label in ("_u", "_l") for qn in qn_list]
    return pd.read_csv(
        marvel_trans_file,
        names=["energy", "unc_orig", "unc"] + trans_qn_columns + ["tag"],
        delim_whitespace=True,
    )


# def adjust_transition_unc(
#     unc: float, offset: float, unc_orig: float, is_bad: bool
# ) -> float:
#     unc_new = unc
#     if (offset > unc) or (offset == unc and is_bad):
#         # Transition uncertainty still too low - increase it.
#         min_step_size = 1 * 10 ** (math.floor(math.log10(unc)) - 4)
#         # -3 would be a change in the last digit of the unc as listed in check_trans, -4 allows change in the following
#         # digit not shown there.
#         step_factor = 0.001 * 10 ** math.ceil(math.log10(offset / unc))
#         unc_new = unc + max(step_factor * max(offset - unc, 0), min_step_size)
#     elif unc > offset:
#         # Transition uncertainty increased more than it needs to be - lower it, but not below original value.
#         unc_new = max((offset + unc) / 2, unc_orig)
#     return unc_new


# def optimise_bad_transitions(
#     marvel_path: t.Union[str, Path],
#     marvel_trans_file: t.Union[str, Path],
#     segment_file: t.Union[str, Path],
#     # check_trans_file: t.Union[str, Path],
#     qn_list: t.List[str],
#     marvel_trans_fortran_format_list: t.List[str],
#     min_size: int = None,
#     bootstrap_iterations: int = 100,
# ):
#     marvel_trans = read_marvel_transitions(
#         marvel_trans_file=marvel_trans_file, qn_list=qn_list
#     )
#     print(marvel_trans)
#     iteration_file_name = (
#         Path(marvel_trans_file).parent
#         / (".".join(str(marvel_trans_file).split(".")[:-1]) + "_iter.txt")
#     ).resolve()
#     print(f"Iterating transitions in file {iteration_file_name}")
#     check_trans_file = (
#         Path(marvel_trans_file).parent / r"./CheckTransitions.txt"
#     ).resolve()
#     # run_marvel(
#     #     marvel_path=marvel_path,
#     #     marvel_trans_file=marvel_trans_file,
#     #     nqn=len(qn_list),
#     #     segment_file=segment_file,
#     #     min_size=min_size,
#     #     bootstrap_iterations=bootstrap_iterations,
#     # )
#     #
#     # initial_bad_trans = parse_check_trans(
#     #     check_trans_file=check_trans_file, qn_list=qn_list
#     # )
#     # num_bad_trans = len(initial_bad_trans)
#     # current_bad_trans = initial_bad_trans.copy()
#     num_bad_trans = 1
#     iteration_num = 1
#     current_bad_trans = None
#     bad_trans_tag_list = None
#     while num_bad_trans > 0:
#         print(f"Iteration {iteration_num}")
#         if iteration_num > 1:
#             # current_bad_trans["unc_new"] = current_bad_trans.apply(
#             #     lambda x: x["unc"] + 0.01 * max(x["offset"] - x["unc"], 0), axis=1
#             # )
#             current_bad_trans["unc_new"] = current_bad_trans.apply(
#                 lambda x: adjust_transition_unc(
#                     unc=x["unc"],
#                     offset=x["offset"],
#                     unc_orig=x["unc_orig"],
#                     is_bad=x["is_bad"],
#                 ),
#                 axis=1,
#             )
#             print(f"LENGTH OF DATA: {len(marvel_trans)}, {len(current_bad_trans)}")
#             marvel_trans = marvel_trans.merge(
#                 current_bad_trans[["tag", "unc_new"]], on=["tag"], how="left"
#             )
#             marvel_trans["unc"] = np.where(
#                 marvel_trans["unc_new"].isna(),
#                 marvel_trans["unc"],
#                 marvel_trans["unc_new"],
#             )
#             del marvel_trans["unc_new"]
#             output_data(
#                 marvel_trans,
#                 filename=iteration_file_name,
#                 fortran_format_list=marvel_trans_fortran_format_list,
#             )
#         run_marvel(
#             marvel_path=marvel_path,
#             marvel_trans_file=marvel_trans_file
#             if iteration_num == 1
#             else iteration_file_name,
#             nqn=len(qn_list),
#             segment_file=segment_file,
#             min_size=min_size,
#             bootstrap_iterations=bootstrap_iterations,
#         )
#         current_bad_trans = parse_check_transitions(
#             check_trans_file=check_trans_file,
#             nqn=len(qn_list),
#             bad_trans_tag_list=bad_trans_tag_list,
#         )
#         current_bad_trans = current_bad_trans.merge(
#             marvel_trans[["tag", "unc_orig"]], on="tag", how="left"
#         )
#         print(current_bad_trans.loc[current_bad_trans["is_bad"]])
#         if iteration_num == 1:
#             bad_trans_tag_list = list(current_bad_trans["tag"].unique())
#         else:
#             bad_trans_tag_list = list(
#                 set(bad_trans_tag_list + list(current_bad_trans["tag"].unique()))
#             )
#         print(f"Length of bad transition tag list = {len(bad_trans_tag_list)}")
#
#         num_bad_trans = len(current_bad_trans.loc[current_bad_trans["is_bad"]])
#         iteration_num += 1


def update_transition_unc(
    unc: float, offset: float, unc_orig: float, is_bad: bool
) -> float:
    unc_new = unc
    if offset > unc:
        # Transition uncertainty still too low - increase it.
        unc_new = offset
    elif unc >= offset and is_bad:
        min_step_size = 1 * 10 ** (math.floor(math.log10(unc)) - 4)
        # -3 would be a change in the last digit of the unc as listed in check_trans, -4 allows change in the following
        # digit not shown there.
        unc_new = unc + min_step_size
    elif unc > offset:
        # Transition uncertainty increased more than it needs to be - lower it, but not below original value.
        order = -(math.floor(math.log10(unc)) - 3)
        if int(unc * 10**order) * 10 ** (-order) == offset:
            unc_new = max(unc - (1 * 10 ** (math.floor(math.log10(unc)) - 4)), unc_orig)
        else:
            unc_new = max((offset + unc) / 2, unc_orig)
    return unc_new


def optimise_transition_unc(
    marvel_path: t.Union[str, Path],
    marvel_trans_file: t.Union[str, Path],
    segment_file: t.Union[str, Path],
    qn_list: t.List[str],
    marvel_trans_fortran_format_list: t.List[str],
    min_size: int = None,
    bootstrap_iterations: t.Union[int, None] = 100,
):
    # TODO: This sometimes gets stuck when all bad trans are included and oscillating about an offset_factor of 1. It is
    #  best to stop and restart the process when that happens, though this may leave a few transitions with slightly
    #  higher unc than they need. Restarting stops the process from tracking and adjusting all of the previously
    #  adjusted trans, including the edge cases, and works to find a solution with the remaining bad trans. Automate
    #  detecting when the number doesn't decrease by much after X iterations and dump the current bad trans and start
    #  again.
    marvel_trans = read_marvel_transitions(
        marvel_trans_file=marvel_trans_file, qn_list=qn_list
    )
    print(marvel_trans)
    iteration_file_name = (
        Path(marvel_trans_file).parent
        / (".".join(str(marvel_trans_file).split(".")[:-1]) + "_iter.txt")
    ).resolve()
    print(f"Iterating transitions in file {iteration_file_name}")
    check_trans_file = (
        Path(marvel_trans_file).parent / r"./CheckTransitions.txt"
    ).resolve()
    num_bad_trans = 1
    iteration_num = 0
    current_bad_trans = None
    update_trans_tag_list = []
    bad_trans_tag_list = None
    while num_bad_trans > 0:
        if iteration_num > 0:
            print(f"Iteration {iteration_num}")
            update_trans = current_bad_trans.loc[
                current_bad_trans["tag"].isin(update_trans_tag_list)
            ].copy()
            update_trans["unc_new"] = update_trans.apply(
                lambda x: update_transition_unc(
                    unc=x["unc"],
                    offset=x["offset"],
                    unc_orig=x["unc_orig"],
                    is_bad=x["is_bad"],
                ),
                axis=1,
            )
            print(
                update_trans.sort_values(
                    by=["offset_factor", "unc_factor"], ascending=[False, False]
                )
            )

            print(f"LENGTH OF DATA: {len(marvel_trans)}, {len(current_bad_trans)}")
            marvel_trans = marvel_trans.merge(
                update_trans[["tag", "unc_new"]], on=["tag"], how="left"
            )
            marvel_trans["unc"] = np.where(
                marvel_trans["unc_new"].isna(),
                marvel_trans["unc"],
                marvel_trans["unc_new"],
            )
            del marvel_trans["unc_new"]
            output_data(
                marvel_trans,
                filename=iteration_file_name,
                fortran_format_list=marvel_trans_fortran_format_list,
            )
        run_marvel(
            marvel_path=marvel_path,
            marvel_trans_file=marvel_trans_file
            if iteration_num == 0
            else iteration_file_name,
            nqn=len(qn_list),
            segment_file=segment_file,
            min_size=min_size,
            bootstrap_iterations=bootstrap_iterations,
        )
        current_bad_trans = parse_check_transitions(
            check_trans_file=check_trans_file,
            nqn=len(qn_list),
            bad_trans_tag_list=bad_trans_tag_list,
        )
        del current_bad_trans["unc"]
        current_bad_trans = current_bad_trans.merge(
            marvel_trans[["tag", "unc", "unc_orig"]], on="tag", how="left"
        )
        current_bad_trans["unc_factor"] = (
            current_bad_trans["unc"] / current_bad_trans["unc_orig"]
        )
        worst_trans_tag = (
            current_bad_trans.loc[
                (current_bad_trans["is_bad"])
                & (~current_bad_trans["tag"].isin(update_trans_tag_list))
            ]
            .sort_values(by=["offset_factor", "unc_factor"], ascending=[False, False])[
                "tag"
            ]
            .values
        )
        if len(worst_trans_tag) != 0:
            worst_trans_tag = worst_trans_tag[0]
            print(f"Current worst transition: {worst_trans_tag}")
            print(current_bad_trans.loc[current_bad_trans["tag"] == worst_trans_tag])
            if iteration_num == 0:
                bad_trans_tag_list = list(current_bad_trans["tag"].unique())
                update_trans_tag_list = [worst_trans_tag]
            else:
                bad_trans_tag_list = list(
                    set(bad_trans_tag_list + list(current_bad_trans["tag"].unique()))
                )
                update_trans_tag_list = list(
                    set(update_trans_tag_list + [worst_trans_tag])
                )
        else:
            print("No bad trans?")

        print(f"Length of bad transition tag list = {len(bad_trans_tag_list)}")
        num_bad_trans = len(current_bad_trans.loc[current_bad_trans["is_bad"]])
        print(f"Number of bad transitions at end of iteration = {num_bad_trans}")
        iteration_num += 1
