import functools
import typing as t
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from pathlib import Path

import pandas as pd


class SourceTag(Enum):
    CALCULATED = "Ca"
    MARVELISED = "Ma"
    EFFECTIVE_HAMILTONIAN = "EH"
    PREDICTED_SHIFT = "PS"
    PS_PARITY_PAIR = "PS_1"
    PS_LINEAR_REGRESSION = "PS_2"
    PS_EXTRAPOLATION = "PS_3"
    PSEUDO_EXPERIMENTAL = "PE"

    def __str__(self):
        return str(self.value)

    def format_output(self):
        if not pd.isna(self) and self in [
            self.PS_PARITY_PAIR,
            self.PS_LINEAR_REGRESSION,
            self.PS_EXTRAPOLATION,
        ]:
            # These subtypes of Predicted Shift should all be output as Predicted Shift.
            return self.PREDICTED_SHIFT
        else:
            return self


# TODO: Add implicit handling for states file output in the below format where "?" denotes optional column and "+" one
#  or more columns matching the description.
#  ID | Energy | g | J/F | unc? | lifetime? | parity_tot | parity_norot? | state/sym | sym_num? | ns_iso? | vib_qn+ |
#  other_qn+ | source_tag.
def output_data(
    data: pd.DataFrame,
    filename: t.Union[str, Path],
    fortran_format_list: t.List[str],
    n_workers: int = 8,
    append: bool = False,
) -> None:
    if "source_tag" in data.columns:
        data["source_tag"] = data["source_tag"].map(SourceTag.format_output)

    worker = functools.partial(format_row, fortran_format_list)
    if append:
        file_mode = "a"
    else:
        file_mode = "w+"
    with open(filename, file_mode) as f, ThreadPoolExecutor(max_workers=n_workers) as e:
        for out_row in e.map(worker, data.itertuples(index=False)):
            f.write(out_row + "\n")


def format_row(fortran_format_list: t.List, data_row: t.Tuple) -> str:
    out_row = ""
    for i in range(0, len(data_row)):
        if i > 0:
            out_row += " "
        out_row += fortran_format(val=data_row[i], fmt=fortran_format_list[i])
    return out_row


def fortran_format(val: str, fmt: str) -> str:
    fmt_letter = fmt[0]
    fmt = fmt[1:]
    if fmt_letter == "a" or (fmt_letter == "i" and pd.isna(val)):
        if len(fmt) == 0:
            return val
        else:
            return "{val:>{fmt}}".format(val=val, fmt=fmt)
    if fmt_letter in ["e", "E", "f"]:
        val = float(val)
        return "{val:{fmt}{fmt_letter}}".format(val=val, fmt=fmt, fmt_letter=fmt_letter)
    elif fmt_letter in ["g", "G"]:
        val = float(val)
        return "{val:#{fmt}{fmt_letter}}".format(
            val=val, fmt=fmt, fmt_letter=fmt_letter
        )
    elif fmt_letter == "i":
        val = int(val)
        if "." in fmt:
            fmt_w = int(fmt.split(".")[0])
            fmt_m = int(fmt.split(".")[1])
            return "{val:>{fmt_w}}".format(val=str(val).zfill(fmt_m), fmt_w=fmt_w)
        else:
            return "{val:{pad}d}".format(val=val, pad=fmt)


def create_tag(source_name: str, index: int, length: int) -> str:
    return f"{source_name}.{index:0{length}d}"
