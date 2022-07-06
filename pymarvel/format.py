import functools
import typing as t
from concurrent.futures import ThreadPoolExecutor
from enum import Enum

import pandas as pd


class SourceTag(Enum):
    CALCULATED: str = "Ca"
    MARVELISED: str = "Ma"
    EFFECTIVE_HAMILTONIAN: str = "EH"
    PREDICTED_SHIFT: str = "PS"
    PSEUDO_EXPERIMENTAL: str = "PE"


def output_data(
    data: pd.DataFrame, filename: str, fortran_format_list: t.List, n_workers: int = 8
):
    worker = functools.partial(format_row, fortran_format_list)

    with open(filename, "w+") as f, ThreadPoolExecutor(max_workers=n_workers) as e:
        for out_row in e.map(worker, data.itertuples(index=False)):
            f.write(out_row + "\n")


def format_row(fortran_format_list: t.List, data_row: t.Tuple) -> str:
    out_row = ""
    for i in range(0, len(data_row)):
        if i > 0:
            out_row += " "
        out_row += fortran_format(val=data_row[i], fmt=fortran_format_list[i])
    return out_row


# def output_fortran_format(data: pd.DataFrame, filename: str, fortran_format_list: list):
#     with open(filename, 'w+') as f:
#         for row in data.itertuples(index=False):
#             out_row = ''
#             for i in range(0, len(row)):
#                 out_row = out_row + ' '
#                 val = row[i]
#                 fmt = fortran_format_list[i]
#                 out_val = fortran_format(val=val, fmt=fmt)
#                 out_row = out_row + out_val
#             f.write(out_row + '\n')


def fortran_format(val: str, fmt: str) -> str:
    fmt_letter = fmt[0]
    fmt = fmt[1:]
    if fmt_letter == "a" or (fmt_letter == "i" and pd.isna(val)):
        if len(fmt) == 0:
            return val
        else:
            # out_val = '{val:>{fmt}}'.format(val=val, fmt=fmt)
            return "{val:>{fmt}}".format(val=val, fmt=fmt)
    # if fmt_letter == 'e':
    #     val = float(val)
    #     fmt_w = int(fmt.split('.')[0])
    #     fmt_e = int(fmt.split('.')[1])
    #     out_val = '{val:.{fmt_e}e}'.format(val=val, fmt_e=fmt_e)
    #     out_val = '{val:>{fmt_w}}'.format(val=out_val, fmt_w=fmt_w)
    #     return out_val
    # if fmt_letter == 'E':
    #     val = float(val)
    #     fmt_w = int(fmt.split('.')[0])
    #     fmt_e = int(fmt.split('.')[1])
    #     out_val = '{val:.{fmt_e}E}'.format(val=val, fmt_e=fmt_e)
    #     out_val = '{val:>{fmt_w}}'.format(val=out_val, fmt_w=fmt_w)
    #     return out_val
    if fmt_letter in ["e", "E", "f"]:
        val = float(val)
        return "{val:{fmt}{fmt_letter}}".format(val=val, fmt=fmt, fmt_letter=fmt_letter)
    # elif fmt_letter == 'f':
    #     val = float(val)
    #     # fmt_w = int(fmt.split('.')[0])
    #     # fmt_d = int(fmt.split('.')[1])
    #     # out_val = '{val:.{fmt_d}f}'.format(val=val, fmt_d=fmt_d)
    #     # out_val = '{val:>{fmt_w}}'.format(val=out_val, fmt_w=fmt_w)
    #     return '{val:{fmt}f}'.format(val=val, fmt=fmt)
    # elif fmt_letter == 'g':
    #     val = float(val)
    #     fmt_w = int(fmt.split('.')[0])
    #     fmt_e = int(fmt.split('.')[1])
    #     out_val = '{val:#.{fmt_e}g}'.format(val=val, fmt_e=fmt_e)
    #     out_val = '{val:>{fmt_w}}'.format(val=out_val, fmt_w=fmt_w)
    #     return out_val
    # elif fmt_letter == 'G':
    #     val = float(val)
    #     fmt_w = int(fmt.split('.')[0])
    #     fmt_e = int(fmt.split('.')[1])
    #     out_val = '{val:#.{fmt_e}G}'.format(val=val, fmt_e=fmt_e)
    #     out_val = '{val:>{fmt_w}}'.format(val=out_val, fmt_w=fmt_w)
    #     return out_val
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
            # out_val = '{val:0>{fmt_m}d}'.format(val=val, fmt_m=fmt_m)
            # out_val = '{val:>{fmt_w}}'.format(val=out_val, fmt_w=fmt_w)
            return "{val:>{fmt_w}}".format(val=str(val).zfill(fmt_m), fmt_w=fmt_w)
        else:
            # out_val = '{val:{pad}d}'.format(val=val, pad=fmt)
            return "{val:{pad}d}".format(val=val, pad=fmt)


def create_tag(source_name: str, index: int, length: int) -> str:
    return f"{source_name}.{index:0{length}d}"


# TODO: Move these to testing.
# fortran_format('12', 'i12.3')
# fortran_format('12', 'i11')
# fortran_format('12.8483', 'f10.2')
# fortran_format('12.34534534', 'f10.9')
# fortran_format('beans', 'a')
# fortran_format('beans', 'a19')
# fortran_format('1.389478939845', 'e10.3')
# fortran_format('5000000.7771389478939845', 'g20.5')
# fortran_format('5000000.7771389478939845', 'G20.5')
# fortran_format('50000.77', 'G10.8')
# fortran_format('502345700.7771389478939845', 'e12.5')
# fortran_format('23485723.4562735234', 'E12.7')
# fortran_format('inf', 'e12.5')
# fortran_format('nan', 'e12.5')
