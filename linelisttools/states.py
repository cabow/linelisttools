import functools
import math
import typing as t
from collections.abc import Iterable
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
from pandas.core.groupby import GroupBy
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler

from .concurrence import ExecutorType, yield_grouped_data
from .format import SourceTag
from .plot import get_vibrant_colors


# TODO: Pass ExoMolStatesHeader in to methods to avoid passing individual column names. Can also be used for formatting
#  by containing the fortran format mapping.
#  Also add some way to handle with the resultant energy columns, i.e.: energy_final, etc.
class ExoMolStatesHeader:
    """
    Stores the column names of an ExoMol states file. Certain columns are mandatory (ID, Energy, Degeneracy and a
    rigorous quanutum number) and appear in the first four columns. Others have set positions if they exist such as
    uncertainty which appears fifth; the lifetime appears after uncertainty if it exists else it is fifth. Other columns
    representing parity (i.e.: total, rotationless), symmetry (electronic state, group symmetry), counting numbers
    (i.e.: symmetry block counting), isomer labelling (i.e.: nuclear spin isomer, structural isomer), vibrational
    quantum numbers and any other quantum numbers follow these in order and can each consist of multiple columns. The
    final possible column is the source tag, which generally only appears in states file that have been Marvelised (or
    are isotopologues of one).
    """

    _state_id_default = "id"
    _energy_default = "energy"
    _degeneracy_default = "degeneracy"
    _j_qn_default = "J"
    _hyperfine_qn_default = "F"
    _nuclear_spin_default = "I"
    _unc_default = "unc"
    _lifetime_default = "lifetime"
    _source_tag_default = "source_tag"

    # TODO: Come up with a better solution than this, because indexing these options out to pass into the constructor is
    #  very janky.
    class StatesParity(Enum):
        TOTAL_PARITY = "parity_tot"
        ROTATIONLESS_PARITY = "parity_norot"

        def __str__(self):
            return str(self.value)

    def __init__(
        self,
        state_id: str = _state_id_default,
        energy: str = _energy_default,
        degeneracy: str = _degeneracy_default,
        is_hyperfine: bool = False,
        hyperfine_qn: str = _hyperfine_qn_default,
        nuclear_spin: str = _nuclear_spin_default,
        j_qn: str = _j_qn_default,
        unc: t.Optional[str] = _unc_default,
        lifetime: t.Optional[str] = _lifetime_default,
        parity: t.Union[StatesParity, t.List[StatesParity]] = StatesParity.TOTAL_PARITY,
        symmetry: t.Union[str, t.List[str]] = None,
        counting_number: t.Optional[t.Union[str, t.List[str]]] = None,
        isomer: t.Optional[t.Union[str, t.List[str]]] = None,
        vibrational_qn: t.Union[str, t.List[str]] = None,
        other_qn: t.Optional[t.Union[str, t.List[str]]] = None,
        source_tag: t.Optional[str] = _source_tag_default,
    ):
        if not is_hyperfine:
            hyperfine_qn = None
            nuclear_spin = None

        self._state_id = state_id
        self._energy = energy
        self._degeneracy = degeneracy
        self._is_hyperfine = is_hyperfine
        self._hyperfine_qn = hyperfine_qn
        self._nuclear_spin = nuclear_spin
        self._j_qn = j_qn
        self._unc = unc
        self._lifetime = lifetime
        self._parity = parity
        self._symmetry = symmetry
        self._counting_number = counting_number
        self._isomer = isomer
        self._vibrational_qn = vibrational_qn
        self._other_qn = other_qn
        self._source_tag = source_tag

    def get_header(self) -> t.List[str]:
        def flatten(nested_list):
            for item in nested_list:
                if isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
                    yield from flatten(item)
                else:
                    yield str(item)

        if self._is_hyperfine:
            header_order = [
                self._state_id,
                self._energy,
                self._degeneracy,
                self._hyperfine_qn,
                self._nuclear_spin,
                self._unc,
                self._lifetime,
                self._parity,
                self._j_qn,
                self._symmetry,
                self._counting_number,
                self._isomer,
                self._vibrational_qn,
                self._other_qn,
                self._source_tag,
            ]
        else:
            header_order = [
                self._state_id,
                self._energy,
                self._degeneracy,
                self._j_qn,
                self._unc,
                self._lifetime,
                self._parity,
                self._symmetry,
                self._counting_number,
                self._isomer,
                self._vibrational_qn,
                self._other_qn,
                self._source_tag,
            ]
        header_order = [column for column in header_order if column is not None]

        return list(flatten(header_order))

    @property
    def state_id(self) -> str:
        return self._state_id

    @state_id.setter
    def state_id(self, value: str):
        self._state_id = value

    @property
    def energy(self) -> str:
        return self._energy

    @energy.setter
    def energy(self, value: str):
        self._energy = value

    @property
    def degeneracy(self) -> str:
        return self._degeneracy

    @degeneracy.setter
    def degeneracy(self, value: str):
        self._degeneracy = value

    @property
    def is_hyperfine(self) -> bool:
        return self._is_hyperfine

    @is_hyperfine.setter
    def is_hyperfine(self, value: bool):
        self._is_hyperfine = value

    @property
    def hyperfine_qn(self) -> str:
        return self._hyperfine_qn

    @hyperfine_qn.setter
    def hyperfine_qn(self, value: str):
        self._hyperfine_qn = value

    @property
    def nuclear_spin(self) -> str:
        return self._nuclear_spin

    @nuclear_spin.setter
    def nuclear_spin(self, value: str):
        self._nuclear_spin = value

    @property
    def j_qn(self) -> str:
        return self._j_qn

    @j_qn.setter
    def j_qn(self, value: str):
        self._j_qn = value

    @property
    def unc(self) -> t.Optional[str]:
        return self._unc

    @unc.setter
    def unc(self, value: t.Optional[str]):
        self._unc = value

    def default_unc(self):
        self._unc = self._unc_default

    @property
    def lifetime(self) -> str:
        return self._lifetime

    @lifetime.setter
    def lifetime(self, value: str):
        self._lifetime = value

    def default_lifetime(self):
        self._lifetime = self._lifetime_default

    @property
    def parity(self) -> t.Union[StatesParity, t.List[StatesParity]]:
        return self._parity

    @parity.setter
    def parity(self, value: t.Union[StatesParity, t.List[StatesParity]]):
        self._parity = value

    @property
    def symmetry(self) -> t.Union[str, t.List[str]]:
        return self._symmetry

    @symmetry.setter
    def symmetry(self, value: t.Union[str, t.List[str]]):
        self._symmetry = value

    @property
    def counting_number(self) -> t.Optional[t.Union[str, t.List[str]]]:
        return self._counting_number

    @counting_number.setter
    def counting_number(self, value: t.Optional[t.Union[str, t.List[str]]]):
        self._counting_number = value

    @property
    def isomer(self) -> t.Optional[t.Union[str, t.List[str]]]:
        return self._isomer

    @isomer.setter
    def isomer(self, value: t.Optional[t.Union[str, t.List[str]]]):
        self._isomer = value

    @property
    def vibrational_qn(self) -> t.Union[str, t.List[str]]:
        return self._vibrational_qn

    @vibrational_qn.setter
    def vibrational_qn(self, value: t.Union[str, t.List[str]]):
        self._vibrational_qn = value

    @property
    def other_qn(self) -> t.Optional[t.Union[str, t.List[str]]]:
        return self._other_qn

    @other_qn.setter
    def other_qn(self, value: t.Optional[t.Union[str, t.List[str]]]):
        self._other_qn = value

    @property
    def source_tag(self) -> t.Optional[str]:
        return self._source_tag

    @source_tag.setter
    def source_tag(self, value: t.Optional[str]):
        self._source_tag = value

    def default_source_tag(self):
        self._source_tag = self._source_tag_default

    def get_rigorous_qn(self) -> str:
        if self.is_hyperfine:
            return self._hyperfine_qn
        else:
            return self.j_qn


def read_states_file(
    file: str,
    states_cols: t.List[str],
) -> pd.DataFrame:
    """
    Reads an ExoMol states file into a DataFrame object and returns it.

    Args:
        file:        The string path to the state file.
        states_cols: The columns contained in the states file.

    Returns:
        A DataFrame object containing the states file data.
    """
    return pd.read_csv(file, delim_whitespace=True, names=states_cols)


def read_exomol_states(
    file: str, exomol_states_header: ExoMolStatesHeader
) -> pd.DataFrame:
    """
    Reads an ExoMol states file into a DataFrame object and returns it.

    An ExoMol states file's first four columns must be: ID, Energy, g (total degeneracy) and the rigorous quantum number
    J (or F in the case of hyperfine resolved states). If an uncertainty is present then it will be the fifth column.
    Likewise, lifetime will be the subsequent column if present. These should be followed by the parity column(s), the
    electronic state (for diatomics) or symmetry (for triatomics or polyatomics). These can be followed some
    symmetry/group counting number column(s) and isomer column(s) (i.e.: nuclear spin isomer, structural isomer). The
    vibrational quantum number column(s) and then any remaining other quantum number column(s) follow. The last
    potential column is the source tag column.

    Args:
        file:                 The string path to the state file.
        exomol_states_header: An ExoMolStatesHeader object containing the string names for the states file columns.

    Returns:
        A DataFrame object containing the states file data.
    """
    return pd.read_csv(
        file, names=exomol_states_header.get_header(), delim_whitespace=True
    )


def parity_norot_to_total(parity_norot: str, j_val: float) -> str:
    parity_factor = 1 if parity_norot == "e" else -1 if parity_norot == "f" else None
    if parity_factor is None:
        raise ValueError(f"Input rotationless parity was not e or f: {parity_norot}")
    j_exponent_factor = 0 if j_val.is_integer() else 0.5
    j_term = (-1) ** (j_val - j_exponent_factor) * parity_factor
    parity_tot = "+" if j_term == 1 else "-" if j_term == -1 else None
    return parity_tot


def parity_total_to_norot(parity_tot: str, j_val: float) -> str:
    parity_factor = 1 if parity_tot == "+" else -1 if parity_tot == "-" else None
    if parity_factor is None:
        raise ValueError(f"Input total parity was not + or -: {parity_tot}")
    j_exponent_factor = 0 if j_val.is_integer() else 0.5
    j_term = (-1) ** (j_val - j_exponent_factor) / parity_factor
    parity_tot = "e" if j_term == 1 else "f" if j_term == -1 else None
    return parity_tot


def propagate_error_in_mean(unc_list: t.List[float]) -> float:
    unc_sq_list = [unc**2 for unc in unc_list]
    sum_unc_sq = sum(unc_sq_list)
    sqrt_sum_unc_sq = math.sqrt(sum_unc_sq)
    error = sqrt_sum_unc_sq / len(unc_list)
    return error


# TODO: There is an edge case when doing a minimal quantum number, closest-energy match where in theory two levels with
#  the same rigorous quantum numbers could match with the same calculated level. Switch the simple dropduplicates()
#  back to the old method of finding the duplicated idxs and ordering them, then perform a rank and compare whereby if
#  two levels A, B match calculated x, y and both are closest to x such that A-x=1, B-x=-2, A-y=10, B-y=7 then though
#  both are closest to x, A is closer so A will match with x and B will then defer to its second match and match with y.
#  Answer: assign each match a ranking based on closeness, with 1 being the closest. Check if any mvl energies have the
#  same ID calculated energy as their rank 1 match, then apply a next-closest approach to determine which gets the
#  closest match.
def match_states(
    states_calc: pd.DataFrame,
    states_obs: pd.DataFrame,
    qn_match_cols: t.List[str],
    match_source_tag: SourceTag,
    states_header: ExoMolStatesHeader,
    states_new_qn_cols: t.List[str] = None,
    suffixes: t.Tuple[str, str] = None,
    is_isotopologue_match: bool = False,
    overwrite_non_match_qn_cols: bool = False,
    check_zero_energies: bool = True,
) -> pd.DataFrame:
    """
    Matches a set of calculated and observational states, generally for the purpose of updating calculated states with
    Marvelised energies. Matching states are determined by a input list of quantum numbers to match on, returning the
    state with the smallest absolute obs.-calc. when there are duplicate matches. Duplicate matches should only occur
    when the list of quantum numbers to match on is not the full set, such as in cases where the calculated states do
    not contain a full quantum number assignment. In this case, the extra quantum numbers from the observational states
    that were not matched on can be specified to be carried over to the output or used to overwrite their existing
    values in the calculated states. The latter use case is particularly useful for adding assignments from Marvel
    networks to calculated states files without full assignments, which is generally the case for triatomics and more
    complicated molecules.

    Isotpoologue matches can also be performed, whereby the obs.-calc. for the matching states of the primary species is
    copied over to equivalently assigned states of the isotopologue species.
    TODO: Double-check that the isotopologue match works as intended; create test cases.
    Args:
        states_calc:                 A DataFrame containing the calculated states file data.
        states_obs:                  A DataFrame containing the observational/Marvel states.
        qn_match_cols:               The list of quantum numbers to match the two sets of states on.
        match_source_tag:            The source tag to set for the states that match between sets.
        states_header:               The ExoMolStatesHeader object containing the column mappings for the calculated
            states file
        states_new_qn_cols:          A list of quantum number columns that may not be present in either of the states
            files that are being matched that should be preserved in the output states.
        suffixes:                    A tuple containing suffixes for any equivalently names states file columns that are
            not matched on.
        is_isotopologue_match:       A boolean determining whether the observed states are for an isotopologue of the
            calculated states.
        overwrite_non_match_qn_cols: A boolean determining whether any quantum number columns, which were not matched
            on, should have their original calculated value overwritten by the values from the observational states.
            Useful for overwriting placeholder NaN quantum numbers with values from observational assignments.
        check_zero_energies:         A boolean determining whether a comparison is made to check whether the zero-energy
            state of the matching states are the same. Defaults to True; set to false when matching a subset of states
            that do not start at zero.

    Returns:
        A Dataframe representing the matched states file.
    """
    if suffixes is None:
        suffixes = ("_calc", "_obs")

    if states_new_qn_cols is None:
        states_new_qn_cols = qn_match_cols

    # Take an inner merge to only get the levels that do have matches.
    states_matched = states_calc.merge(
        states_obs,
        left_on=qn_match_cols,
        right_on=qn_match_cols,
        suffixes=suffixes,
        how="inner",
    )
    states_header.default_unc()
    print("STATES MATCHED: \n", states_matched)
    if len(states_matched) == 0:
        raise RuntimeWarning(
            "No matching levels found. New levels will be appended to existing set."
        )
        # TODO: Change to error, or add input fail_on_no_matches?

    energy_calc_col = states_header.energy + suffixes[0]
    energy_obs_col = states_header.energy + suffixes[1]
    energy_dif_col = states_header.energy + "_dif"
    energy_dif_mag_col = energy_dif_col + "_mag"

    states_matched[energy_dif_col] = states_matched.apply(
        lambda x: x[energy_obs_col] - x[energy_calc_col], axis=1
    )
    states_matched[energy_dif_mag_col] = states_matched[energy_dif_col].map(abs)

    # This was bad as it changed the structure of the output DataFrame to be that of the GroupBy frame.
    # levels_matched = levels_matched.sort_values(energy_dif_mag_col).groupby(by=qn_match_cols, as_index=False).first()

    qn_cols_not_match = np.setdiff1d(states_new_qn_cols, qn_match_cols)
    # If not matching on the full set of quantum numbers that uniquely determine a level, check for duplicates such that
    # no more than one level (defined by the input set of quantum numbers qn_match_cols) is matching on the same
    # levels_new level.
    qn_dupe_cols = (
        qn_match_cols
        if len(qn_cols_not_match) == 0
        else qn_match_cols + [energy_obs_col]
    )

    # TODO: Include energy_obs in sort list as this will be the same for any duplicate states_obs joined on?
    print(states_matched[qn_dupe_cols + [energy_dif_mag_col]])
    states_matched = states_matched.sort_values(
        by=qn_dupe_cols + [energy_dif_mag_col]
    ).drop_duplicates(subset=qn_dupe_cols, keep="first")
    states_matched = states_matched.sort_values(by=states_header.state_id)

    # Remove the energy difference magnitude column as it is not needed beyond this point.
    del states_matched[energy_dif_mag_col]

    # Check the 0 energy levels are the same.
    if check_zero_energies:
        zero_energy_level_matches = len(
            states_matched.loc[
                (states_matched[energy_calc_col] == 0)
                & (states_matched[energy_obs_col] == 0)
            ]
        )
        if zero_energy_level_matches != 1:
            raise RuntimeError(
                f"0 ENERGY LEVELS DO NOT MATCH ASSIGNMENTS IN BOTH DATASETS.\n"
                f"ORIGINAL: {states_matched.loc[states_matched[energy_calc_col] == 0]}\n"
                f"UPDATE: {states_matched.loc[states_matched[energy_obs_col] == 0]}\n"
            )

    if not is_isotopologue_match:
        # Merge the sets of qn_match_cols in each DataFrame with an indicator to find any rows that are only in
        # levels_new to concat them to levels_matched.
        states_new_check_matched = states_obs[qn_match_cols].merge(
            states_matched[qn_match_cols], how="left", indicator=True
        )
        # Get the indexes of levels_new where the unique qn_match_cols are not yet in levels_matched.
        states_new_to_concat_idx = states_new_check_matched[
            (states_new_check_matched["_merge"] == "left_only")
        ].index
        # Isolate only the rows with those indexes to be concatenated.
        states_new_to_concat = states_obs[
            states_obs.index.isin(states_new_to_concat_idx)
        ]
        # Rename energy_col_name in the rows to be concatenated to energy_obs_col to avoid creating a new column.
        states_new_to_concat = states_new_to_concat.rename(
            columns={states_header.energy: energy_obs_col}
        )
        if len(qn_cols_not_match) > 0:
            states_new_to_concat = states_new_to_concat.rename(
                columns={qn_col: qn_col + suffixes[1] for qn_col in qn_cols_not_match}
            )
        states_matched = pd.concat([states_matched, states_new_to_concat])

    if states_header.source_tag is None:
        states_header.default_source_tag()
    states_matched[states_header.source_tag] = match_source_tag
    states_matched["energy_final"] = states_matched[energy_obs_col]

    # Rename original levels' energy to match the column name in levels_matched
    states_calc = states_calc.rename(columns={states_header.energy: energy_calc_col})
    # Add missing original levels that are not in the final matching set
    states_calc_to_concat = states_calc.loc[
        ~states_calc[states_header.state_id].isin(
            states_matched[states_header.state_id]
        )
    ]
    if len(qn_cols_not_match) > 0:
        states_calc_to_concat = states_calc_to_concat.rename(
            columns={qn_col: qn_col + suffixes[0] for qn_col in qn_cols_not_match}
        )

    if (
        states_header.unc in states_calc_to_concat.columns
        and states_header.unc + suffixes[0] in states_matched.columns
    ):
        states_calc_to_concat = states_calc_to_concat.rename(
            columns={states_header.unc: states_header.unc + suffixes[0]}
        )

    states_matched = pd.concat([states_matched, states_calc_to_concat])

    # Create table to provide energy shifts and std (for unc estimates) based on a qn grouping.
    # shift_table = generate_shift_table(states=levels_matched, shift_table_qn_cols=shift_table_qn_cols,
    #                                    energy_calc_col=energy_calc_col, energy_obs_col=energy_obs_col,
    #                                    energy_dif_col=energy_dif_col, unc_col=unc_col)

    if overwrite_non_match_qn_cols and len(qn_cols_not_match) > 0:
        for qn_col_not_match in qn_cols_not_match:
            states_matched[qn_col_not_match] = np.where(
                states_matched[qn_col_not_match + suffixes[1]].isna(),
                states_matched[qn_col_not_match + suffixes[0]],
                states_matched[qn_col_not_match + suffixes[1]],
            )
            del states_matched[qn_col_not_match + suffixes[0]]
            del states_matched[qn_col_not_match + suffixes[1]]

    return states_matched


# def match_levels(
#     levels_initial: pd.DataFrame,
#     levels_new: pd.DataFrame,
#     qn_match_cols: t.List[str],
#     match_source_tag: SourceTag,
#     levels_new_qn_cols: t.List[str] = None,
#     suffixes: t.Tuple[str, str] = None,
#     energy_col: str = "energy",
#     unc_col: str = "unc",
#     source_tag_col: str = "source_tag",
#     id_col: str = "ID",
#     is_isotopologue_match: bool = False,
#     overwrite_non_match_qn_cols: bool = False,
# ) -> pd.DataFrame:
#     """
#     This is an old version of match_states that does not use the ExoMolStatesHeader class.
#
#     Does passing out the shift_table make things easier or is it too constraining for later processes to use the
#     predefined grouping from the match quantum numbers? Allow subsequent methods to redo the grouping or is that just
#     extra room for error/mistakes?
#
#     Args:
#         levels_initial:
#         levels_new:
#         qn_match_cols:
#         match_source_tag:
#         levels_new_qn_cols:
#         suffixes:
#         energy_col:
#         unc_col:
#         source_tag_col:
#         id_col:
#         is_isotopologue_match:
#         overwrite_non_match_qn_cols:
#
#     Returns:
#
#     """
#     if suffixes is None:
#         suffixes = ("_calc", "_obs")
#
#     if levels_new_qn_cols is None:
#         levels_new_qn_cols = qn_match_cols
#
#     # Take an inner merge to only get the levels that do have matches.
#     levels_matched = levels_initial.merge(
#         levels_new,
#         left_on=qn_match_cols,
#         right_on=qn_match_cols,
#         suffixes=suffixes,
#         how="inner",
#     )
#     print("LEVELS MATCHED: \n", levels_matched)
#     if len(levels_matched) == 0:
#         raise RuntimeWarning(
#             "No matching levels found. New levels will be appended to existing set."
#         )
#         # TODO: Change to error, or add input fail_on_no_matches?
#
#     energy_calc_col = energy_col + suffixes[0]
#     energy_obs_col = energy_col + suffixes[1]
#     energy_dif_col = energy_col + "_dif"
#     energy_dif_mag_col = energy_dif_col + "_mag"
#
#     levels_matched[energy_dif_col] = levels_matched.apply(
#         lambda x: x[energy_obs_col] - x[energy_calc_col], axis=1
#     )
#     levels_matched[energy_dif_mag_col] = levels_matched[energy_dif_col].apply(
#         lambda x: abs(x)
#     )
#
#     # This was bad as it changed the structure of the output DataFrame to be that of the GroupBy frame.
#     # levels_matched = levels_matched.sort_values(energy_dif_mag_col).groupby(by=qn_match_cols, as_index=False).first()
#
#     qn_cols_not_match = np.setdiff1d(levels_new_qn_cols, qn_match_cols)
#     # If not matching on the full set of quantum numbers that uniquely determine a level, check for duplicates such that
#     # no more than one level (defined by the input set of quantum numbers qn_match_cols) is matching on the same
#     # levels_new level.
#     qn_dupe_cols = (
#         qn_match_cols
#         if len(qn_cols_not_match) == 0
#         else qn_match_cols + [energy_obs_col]
#     )
#
#     levels_matched_dup = levels_matched[
#         levels_matched.duplicated(subset=qn_dupe_cols, keep=False)
#     ].sort_values(qn_dupe_cols + [energy_dif_mag_col])
#
#     if len(levels_matched_dup) > 0:
#         # Get the index of the lowest energy_agreement entry, for each tag which has duplicates
#         levels_matched_dup_idx_min = (
#             levels_matched_dup.groupby(by=qn_dupe_cols, sort=False)[
#                 energy_dif_mag_col
#             ].transform(min)
#             == levels_matched_dup[energy_dif_mag_col]
#         )
#         # Take the index of everything other than the lowest energy_agreement entry for each duplicated tag and remove
#         # it from the comparison dataframe
#         levels_matched = levels_matched.drop(
#             levels_matched_dup[~levels_matched_dup_idx_min].index
#         )
#     # levels_matched = levels_matched.sort_values(by=[qn_dupe_cols + [energy_dif_mag_col]]).drop_duplicates(
#     #     subset=qn_dupe_cols, keep='first')
#
#     # Remove the energy difference magnitude column as it is not needed beyond this point.
#     del levels_matched[energy_dif_mag_col]
#
#     # Check the 0 energy level.
#     zero_energy_level_matches = len(
#         levels_matched.loc[
#             (levels_matched[energy_calc_col] == 0)
#             & (levels_matched[energy_obs_col] == 0)
#         ]
#     )
#     if zero_energy_level_matches != 1:
#         raise RuntimeError(
#             "0 ENERGY LEVELS DO NOT MATCH ASSIGNMENTS IN BOTH DATASETS.\nORIGINAL:\n",
#             levels_matched.loc[levels_matched[energy_calc_col] == 0],
#             "\nUPDATE:\n",
#             levels_matched.loc[levels_matched[energy_obs_col] == 0],
#         )
#
#     if not is_isotopologue_match:
#         # Merge the sets of qn_match_cols in each DataFrame with an indicator to find any rows that are only in
#         # levels_new to concat them to levels_matched.
#         levels_new_check_matched = levels_new[qn_match_cols].merge(
#             levels_matched[qn_match_cols], how="left", indicator=True
#         )
#         # Get the indexes of levels_new where the unique qn_match_cols are not yet in levels_matched.
#         levels_new_to_concat_idx = levels_new_check_matched[
#             (levels_new_check_matched["_merge"] == "left_only")
#         ].index
#         # Isolate only the rows with those indexes to be concatenated.
#         levels_new_to_concat = levels_new[
#             levels_new.index.isin(levels_new_to_concat_idx)
#         ]
#         # Rename energy_col_name in the rows to be concatenated to energy_obs_col to avoid creating a new column.
#         levels_new_to_concat = levels_new_to_concat.rename(
#             columns={energy_col: energy_obs_col}
#         )
#         if len(qn_cols_not_match) > 0:
#             levels_new_to_concat = levels_new_to_concat.rename(
#                 columns={qn_col: qn_col + suffixes[1] for qn_col in qn_cols_not_match}
#             )
#         # levels_matched = levels_matched.append(levels_new_to_append)
#         levels_matched = pd.concat([levels_matched, levels_new_to_concat])
#
#     levels_matched[source_tag_col] = match_source_tag.value
#     # TODO: Change to rename original column? Or worth keeping both?
#     levels_matched["energy_final"] = levels_matched[energy_obs_col]
#
#     # Rename original levels' energy to match the column name in levels_matched
#     levels_initial = levels_initial.rename(columns={energy_col: energy_calc_col})
#     # Add missing original levels that are not in the final matching set
#     # levels_matched = levels_matched.append(levels_initial.loc[~levels_initial['id'].isin(levels_matched['id'])])
#     levels_initial_to_concat = levels_initial.loc[
#         ~levels_initial[id_col].isin(levels_matched[id_col])
#     ]
#     if len(qn_cols_not_match) > 0:
#         levels_initial_to_concat = levels_initial_to_concat.rename(
#             columns={qn_col: qn_col + suffixes[0] for qn_col in qn_cols_not_match}
#         )
#
#     if (
#         unc_col in levels_initial_to_concat.columns
#         and unc_col + suffixes[0] in levels_matched.columns
#     ):
#         levels_initial_to_concat = levels_initial_to_concat.rename(
#             columns={unc_col: unc_col + suffixes[0]}
#         )
#
#     levels_matched = pd.concat([levels_matched, levels_initial_to_concat])
#
#     # Create table to provide energy shifts and std (for unc estimates) based on a qn grouping.
#     # shift_table = generate_shift_table(states=levels_matched, shift_table_qn_cols=shift_table_qn_cols,
#     #                                    energy_calc_col=energy_calc_col, energy_obs_col=energy_obs_col,
#     #                                    energy_dif_col=energy_dif_col, unc_col=unc_col)
#     # print("SHIFT TABLE: \n", shift_table)
#
#     if overwrite_non_match_qn_cols and len(qn_cols_not_match) > 0:
#         for qn_col_not_match in qn_cols_not_match:
#             levels_matched[qn_col_not_match] = np.where(
#                 levels_matched[qn_col_not_match + suffixes[1]].isna(),
#                 levels_matched[qn_col_not_match + suffixes[0]],
#                 levels_matched[qn_col_not_match + suffixes[1]],
#             )
#             del levels_matched[qn_col_not_match + suffixes[0]]
#             del levels_matched[qn_col_not_match + suffixes[1]]
#
#     return levels_matched


def generate_shift_table(
    states: pd.DataFrame,
    shift_table_qn_cols: t.List[str],
    energy_calc_col: str = "energy_calc",
    energy_obs_col: str = "energy_obs",
    energy_dif_col: str = "energy_dif",
    unc_col: str = "unc",
) -> pd.DataFrame:
    shift_table = (
        states.loc[(~states[energy_calc_col].isna()) & (~states[energy_obs_col].isna())]
        .groupby(by=shift_table_qn_cols, as_index=False)
        .apply(
            lambda x: pd.Series(
                {
                    "energy_dif_mean": np.average(x[energy_dif_col]),
                    "energy_dif_unc": propagate_error_in_mean(x[unc_col]),
                }
            )
        )
    )
    return shift_table


def estimate_uncertainty(
    j_val: float, v_val: float, j_factor: float, v_factor: float
) -> float:
    """
    Provides an uncertainty estimate for a state based on a quadratic J and linear v term.
    Currently, only intended for use with diatomics, due to the singular vibrational quantum number input.

    Args:
        j_val:    The J value of the state.
        v_val:    The v value of the state.
        j_factor: The value to scale the quadratic J term by.
        v_factor: The value to scale the linear v term by.

    Returns:
        A float representing an uncertainty estimate for a state.
    """

    return j_factor * j_val * (j_val + 1) + v_factor * v_val


def set_predicted_unc(
    std: float,
    j_factor: float,
    v_factor: float,
    j_val: float = 0,
    j_max: float = 0,
    v_val: float = 0,
    v_max: float = 0,
) -> float:
    """
    Provides an increased uncertainty estimate for a state based on an initial input uncertainty and a calculated
    increase proportional to the states quantum numbers J and v. This increase is obtained from
    :func:`linelisttools.states.estimate_uncertainty`.

    In cases where uncertainties are to be determined for higher J states in a series (i.e.: a vibronic band) then J_max
    should be set as the highest J present in that series, such that the resulting uncertainty estimate is only
    dependent on how far beyond J_max the user is extrapolating to. Generally, scaling based on v and v_max is for use
    with providing uncertainty estimates for calculated or pseudo-experimentally corrected states in vibrational bands
    above the observational data.

    Currently, this method is only intended for use with diatomics, due to the singular vibrational quantum number
    input.

    Args:
        std:      The base uncertainty of a state to be increased.
        j_factor: The value to scale the quadratic J term by.
        v_factor: The value to scale the linear v term by.
        j_val:    The J value of the state.
        j_max:    The maximum value of J that exists in the series that predictions are being made over. If present then
            the uncertainty estimate J term will only scale off how much greater the current J value is that this.
        v_val:    The v value of the state.
        v_max:    The maximum value of v that exists in the series that predictions are being made over. If present then
            the uncertainty estimate v term will only scale off how much greater the current v value is that this.

    Returns:
        A float representing the updated uncertainty of a state with an applied energy prediction.
    """
    j_dif = max(0, int(j_val - j_max)) if j_val != 0 and j_max != 0 else 0
    v_dif = max(0, int(v_val - v_max)) if v_val != 0 and v_max != 0 else 0
    unc_extrapolated = estimate_uncertainty(j_dif, v_dif, j_factor, v_factor)
    return std + unc_extrapolated


def generate_fit_groups(
    states: pd.DataFrame,
    fit_qn_list: t.List[str],
    fit_x_col: str,
    energy_calc_col: str = "energy_calc",
    energy_obs_col: str = "energy_obs",
    energy_dif_col: str = "energy_dif",
    unc_col: str = "unc",
) -> GroupBy:
    shift_table = generate_shift_table(
        states=states,
        shift_table_qn_cols=fit_qn_list + [fit_x_col],
        energy_calc_col=energy_calc_col,
        energy_obs_col=energy_obs_col,
        energy_dif_col=energy_dif_col,
        unc_col=unc_col,
    )
    return shift_table.groupby(by=fit_qn_list, as_index=False)


def predict_shifts(
    states_matched: pd.DataFrame,
    states_header: ExoMolStatesHeader,
    fit_qn_list: t.List[str],
    j_segment_threshold_size: int = 14,
    show_plot: bool = False,
    plot_states: t.List[str] = None,
    energy_calc_col: str = "energy_calc",
    energy_obs_col: str = "energy_obs",
    energy_dif_col: str = "energy_dif",
    executor_type: ExecutorType = ExecutorType.THREADS,
    n_workers: int = 8,
) -> pd.DataFrame:
    """
    Calculates predictions for the obs.-calc. energy differences of levels in an arbitrary grouping, for which some
    observational data exists. The arbitrary grouping is based on the quantum numbers passed to fit_qn_list and as such
    can be used to predict energy shifts for groupings such as vibronic bands or spin-orbit components within vibronic
    bands (i.e.: by passing ["state", "v"] or ["state", "v", "Omega"] respectively).

    Predicted energy shifts are calculated as a function of the rigorous quantum number specified in the
    :func:`~linelisttools.states.ExoMolStatesHeader` object corresponding to the states file DataFrame.

    The original, marvel and dif energy columns are intended to come from :func:`linelisttools.states.match_levels`,
    based on the name of the energy column and suffixes passed to that method. Defaults to the same default values if
    none are passed.


    Args:
        states_matched:           The matched Marvel/calculated states from which the obs.-calc. values to fit to are
            derived.
        states_header:            The ExoMolStatesHeader object containing the column mappings for the states file.
        fit_qn_list:              The list of arbitrary quantum numbers to group the obs.-calc. trends on for fitting.
            Generally should be the same as those used to generate the shift table. All quantum numbers must exist as
             columns within the shift_table and levels_matched DataFrames.
        j_segment_threshold_size: The minimum number of J data-points that must be present in a given segment to fit to.
            The segments that obs.-calc. predictions are fit to will increase in size if multiple sets of missing data
            exist within an array of sequential J values of length equal to this argument.
        show_plot:                Determines whether plots of the input and fitted data are shown.
        plot_states:              The text labels indicating which states plots should be shown for, when show_plot is
            True.
        energy_calc_col:          The string column name for the calculated energy column in states_matched.
        energy_obs_col:           The string column name for the observed energy column in states_matched
        energy_dif_col:           The string column name for the energy difference column in states_matched
        executor_type:            Determines whether the fitting will be carried out with multiple threads or processes.
            Defaults to multithreading.
        n_workers:                The number of threads/processes to concurrently execute for the fitting.

    Returns:
        Outputs the states_matched DataFrame with updated interpolated and extrapolated energy shifts in the series
        defined by fit_qn_list for which Marvel data exists.
    """

    shift_predictions = []
    extrapolate_j_shifts = []

    worker = functools.partial(
        fit_predictions,
        j_segment_threshold_size,
        states_header.get_rigorous_qn(),
    )
    shift_groups = generate_fit_groups(
        states=states_matched,
        fit_qn_list=fit_qn_list,
        fit_x_col=states_header.get_rigorous_qn(),
        energy_calc_col=energy_calc_col,
        energy_obs_col=energy_obs_col,
        energy_dif_col=energy_dif_col,
        unc_col=states_header.unc,
    )

    with executor_type.value(max_workers=n_workers) as e:
        for result in tqdm.tqdm(
            e.map(worker, yield_grouped_data(shift_groups)), total=len(shift_groups)
        ):
            shift_predictions.append(result[0])
            extrapolate_j_shifts.append(result[1])

    if show_plot:
        for fit_idx, fitted_group_data in enumerate(yield_grouped_data(shift_groups)):
            # Plot if shifts were calculated and either no plot_states specified or fitted state in plot_states.
            if len(shift_predictions[fit_idx]) != 0 and (
                plot_states is None
                or fitted_group_data[0][fit_qn_list.index("state")] in plot_states
            ):
                plt.scatter(
                    fitted_group_data[1][states_header.get_rigorous_qn()],
                    fitted_group_data[1]["energy_dif_mean"],
                    marker="x",
                    linewidth=0.5,
                    facecolors=get_vibrant_colors(1),
                    label=" ".join(str(qn) for qn in fitted_group_data[0]),
                    zorder=1,
                )
                plt.scatter(
                    [
                        prediction[len(fitted_group_data[0])]
                        for prediction in shift_predictions[fit_idx]
                    ],
                    [
                        prediction[len(fitted_group_data[0]) + 1]
                        for prediction in shift_predictions[fit_idx]
                    ],
                    marker="^",
                    linewidth=0.5,
                    edgecolors="#000000",
                    facecolors="none",
                    label=f"{' '.join(str(qn) for qn in fitted_group_data[0])} FIT",
                    zorder=2,
                )
                plt.legend(loc="best", prop={"size": 10})
                plt.xlabel(xlabel=states_header.get_rigorous_qn())
                plt.ylabel(ylabel=r"Obs.-Calc. (cm-1)")
                plt.tight_layout()
                plt.show()

    shift_predictions = [item for items in shift_predictions for item in items]
    extrapolate_j_shifts = [item for items in extrapolate_j_shifts for item in items]

    # Update energies with shift predictions:
    pe_fit_shifts = pd.DataFrame(
        data=shift_predictions,
        columns=fit_qn_list
        + [states_header.get_rigorous_qn(), "pe_fit_energy_shift", "pe_fit_unc"],
    )
    # pe_fit_shifts[fit_qn_list] = pe_fit_shifts["fit_qn"].str.split("|", len(fit_qn_list), expand=True)
    # del pe_fit_shifts["fit_qn"]

    qn_merge_cols = fit_qn_list + [states_header.get_rigorous_qn()]
    states_matched = states_matched.merge(
        pe_fit_shifts, left_on=qn_merge_cols, right_on=qn_merge_cols, how="left"
    )
    states_matched.loc[
        (states_matched["energy_final"].isna())
        & (~states_matched["pe_fit_energy_shift"].isna())
        & (~states_matched["energy_calc"].isna()),
        states_header.source_tag,
    ] = SourceTag.PS_LINEAR_REGRESSION
    states_matched[states_header.unc] = np.where(
        states_matched[states_header.source_tag] == SourceTag.PS_LINEAR_REGRESSION,
        states_matched["pe_fit_unc"],
        states_matched[states_header.unc],
    )
    states_matched["energy_final"] = np.where(
        states_matched[states_header.source_tag] == SourceTag.PS_LINEAR_REGRESSION,
        states_matched["energy_calc"] + states_matched["pe_fit_energy_shift"],
        states_matched["energy_final"],
    )
    del states_matched["pe_fit_energy_shift"]
    del states_matched["pe_fit_unc"]
    print(
        "PS_2: \n",
        states_matched.loc[
            states_matched[states_header.source_tag] == SourceTag.PS_LINEAR_REGRESSION
        ],
    )

    # Update energies with higher-J shift extrapolations:
    j_max_col = states_header.get_rigorous_qn() + "_max"
    pe_extrapolate_shifts = pd.DataFrame(
        data=extrapolate_j_shifts,
        columns=fit_qn_list
        + [
            j_max_col,
            "pe_extrapolate_energy_shift",
            "pe_extrapolate_energy_shift_std",
        ],
    )
    # pe_extrapolate_shifts[fit_qn_list] = pe_extrapolate_shifts["fit_qn"].str.split("|", len(fit_qn_list), expand=True)
    # del pe_extrapolate_shifts["fit_qn"]

    states_matched = states_matched.merge(
        pe_extrapolate_shifts, left_on=fit_qn_list, right_on=fit_qn_list, how="left"
    )
    states_matched.loc[
        (states_matched["energy_final"].isna())
        & (~states_matched[j_max_col].isna())
        & (~states_matched["energy_calc"].isna())
        & (states_matched[states_header.get_rigorous_qn()] > states_matched[j_max_col]),
        states_header.source_tag,
    ] = SourceTag.PS_EXTRAPOLATION
    # Scale unc based on j over j_max.
    # states_matched['unc'] = states_matched.apply(
    #     lambda x: scale_uncertainty(std=x['pe_extrapolate_energy_shift_std'], std_scale=2, j_val=x['j'],
    #                                 j_max=x['j_max'], j_scale=0.05)
    #     if math.isnan(x['energy_final']) and not math.isnan(x['energy_calc']) and not math.isnan(x['j_max'])
    #        and x['j'] > x['j_max'] else x['unc'], axis=1)
    states_matched[states_header.unc] = states_matched.apply(
        lambda x: set_predicted_unc(
            std=x["pe_extrapolate_energy_shift_std"],
            j_factor=0.0001,
            v_factor=0.05,
            j_val=x[states_header.get_rigorous_qn()],
            j_max=x[j_max_col],
        )
        if x[states_header.source_tag] == SourceTag.PS_EXTRAPOLATION
        else x[states_header.unc],
        axis=1,
    )
    states_matched["energy_final"] = np.where(
        states_matched[states_header.source_tag] == SourceTag.PS_EXTRAPOLATION,
        states_matched["energy_calc"] + states_matched["pe_extrapolate_energy_shift"],
        states_matched["energy_final"],
    )
    del states_matched[j_max_col]
    del states_matched["pe_extrapolate_energy_shift"]
    del states_matched["pe_extrapolate_energy_shift_std"]
    print(
        "PS_3: \n",
        states_matched.loc[
            states_matched[states_header.source_tag] == SourceTag.PS_EXTRAPOLATION
        ],
    )

    # UNNECESSARY IF NOT OUTPUTTING THESE DATAFRAMES.
    # Add shift predictions to shift table.
    # What to do when unc = NaN?
    # shift_table_full_j = shift_table.copy()
    # shift_table_full_j = shift_table_full_j.rename(
    #     columns={
    #         "energy_dif_mean": "energy_shift",
    #         "energy_dif_std": "energy_shift_unc",
    #     }
    # )
    # pe_fit_shifts = pe_fit_shifts.rename(
    #     columns={
    #         "pe_fit_energy_shift": "energy_shift",
    #         "pe_fit_unc": "energy_shift_unc",
    #     }
    # )
    # shift_table_full_j = shift_table_full_j.append(pe_fit_shifts)
    # shift_table_full_j = shift_table_full_j.sort_values(
    #     by=["state", "v", "Omega", j_col], ascending=[1, 1, 1, 1]
    # )
    #
    # pe_extrapolate_shifts = pe_extrapolate_shifts.rename(
    #     columns={
    #         "pe_extrapolate_energy_shift": "extrapolation_energy_shift",
    #         "pe_extrapolate_energy_shift_std": "extrapolation_energy_unc",
    #     }
    # )
    return states_matched


def fit_predictions(
    j_segment_threshold_size: int,
    j_col: str,
    grouped_data: t.Tuple[t.List[str], pd.DataFrame],
) -> t.Tuple[t.List[tuple], t.List[tuple]]:
    """

    Args:
        j_segment_threshold_size:
        j_col:       The String column name for the J column.
        grouped_data:

    Returns:

    """
    # TODO: Why does internal plotting not work well with multithreading - is this a IDE thing?
    shift_predictions = []
    extrapolate_j_shifts = []
    fit_qn_list = tuple(grouped_data[0])
    df_group = grouped_data[1]

    j_max = df_group[j_col].max()

    j_is_integer = (2 * j_max) % 2 == 0

    if j_is_integer:
        min_allowed_j = 0
        j_coverage_to_max = range(0, j_max + 1)
    else:
        min_allowed_j = 0.5
        j_coverage_to_max = [x / 2 for x in range(1, int(j_max * 2), 2)]
    missing_j = np.array(np.setdiff1d(j_coverage_to_max, df_group[j_col]))
    if len(missing_j) > 0:
        delta_missing_j = np.abs(missing_j[1:] - missing_j[:-1])
        split_idx = np.where(delta_missing_j >= j_segment_threshold_size)[0] + 1
        missing_j_segments = np.array_split(missing_j, split_idx)

        for j_segment in missing_j_segments:
            # If the segment is entirely within the slice then the wing size is half the threshold.
            if (
                min(j_segment) > df_group[j_col].min()
                and max(j_segment) < df_group[j_col].max()
            ):
                segment_wing_size = int(j_segment_threshold_size / 2)
            else:
                segment_wing_size = int(j_segment_threshold_size)

            segment_j_lower_limit = max(
                min_allowed_j, min(j_segment) - segment_wing_size
            )
            segment_j_upper_limit = max(
                j_segment_threshold_size + min_allowed_j,
                max(j_segment) + segment_wing_size,
            )
            if j_is_integer:
                segment_j_coverage = range(
                    int(segment_j_lower_limit), int(segment_j_upper_limit) + 1
                )
            else:
                segment_j_coverage = [
                    x / 2
                    for x in range(
                        int(segment_j_lower_limit * 2),
                        int((segment_j_upper_limit + 1) * 2),
                        2,
                    )
                ]
            # Huber regression - resist outliers.
            x_scaler, y_scaler = StandardScaler(), StandardScaler()
            x_train = x_scaler.fit_transform(
                np.array(
                    df_group.loc[
                        (df_group[j_col] >= segment_j_lower_limit)
                        & (df_group[j_col] <= segment_j_upper_limit),
                        j_col,
                    ]
                )[..., None]
            )
            y_train = y_scaler.fit_transform(
                np.array(
                    df_group.loc[
                        (df_group[j_col] >= segment_j_lower_limit)
                        & (df_group[j_col] <= segment_j_upper_limit),
                        "energy_dif_mean",
                    ]
                )[..., None]
            )
            model = HuberRegressor(epsilon=1.35, max_iter=500)
            model.fit(x_train, y_train.ravel())

            model_segment_predictions = model.predict(
                x_scaler.transform(j_segment[..., None])
            ).reshape(-1, 1)
            segment_predictions = y_scaler.inverse_transform(
                model_segment_predictions
            ).ravel()
            # Find the J we are making predictions for that we also have known energy_dif values for.
            segment_j_in_slice = np.array(
                np.intersect1d(segment_j_coverage, df_group[j_col])
            )
            segment_j_outliers = np.array(
                df_group.loc[
                    (df_group[j_col].isin(segment_j_in_slice))
                    & (
                        abs(
                            df_group["energy_dif_mean"]
                            - df_group["energy_dif_mean"].mean()
                        )
                        > (2 * df_group["energy_dif_mean"].std())
                    ),
                    j_col,
                ]
            )
            segment_j_in_slice_no_outliers = np.setdiff1d(
                segment_j_in_slice, segment_j_outliers
            )

            # standard_error_of_estimate = mean_squared_error(
            #     shift_table_slice.loc[shift_table_slice['j'].isin(segment_j_in_slice_no_outliers),
            #                           'energy_dif_mean'],
            #     y_scaler.inverse_transform(model.predict(x_scaler.transform(
            #         segment_j_in_slice_no_outliers[..., None]))),
            #     squared=False)
            real_energy = np.array(
                df_group.loc[
                    df_group[j_col].isin(segment_j_in_slice_no_outliers),
                    "energy_dif_mean",
                ]
            )
            mode_present_predictions = model.predict(
                x_scaler.transform(segment_j_in_slice_no_outliers[..., None])
            ).reshape(-1, 1)
            present_predictions = y_scaler.inverse_transform(
                mode_present_predictions
            ).ravel()

            dif_energy = real_energy - present_predictions
            dif_squared_energy = dif_energy**2
            std_energy = np.sqrt(sum(dif_squared_energy) / len(dif_energy))

            for entry in [
                fit_qn_list + (j, prediction, std_energy)
                for j, prediction in zip(j_segment, list(segment_predictions))
            ]:
                shift_predictions.append(entry)

    # Now take the mean of the last 10 points within 2std and take the mean shift there and apply it to
    # all later trans.
    shift_table_final_rows = (
        df_group.loc[
            abs(df_group["energy_dif_mean"] - df_group["energy_dif_mean"].mean())
            < (2 * df_group["energy_dif_mean"].std())
        ]
        .sort_values(by=j_col, ascending=[1])
        .tail(j_segment_threshold_size)
    )
    extrapolate_j_shift_mean = shift_table_final_rows["energy_dif_mean"].mean()
    extrapolate_j_shift_std = shift_table_final_rows["energy_dif_mean"].std()
    extrapolate_j_shifts.append(
        fit_qn_list + (j_max, extrapolate_j_shift_mean, extrapolate_j_shift_std)
    )

    return shift_predictions, extrapolate_j_shifts


def set_calc_states(
    states: pd.DataFrame,
    states_header: ExoMolStatesHeader,
    unc_j_factor: float = 0.0001,
    unc_v_factor: float = 0.05,
    energy_final_col: str = "energy_final",
    energy_calc_col: str = "energy_calc",
    energy_dif_col: str = "energy_dif",
) -> pd.DataFrame:
    """
    Updates all states with no assigned source tag to Calculated and estimates their uncertainty. This is done by
    starting with an initial base calculated uncertainty equal to twice the standard deviation of the absolute values of
    the known obs.-calc. differences, plus a quantum number dependent uncertainty extraploation calculated using
    :func:`linelisttools.states.estimate_uncertainty`.

    This currently only works for diatomic state files given the v scaling in the uncertainty estimator.

    Args:
        states:           A DataFrame containing all states, those of which without a source_tag set will be updated to
            calculated.
        states_header:    The ExoMolStatesHeader object containing the column mappings for the states file.
        unc_j_factor:     The uncertainty scale factor for the J term.
        unc_v_factor:     The uncertainty scale factor for the v term.
        energy_final_col: The string label for the final energy column in states.
        energy_calc_col:  The string label for the calculated energy column in states.
        energy_dif_col:   The string label for the energy difference column in states.

    Returns:
        A DataFrame where all input states without a source tag assigned have been set to Calculated and had their
            uncertainty estimated.
    """
    states[states_header.source_tag] = np.where(
        states[states_header.source_tag].isna(),
        SourceTag.CALCULATED,
        states[states_header.source_tag],
    )
    states[energy_final_col] = np.where(
        states[states_header.source_tag] == SourceTag.CALCULATED,
        states[energy_calc_col],
        states[energy_final_col],
    )
    states[states_header.unc] = states.apply(
        lambda x: (2 * states[energy_dif_col].abs().std())
        + estimate_uncertainty(
            x[states_header.get_rigorous_qn()],
            x[states_header.vibrational_qn],
            unc_j_factor,
            unc_v_factor,
        )
        if x[states_header.source_tag] == SourceTag.CALCULATED
        else x[states_header.unc],
        axis=1,
    )

    return states


def shift_parity_pairs(
    states: pd.DataFrame,
    states_header: ExoMolStatesHeader,
    shift_table_qn_cols: t.List[str],
    energy_calc_col: str = "energy_calc",
    energy_obs_col: str = "energy_obs",
    energy_dif_col: str = "energy_dif",
    energy_final_col: str = "energy_final",
) -> pd.DataFrame:
    """
    Updates levels that have not had their source_tag set but have a level with equivalent quantum numbers in the Marvel
    data. This is determined through merging those states without a source_tag on the shift_table, which contains a list
    of the unique combinations of quantum numbers in the Marvel data. This assumes the shift table/level matching was
    performed over a full set of quantum numbers; if matching was done on a partial set, a shift table should be
    manually created for the set of quantum numbers used to determine a parity pair.

    Args:
        states:              A DataFrame containing the states to search for parity pairs in.
        states_header:       The ExoMolStatesHeader object containing the column mappings for the states file.
        shift_table_qn_cols: The quantum number columns in states that should be grouped on to find the parity shift for
            each J; this list should not include "J".
        energy_calc_col:     The string label for the calculated energy column in states.
        energy_obs_col:      The string label for the observed energy column in states.
        energy_dif_col:      The string label for the energy difference column in states.
        energy_final_col:    The string label for the final energy column in states.

    Returns:
        The states DataFrame updated with the mean energy shift from the shift table applied to any parity pair
            counterparts that were not updated with Marvel data.
    """
    shift_table = generate_shift_table(
        states=states,
        shift_table_qn_cols=shift_table_qn_cols,
        energy_calc_col=energy_calc_col,
        energy_obs_col=energy_obs_col,
        energy_dif_col=energy_dif_col,
        unc_col=states_header.unc,
    )
    # energy_dif_mean and energy_dif_unc are implicit column names of the shift table: other columns are the quantum
    # numbers it was grouped on.
    energy_dif_mean_col = "energy_dif_mean"
    energy_dif_unc_col = "energy_dif_unc"
    # shift_table_qn_cols = [
    #     qn
    #     for qn in shift_table.columns
    #     if qn not in (energy_dif_mean_col, energy_dif_unc_col)
    # ]

    # Inner merge here gets us only the states that have matching quantum numbers to an entry in the shift table but has
    # not had its source_tag set, i.e.: those for which we have Marvel data for another level with the same quantum
    # numbers, excluding parity.
    states_missing_parity_pairs = states.loc[
        states[states_header.source_tag].isna()
    ].merge(shift_table, on=shift_table_qn_cols, how="inner")
    if len(states_missing_parity_pairs) == 0:
        return states
    else:
        # Apply shift table mean energy difference to calculated energy.
        states_missing_parity_pairs[
            energy_final_col
        ] = states_missing_parity_pairs.apply(
            lambda x: x[energy_calc_col] + x[energy_dif_mean_col], axis=1
        )

        # Left merge parity pair shifts onto states to keep all states and give shift data where needed.
        states = states.merge(
            states_missing_parity_pairs[
                [states_header.state_id, energy_final_col, energy_dif_unc_col]
            ],
            on=[states_header.state_id],
            how="left",
            suffixes=("", "_temp"),
        )
        states.loc[
            states[states_header.state_id].isin(
                states_missing_parity_pairs[states_header.state_id]
            ),
            states_header.source_tag,
        ] = SourceTag.PS_PARITY_PAIR
        # Take the energy_final_temp from the merged DataFrame as energy_final where it exists, and the energy_dif_unc as
        # the unc for these rows.
        energy_final_temp_col = energy_final_col + "_temp"
        # Change to find new merge cols based on "PS_1" source_tag?
        states[energy_final_col] = np.where(
            states[energy_final_temp_col].isna(),
            states[energy_final_col],
            states[energy_final_temp_col],
        )
        states[states_header.unc] = np.where(
            states[energy_final_temp_col].isna(),
            states[states_header.unc],
            states[energy_dif_unc_col],
        )
        # Drop any temp columns, including the now useless (as it has been copied over into unc) energy_dif_unc column.
        states = states.drop(
            list(states.filter(regex="_temp")) + [energy_dif_unc_col], axis=1
        )
        # states = states.drop(energy_dif_unc_col, axis=1)
        # Reorder on id for convenience.
        states = states.sort_values(by=[states_header.state_id], ascending=True)
        return states
