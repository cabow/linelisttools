from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from enum import Enum

from pandas.core.groupby import GroupBy


class ExecutorType(Enum):
    THREADS = ThreadPoolExecutor
    PROCESS = ProcessPoolExecutor


def yield_grouped_data(grouped_data: GroupBy):
    for group_name, df_group in grouped_data:
        yield group_name, df_group
