"""
Chronon GroupBy: Transaction-level aggregations for fraud detection.

Aggregates transaction data by card1 (card identifier) over rolling windows.
This is the baseline — add more aggregations, keys, and windows to improve AUC-PR.
"""

from ai.chronon.api.ttypes import Source, EventSource
from ai.chronon.query import Query, select
from ai.chronon.group_by import (
    GroupBy,
    Aggregation,
    Operation,
    Window,
    TimeUnit,
)

source = Source(
    events=EventSource(
        table="ieee_fraud.transactions",
        query=Query(
            selects=select(
                "card1",
                "TransactionAmt",
                "isFraud",
            ),
            time_column="ts",
        ),
    )
)

window_sizes = [
    Window(length=3, timeUnit=TimeUnit.DAYS),
    Window(length=7, timeUnit=TimeUnit.DAYS),
    Window(length=14, timeUnit=TimeUnit.DAYS),
    Window(length=30, timeUnit=TimeUnit.DAYS),
]

v1 = GroupBy(
    sources=[source],
    keys=["card1"],
    online=True,
    aggregations=[
        Aggregation(
            input_column="TransactionAmt",
            operation=Operation.SUM,
            windows=window_sizes,
        ),
        Aggregation(
            input_column="TransactionAmt",
            operation=Operation.COUNT,
            windows=window_sizes,
        ),
        Aggregation(
            input_column="TransactionAmt",
            operation=Operation.AVERAGE,
            windows=window_sizes,
        ),
    ],
)
