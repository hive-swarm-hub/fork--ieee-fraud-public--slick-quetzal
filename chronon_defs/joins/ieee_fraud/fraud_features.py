"""
Chronon Join: Combines feature GroupBys into a training/serving feature set.

Left side = transaction events (defines the entity keys and timestamps).
Right side = GroupBy feature sources.
"""

from ai.chronon.api.ttypes import Source, EventSource
from ai.chronon.query import Query, select
from ai.chronon.join import Join, JoinPart
from group_bys.ieee_fraud.transaction_features import v1 as txn_features_v1

# Left source: the transaction events we want to enrich with features
left_source = Source(
    events=EventSource(
        table="ieee_fraud.transactions",
        query=Query(
            selects=select(
                "card1",
                "TransactionID",
                "TransactionAmt",
                "TransactionDT",
                "isFraud",
            ),
            time_column="ts",
        ),
    )
)

v1 = Join(
    left=left_source,
    right_parts=[
        JoinPart(group_by=txn_features_v1),
    ],
    online=True,
)
