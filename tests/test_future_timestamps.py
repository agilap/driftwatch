from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.schemas.ingest import SnapshotPayload


@pytest.mark.parametrize(
    "timestamp",
    [
        datetime.now(timezone.utc) + timedelta(days=2),
        datetime(9999, 12, 31, 23, 59, 59, tzinfo=timezone.utc),
    ],
)
def test_future_snapshot_timestamp_rejected(timestamp: datetime) -> None:
    with pytest.raises(ValidationError, match="more than one day"):
        SnapshotPayload(model_id=uuid4(), timestamp=timestamp, features={"x": [1.0]})


@pytest.mark.parametrize("offset", [timedelta(hours=12), timedelta(days=-1)])
def test_snapshot_timestamp_within_limit_accepted(offset: timedelta) -> None:
    timestamp = datetime.now(timezone.utc) + offset
    payload = SnapshotPayload(
        model_id=uuid4(), timestamp=timestamp, features={"x": [1.0]}
    )
    assert payload.timestamp == timestamp
