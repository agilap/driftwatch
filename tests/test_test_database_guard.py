"""Pytest must refuse to start without a dedicated test database."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# Nothing listens here, so a guard failure cannot reach any real database.
UNREACHABLE = "postgresql+asyncpg://driftwatch:driftwatch@127.0.0.1:1/{}"


@pytest.mark.parametrize(
    "test_url, message",
    [
        (None, "TEST_DATABASE_URL is not set"),
        (UNREACHABLE.format("driftwatch"), "must end in _test"),
    ],
)
def test_pytest_exits_before_running_anything(test_url, message) -> None:
    env = {**os.environ, "DATABASE_URL": UNREACHABLE.format("driftwatch")}
    env.pop("TEST_DATABASE_URL")
    if test_url:
        env["TEST_DATABASE_URL"] = test_url
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_health.py", "-q"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0
    assert message in result.stdout + result.stderr
    assert "passed" not in result.stdout
