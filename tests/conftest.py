"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from twoasset.runlog import RunLog


@pytest.fixture(scope="session")
def runlog(request):
    log = RunLog(module="pytest", kind="test")
    yield log
    log.close()


@pytest.fixture(autouse=True)
def _runlog_test_id(request, runlog):
    """Stamp the current test's nodeid onto every row it logs, without requiring each
    call site to pass `test_id=` explicitly."""
    original_result = runlog.result

    def _result_with_test_id(*args, **kwargs):
        kwargs.setdefault("test_id", request.node.nodeid)
        return original_result(*args, **kwargs)

    runlog.result = _result_with_test_id
    yield
    runlog.result = original_result
