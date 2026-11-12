"""Evidence-critical tests fail, rather than skip, when a stored artefact is missing:
a green suite must mean every evidence gate actually ran."""

from __future__ import annotations

from pathlib import Path

import pytest


def require_artefact(path, *, regen_command: str, request,
                     regen_fn=None, cost_note: str = "") -> None:
    path = Path(path)
    if bool(request.config.getoption("--regen-evidence", False)):
        if regen_fn is None:
            pytest.fail(f"--regen-evidence was set but no in-test regeneration is wired "
                        f"for {path}. Regenerate manually: {regen_command}")
        regen_fn()
    if not path.exists():
        pytest.fail(f"Evidence-critical artefact missing: {path}\n"
                    f"Regenerate with: {regen_command}"
                    f"{(' (' + cost_note + ')') if cost_note else ''}")
