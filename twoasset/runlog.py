"""Run log: records results and provenance (code version, library versions, platform)
of each run."""

from __future__ import annotations

import csv
import io
import json
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
VALIDATION_DIR = ROOT / "validation"
MASTER_LEDGER = VALIDATION_DIR / "runlog.csv"

CSV_FIELDS = [
    "timestamp_utc", "run_id", "kind", "module", "test_id", "config", "metric",
    "value", "unit", "sem", "ci_lo", "ci_hi",
    "gate_class", "threshold", "passed",
    "seed", "n_paths", "n_steps", "commit", "extra_json",
]


def _git_commit() -> str:
    """Short git commit hash, or 'unknown'."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def _package_version(name: str) -> str:
    try:
        mod = __import__(name)
        return str(getattr(mod, "__version__", "unknown"))
    except ImportError:
        return "not installed"


def build_provenance(**extra) -> dict:
    """Provenance record: commit, library versions, platform and any extra fields."""
    return {
        "commit": _git_commit(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "numpy": _package_version("numpy"),
        "scipy": _package_version("scipy"),
        "matplotlib": _package_version("matplotlib"),
        "platform": platform.platform(),
        **extra,
    }


def _fmt(value) -> str:
    """Lossless-round-trip formatting: `repr()` for floats, `str()` otherwise; `None`
    -> empty string (the CSV convention for not-applicable)."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    return str(value)


def _run_id(module: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{ts}_{module}_{_git_commit()}"


@dataclass
class RunLog:
    """Collects the results of one run and writes them to validation/runs/ on close."""

    module: str
    kind: str = "study"                 # "study" | "test"
    scratch: bool = False
    run_id: str = field(default="", repr=False)
    _rows: list = field(default_factory=list, repr=False)
    _provenance: dict = field(default_factory=dict, repr=False)
    _closed: bool = field(default=False, repr=False)

    def __post_init__(self):
        if not self.run_id:
            self.run_id = _run_id(self.module)
        self._provenance = build_provenance(run_id=self.run_id,
                                            module=self.module, kind=self.kind)

    def _out_dir(self) -> Path:
        base = VALIDATION_DIR / "scratch" if self.scratch else VALIDATION_DIR
        return base / "runs"

    def _master_ledger_path(self) -> Path:
        return (VALIDATION_DIR / "scratch" / "runlog.csv") if self.scratch \
            else MASTER_LEDGER

    def params(self, **kv) -> None:
        """Record the inputs of the run."""
        import hashlib
        cfg_files = {}
        for k, v in kv.items():
            if isinstance(v, str) and v.endswith((".toml", ".json")):
                p = Path(v)
                if not p.is_absolute():
                    p = ROOT / v
                if p.exists():
                    cfg_files[v] = hashlib.sha256(p.read_bytes()).hexdigest()
        self._provenance.setdefault("params", {}).update(kv)
        if cfg_files:
            self._provenance.setdefault("config_files", {}).update(cfg_files)

    def result(self, metric: str, value, *, config: str = "", unit: str = "",
              sem: Optional[float] = None, ci: Optional[tuple] = None,
              gate_class: str = "", threshold=None, passed: Optional[bool] = None,
              test_id: str = "", seed=None, n_paths=None, n_steps=None,
              **extra) -> dict:
        """Buffer one row."""
        row = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "kind": self.kind,
            "module": self.module,
            "test_id": test_id,
            "config": config,
            "metric": metric,
            "value": _fmt(value),
            "unit": unit,
            "sem": _fmt(sem),
            "ci_lo": _fmt(ci[0]) if ci is not None else "",
            "ci_hi": _fmt(ci[1]) if ci is not None else "",
            "gate_class": gate_class,
            "threshold": _fmt(threshold),
            "passed": _fmt(passed),
            "seed": _fmt(seed),
            "n_paths": _fmt(n_paths),
            "n_steps": _fmt(n_steps),
            "commit": self._provenance["commit"],
            "extra_json": json.dumps(extra, sort_keys=True, default=str) if extra else "",
        }
        self._rows.append(row)
        return row

    def _write_csv(self, path: Path, rows: list, *, append: bool) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = append and not path.exists()
        mode = "a" if append else "w"
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=CSV_FIELDS)
        if (not append) or write_header:
            w.writeheader()
        for r in rows:
            w.writerow(r)
        with open(path, mode, newline="", encoding="utf-8") as f:
            f.write(buf.getvalue())

    def close(self) -> None:
        """Write the run's results and provenance."""
        if self._closed:
            return
        out_dir = self._out_dir()
        self._write_csv(out_dir / f"{self.run_id}.csv", self._rows, append=False)
        self._write_csv(self._master_ledger_path(), self._rows, append=True)
        prov = dict(self._provenance)
        prov["n_rows"] = len(self._rows)
        with open(out_dir / f"{self.run_id}.provenance.json", "w") as f:
            json.dump(prov, f, indent=2, default=str)
        self._closed = True

    def __enter__(self) -> "RunLog":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def assert_gate(log: RunLog, metric: str, value, threshold, *, cmp: str = "<",
                gate_class: str = "S", assert_: bool = True, **ctx) -> bool:
    """Record a checked value and assert that it passes the threshold."""
    ops = {
        "<": lambda a, b: a < b, "<=": lambda a, b: a <= b,
        ">": lambda a, b: a > b, ">=": lambda a, b: a >= b,
        "==": lambda a, b: a == b,
    }
    if cmp not in ops:
        raise ValueError(f"unsupported cmp {cmp!r}; use one of {sorted(ops)}")
    passed = bool(ops[cmp](value, threshold))
    log.result(metric, value, threshold=threshold, gate_class=gate_class,
              passed=passed, **ctx)
    if assert_:
        assert passed, (
            f"gate failed: {metric} = {value!r} not {cmp} {threshold!r} "
            f"({gate_class}-class){' ctx=' + repr(ctx) if ctx else ''}"
        )
    return passed
