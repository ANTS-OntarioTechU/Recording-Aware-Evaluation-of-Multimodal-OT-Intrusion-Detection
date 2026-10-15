"""Leakage guard for candidate feature matrices.

The guard rejects any column that could let a detector recover recording identity, ground truth,
or a raw identifier instead of learning attack behaviour. It runs before every model-fitting
section in the notebooks and emits a readable allow/reject report.

Two categories are rejected for different reasons:

* ground-truth and partition fields leak the answer directly;
* raw identifiers (addresses, ports, paths, PIDs, hashes, timestamps) differ systematically
  between recordings, so a model can use them as a recording fingerprint.

Availability indicators are rejected because the presence or absence of a view is itself a
recording property: in this dataset M2 is present in 77.6% of windows versus 94.9% for M1 and M3.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import pandas as pd

# Token-boundary patterns. `_` is treated as a separator, not a word character: a plain `\\bip\\b`
# does NOT fire on `dst_ip`, and `\\bpid\\b` does not fire on `proc_pid`, which are precisely the
# columns that must be rejected. `_T(x)` builds a boundary that respects underscores.
def _T(token: str) -> str:
    """Match ``token`` when delimited by string start/end or a non-alphanumeric character."""
    return rf"(?:^|[^a-z0-9]){token}(?:[^a-z0-9]|$)"


REJECT_PATTERNS: dict[str, str] = {
    "ground_truth": r"attack|attacker|label|ground_?truth|" + _T("target") + r"|" + _T("phase"),
    "state_disclosure": r"old_value|new_value",
    "partition": _T("scenario") + r"|" + _T("split") + r"|" + _T("file") + r"|filename|" + _T("campaign"),
    "network_identifier": (_T("ip") + r"|ipv4|ipv6|addr|" + _T("mac") + r"|port|"
                           r"trans(action)?_?id|" + _T("tid")),
    "host_identifier": (_T("pid") + r"|process_?id|" + _T("ppid") + r"|event_?id|hash|"
                        + _T("exe") + r"|executable|cmdline|command_?line|" + _T("path")),
    "time": r"time|timestamp|epoch|" + _T("date") + r"|window_(start|end)",
    "availability": r"__present$|_present$|availability|missing(ness)?",
}

# Exact column names reviewed and permitted despite matching a pattern. Semantic categories, not
# identifiers. Extend only with a written justification.
REVIEWED_EXEMPTIONS: dict[str, str] = {
    # Counts and dispersion statistics DERIVED from identifiers. The extractors never emit the
    # identifier value itself; these summarise it. Each was reviewed against the released feature
    # list actually used in the submitted paper.
    "m3_n_client_ports": (
        "Count of distinct ephemeral client ports in the window, not a port value. The port "
        "numbers themselves are discarded at parse time."
    ),
    "m3_tid_reset_frac": (
        "Fraction of consecutive Modbus transaction IDs that decrease, i.e. a wrap/reset rate. "
        "No transaction ID value is retained."
    ),
    "m3_tid_span": (
        "Range (max - min) of Modbus transaction IDs in the window; a dispersion statistic, not "
        "an identifier."
    ),
    "m1_subtypes_file": (
        "SPADE artifact subtype category 'file' (as opposed to socket, pipe, or unknown). A "
        "semantic vertex type, not a filename or partition field."
    ),
    "m1_subtypes_file__frac": "Normalised share of the 'file' artifact subtype; see m1_subtypes_file.",
    "m2_subtypes_file": "Same semantic subtype category on the M2 provenance view.",
    "m2_subtypes_file__frac": "Normalised share of the 'file' artifact subtype on M2.",
    "m2_pnames_sshd-session": (
        "Process NAME from the train-fitted M2 vocabulary; 'sshd-session' is the OpenSSH "
        "per-connection helper binary. The feature is a count of edges touching processes with "
        "that name, not a per-recording session identifier."
    ),
}


@dataclass
class GuardReport:
    """Outcome of a guard run."""

    allowed: list[str] = field(default_factory=list)
    rejected: dict[str, str] = field(default_factory=dict)
    exempted: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.rejected

    def to_text(self, name: str = "feature matrix") -> str:
        """Render a readable report."""
        lines = [f"schema guard — {name}",
                 f"  allowed  : {len(self.allowed)}",
                 f"  rejected : {len(self.rejected)}",
                 f"  exempted : {len(self.exempted)}"]
        for column, reason in sorted(self.rejected.items()):
            lines.append(f"    REJECT  {column}  [{reason}]")
        for column, reason in sorted(self.exempted.items()):
            lines.append(f"    EXEMPT  {column}  ({reason[:60]}…)")
        return "\n".join(lines)


def inspect(columns: Iterable[str]) -> GuardReport:
    """Classify column names without raising.

    Returns a :class:`GuardReport`. Use :func:`enforce` to fail closed.
    """
    report = GuardReport()
    for column in columns:
        name = str(column)
        if name in REVIEWED_EXEMPTIONS:
            report.exempted[name] = REVIEWED_EXEMPTIONS[name]
            report.allowed.append(name)
            continue
        hit = next((category for category, pattern in REJECT_PATTERNS.items()
                    if re.search(pattern, name, re.IGNORECASE)), None)
        if hit:
            report.rejected[name] = hit
        else:
            report.allowed.append(name)
    return report


def enforce(columns: Sequence[str], where: str = "feature matrix") -> GuardReport:
    """Raise ``AssertionError`` if any column is rejected.

    ``where`` is used in the error message so a failure names the offending matrix.
    """
    report = inspect(columns)
    if not report.ok:
        listing = ", ".join(f"{c} [{r}]" for c, r in sorted(report.rejected.items())[:12])
        raise AssertionError(f"schema guard rejected {len(report.rejected)} column(s) in {where}: {listing}")
    return report


def guard_frame(frame: pd.DataFrame, where: str = "feature matrix") -> tuple[pd.DataFrame, GuardReport]:
    """Return the frame restricted to allowed columns, plus the report.

    Raises ``AssertionError`` if any column is rejected, so a caller cannot silently proceed on a
    matrix that still contains a leaking field.
    """
    report = enforce(list(frame.columns), where)
    return frame[report.allowed], report
