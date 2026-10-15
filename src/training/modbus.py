"""Original Modbus counter condensation; merged extraction is in streaming.py.

The old flush-on-key-change extractor is deliberately not exposed here.
"""
from __future__ import annotations
import math


READ_FC, WRITE_FC = {1, 2, 3, 4, 7, 23}, {5, 6, 15, 16, 22, 23}


PLC_PORT = 502


def condense(acc: dict, w: int = 60) -> dict:
    """Collapse a window's raw counters into a compact, JSON-serialisable record."""
    import numpy as np
    ts = np.asarray(acc["ts"]); ts.sort()
    dt = np.diff(ts) if len(ts) > 1 else np.array([0.0])
    wc = np.asarray(acc["wc"]) if acc["wc"] else np.array([0])
    tid = np.asarray(acc["tid"]) if acc["tid"] else np.array([0])
    sub = np.bincount(((ts - ts[0]) if len(ts) else np.array([0])).astype(int).clip(0, w - 1),
                      minlength=w) if len(ts) else np.zeros(w)
    fcn = sum(acc["fc"].values()) or 1
    return dict(
        n=acc["n"], req=acc["req"], resp=acc["resp"],
        n_client_ports=len(acc["cports"]),
        n_units=len(acc["unit"]),
        fc=dict(acc["fc"]), reg=dict(acc["reg"]),
        n_distinct_reg=len(acc["reg"]),
        reg_min=(acc["reg_min"] if acc["reg_min"] < math.inf else -1),
        reg_max=(acc["reg_max"] if acc["reg_max"] > -math.inf else -1),
        reg_entropy=float(-sum((c / sum(acc["reg"].values())) * math.log2(c / sum(acc["reg"].values()))
                               for c in acc["reg"].values())) if acc["reg"] else 0.0,
        read_frac=float(sum(c for k, c in acc["fc"].items() if k in READ_FC) / fcn),
        write_frac=float(sum(c for k, c in acc["fc"].items() if k in WRITE_FC) / fcn),
        wc_mean=float(wc.mean()), wc_max=float(wc.max()), wc_std=float(wc.std()),
        dt_mean=float(dt.mean()), dt_std=float(dt.std()), dt_med=float(np.median(dt)),
        dt_p95=float(np.percentile(dt, 95)), dt_max=float(dt.max()),
        burstiness=float((dt.std() - dt.mean()) / (dt.std() + dt.mean() + 1e-12)),
        fano=float(sub.var() / (sub.mean() + 1e-12)),
        subsec_active=float((sub > 0).mean()),
        tid_reset_frac=float((np.diff(tid) < 0).mean()) if len(tid) > 1 else 0.0,
        tid_span=float(tid.max() - tid.min()),
    )
