"""Original provenance counter condensation and semantic categories.

Use streaming.py for merged repeated-window extraction; no last-wins streamer
is included in this training package.
"""
from __future__ import annotations
import math


PATH_PREFIXES = ["/etc", "/tmp", "/home", "/usr", "/var", "/proc", "/dev", "/lib",
                 "/bin", "/sbin", "/opt", "/root", "/run", "/sys", "/mnt", "/srv"]


EDGE_TYPES = ["WasTriggeredBy", "Used", "WasGeneratedBy", "WasDerivedFrom", "WasControlledBy"]


def _attrs(blob: str) -> dict:
    """Parse '{k:v,k:v,}' tolerantly."""
    out = {}
    for part in blob.strip().lstrip("{").rstrip("}").split(","):
        if not part:
            continue
        k, _, v = part.partition(":")
        if _:
            out[k.strip()] = v
    return out


def _path_bucket(p: str) -> int:
    for i, pre in enumerate(PATH_PREFIXES):
        if p.startswith(pre):
            return i
    return len(PATH_PREFIXES)


def condense_prov(acc: dict, w: int = 60) -> dict:
    import numpy as np
    ts = np.asarray(acc["ts"]); ts.sort()
    dt = np.diff(ts) if len(ts) > 1 else np.array([0.0])
    od = np.asarray(list(acc["outdeg"].values()) or [0], float)
    idg = np.asarray(list(acc["indeg"].values()) or [0], float)
    nv = max(len(acc["verts"]), 1)
    tot_ops = sum(acc["ops"].values()) or 1

    def ent(c):
        s = sum(c.values()) or 1
        return float(-sum((v / s) * math.log2(v / s) for v in c.values() if v))

    return dict(
        n_edges=acc["edges"], n_vertices=len(acc["verts"]),
        edges_per_vertex=acc["edges"] / nv,
        new_vertex_rate=acc["new_v"] / max(2 * acc["edges"], 1),
        n_new_vertices=acc["new_v"],
        proc_frac=acc["proc_v"] / max(acc["proc_v"] + acc["art_v"], 1),
        root_frac=acc["root_edges"] / max(acc["proc_v"], 1),
        outdeg_mean=float(od.mean()), outdeg_max=float(od.max()), outdeg_std=float(od.std()),
        indeg_mean=float(idg.mean()), indeg_max=float(idg.max()), indeg_std=float(idg.std()),
        op_entropy=ent(acc["ops"]), name_entropy=ent(acc["pnames"]),
        bucket_entropy=ent(acc["pbuckets"]), subtype_entropy=ent(acc["subtypes"]),
        n_distinct_ops=len(acc["ops"]), n_distinct_names=len(acc["pnames"]),
        dt_mean=float(dt.mean()), dt_std=float(dt.std()), dt_max=float(dt.max()),
        ops=dict(acc["ops"]), etypes=dict(acc["etypes"]),
        pnames=dict(acc["pnames"].most_common(40)),
        pbuckets={str(k): v for k, v in acc["pbuckets"].items()},
        subtypes=dict(acc["subtypes"]),
    )
