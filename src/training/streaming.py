"""Original corrected multi-resolution extraction.

Repeated window keys are merged before condensation, including the M2 fragment
repair. This preserves the recorded implementation; no new sorting convention
or feature definition is introduced.
"""
from __future__ import annotations
import json
import math
from collections import Counter
from .modbus import READ_FC, WRITE_FC, PLC_PORT, condense as m3_condense
from .provenance import _attrs, _path_bucket, condense_prov


WINDOWS = (5, 10, 30, 60)


def _merge_m3(dst, src):
    dst["n"] += src["n"]; dst["req"] += src["req"]; dst["resp"] += src["resp"]
    dst["ts"].extend(src["ts"]); dst["wc"].extend(src["wc"]); dst["tid"].extend(src["tid"])
    dst["cports"] |= src["cports"]
    for f in ("fc", "reg", "unit"):
        dst[f].update(src[f])
    dst["reg_min"] = min(dst["reg_min"], src["reg_min"])
    dst["reg_max"] = max(dst["reg_max"], src["reg_max"])
    return dst


def _merge_prov(dst, src):
    for f in ("edges", "new_v", "root_edges", "proc_v", "art_v"):
        dst[f] += src[f]
    dst["ts"].extend(src["ts"]); dst["verts"] |= src["verts"]
    for f in ("ops", "etypes", "outdeg", "indeg", "pnames", "pbuckets", "subtypes"):
        dst[f].update(src[f])
    return dst


def _m3_new():
    return dict(n=0, req=0, resp=0, fc=Counter(), reg=Counter(), unit=Counter(),
                wc=[], ts=[], cports=set(), tid=[], reg_min=math.inf, reg_max=-math.inf)


def stream_m3_multi(path, windows=WINDOWS, progress_every=5_000_000, log=print):
    """Return {w: {window_index: condensed_record}} in a single pass."""
    out = {w: {} for w in windows}
    cur = {w: None for w in windows}
    acc = {w: None for w in windows}
    n = 0
    dup = {w: 0 for w in windows}
    with open(path, "r", errors="replace") as f:
        for line in f:
            if line.startswith('{"index"'):
                continue
            try:
                L = json.loads(line)["layers"]
            except Exception:
                continue
            try:
                t = float(L["frame_time_epoch"][0])
            except Exception:
                continue
            sp = L.get("tcp_srcport", ["0"])[0]
            dp = L.get("tcp_dstport", ["0"])[0]
            fc = int(L["modbus_func_code"][0]) if "modbus_func_code" in L else None
            rn = int(L["modbus_reference_num"][0]) if "modbus_reference_num" in L else None
            wc = int(L["modbus_word_cnt"][0]) if "modbus_word_cnt" in L else None
            ui = L["mbtcp_unit_id"][0] if "mbtcp_unit_id" in L else None
            ti = int(L["mbtcp_trans_id"][0]) if "mbtcp_trans_id" in L else None
            for w in windows:
                k = int(t // w)
                if k != cur[w]:
                    if acc[w] is not None:
                        if cur[w] in out[w]:
                            dup[w] += 1; _merge_m3(out[w][cur[w]], acc[w])
                        else:
                            out[w][cur[w]] = acc[w]
                    cur[w], acc[w] = k, _m3_new()
                a = acc[w]
                a["n"] += 1
                a["ts"].append(t)
                if dp == str(PLC_PORT):
                    a["req"] += 1; a["cports"].add(sp)
                elif sp == str(PLC_PORT):
                    a["resp"] += 1; a["cports"].add(dp)
                if fc is not None:
                    a["fc"][fc] += 1
                if rn is not None:
                    a["reg"][rn] += 1
                    a["reg_min"] = min(a["reg_min"], rn); a["reg_max"] = max(a["reg_max"], rn)
                if wc is not None:
                    a["wc"].append(wc)
                if ui is not None:
                    a["unit"][ui] += 1
                if ti is not None:
                    a["tid"].append(ti)
            n += 1
            if progress_every and n % progress_every == 0:
                log("    ...%s records" % format(n, ","))
    for w in windows:
        if acc[w] is not None:
            if cur[w] in out[w]:
                dup[w] += 1; _merge_m3(out[w][cur[w]], acc[w])
            else:
                out[w][cur[w]] = acc[w]
        out[w] = {k: m3_condense(a, w) for k, a in out[w].items()}
    log("    M3 done: %s records; windows %s; out-of-order flushes %s"
        % (format(n, ","), {w: len(out[w]) for w in windows}, dup))
    return out, dict(records=n, dup=dup)


def _prov_new():
    return dict(edges=0, ops=Counter(), etypes=Counter(), verts=set(),
                outdeg=Counter(), indeg=Counter(), new_v=0,
                pnames=Counter(), pbuckets=Counter(), subtypes=Counter(),
                root_edges=0, proc_v=0, art_v=0, ts=[])


def stream_prov_multi(path, windows=WINDOWS, progress_every=5_000_000, log=print):
    vtype, vname, vbucket, vroot, vsub = {}, {}, {}, {}, {}
    seen_v = set()
    out = {w: {} for w in windows}
    cur = {w: None for w in windows}
    acc = {w: None for w in windows}
    n = 0
    dup = {w: 0 for w in windows}
    with open(path, "r", errors="replace") as f:
        for line in f:
            if line.startswith("VERTEX "):
                try:
                    h = line[8:line.index(")")]
                    a = _attrs(line[line.index("{"):])
                except Exception:
                    continue
                ty = a.get("type", "?")
                vtype[h] = ty
                if ty == "Process":
                    vname[h] = a.get("name", "?")[:24]
                    vroot[h] = 1 if a.get("uid", "1") == "0" or a.get("euid", "1") == "0" else 0
                    vbucket[h] = _path_bucket(a.get("exe", ""))
                else:
                    vsub[h] = a.get("subtype", "?")
                    vbucket[h] = _path_bucket(a.get("path", ""))
                continue
            if not line.startswith("EDGE "):
                continue
            try:
                arrow = line.index(" -> "); close = line.index(")", arrow)
                src = line[6:arrow]; dst = line[arrow + 4:close]
                a = _attrs(line[line.index("{"):])
                t = float(a["time"])
            except Exception:
                continue
            op = a.get("operation", "?"); et = a.get("type", "?")
            # global novelty is defined ONCE over the stream (identical to canonical single-w pass)
            n_novel = 0
            for h in (src, dst):
                if h not in seen_v:
                    seen_v.add(h); n_novel += 1
            for w in windows:
                k = int(t // w)
                if k != cur[w]:
                    if acc[w] is not None:
                        if cur[w] in out[w]:
                            dup[w] += 1; _merge_prov(out[w][cur[w]], acc[w])
                        else:
                            out[w][cur[w]] = acc[w]
                    cur[w], acc[w] = k, _prov_new()
                ac = acc[w]
                ac["edges"] += 1; ac["ts"].append(t)
                ac["ops"][op] += 1; ac["etypes"][et] += 1
                ac["outdeg"][src] += 1; ac["indeg"][dst] += 1
                ac["new_v"] += n_novel
                for h in (src, dst):
                    ac["verts"].add(h)
                    ty = vtype.get(h)
                    if ty == "Process":
                        ac["proc_v"] += 1
                        ac["pnames"][vname.get(h, "?")] += 1
                        ac["root_edges"] += vroot.get(h, 0)
                    elif ty is not None:
                        ac["art_v"] += 1
                        ac["subtypes"][vsub.get(h, "?")] += 1
                    if h in vbucket:
                        ac["pbuckets"][vbucket[h]] += 1
            n += 1
            if progress_every and n % progress_every == 0:
                log("    ...%s edges" % format(n, ","))
    for w in windows:
        if acc[w] is not None:
            if cur[w] in out[w]:
                dup[w] += 1; _merge_prov(out[w][cur[w]], acc[w])
            else:
                out[w][cur[w]] = acc[w]
        out[w] = {k: condense_prov(a, w) for k, a in out[w].items()}
    log("    PROV done: %s edges, %s vertices; windows %s; out-of-order flushes %s"
        % (format(n, ","), format(len(vtype), ","), {w: len(out[w]) for w in windows}, dup))
    return out, dict(edges=n, vertices=len(vtype), dup=dup)
