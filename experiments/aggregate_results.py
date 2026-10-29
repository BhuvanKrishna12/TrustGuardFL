r"""
aggregate_results.py
-------------------------------------------------------------------
Reads every run JSON written by run_experiments.py and writes the LaTeX
inputs of the paper into --tex-dir (default paper/generated):

  results_macros.tex   \R{key} -> value (percentages: mean +- std over seeds)
  tab_headline.tex tab_fed_vs_central.tex tab_ablation.tex tab_perclass.tex
  tab_perday.tex tab_clients.tex tab_config.tex tab_prevalence.tex
  tab_unseen.tex tab_mix.tex tab_rules.tex tab_confusion.tex
  tab_importance.tex tab_v2v3.tex
  tab_audit.tex        copied from leakage_audit.py output with --audit-table
                       (an existing file is never overwritten otherwise)
  fig_sweep.tex fig_clients.tex
  sweep_<model>_<fs>.dat clientsizes.dat   (pgfplots data)

--placeholder writes the same files with "--" so the paper compiles before
any compute. --v2-runs-dir adds the comparison with runs on the old (v2)
labelled files.

Usage:
  python3 experiments/aggregate_results.py --runs-dirs /mnt/d/fl-ids-sweep/rev_runs \
      --v2-runs-dir /mnt/d/fl-ids-sweep/rev_runs_v2 \
      --audit-table /mnt/d/fl-ids-sweep/audit_v3/feature_audit_table.tex
  python3 experiments/aggregate_results.py --placeholder
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

MODELS = [("fed_cnn", "fedcnn", "Fed.\\ CNN"),
          ("central_cnn", "centralcnn", "Central CNN"),
          ("lgbm", "lgbm", "LightGBM")]
MNAME = {m[0]: m[2] for m in MODELS}
FSETS = [("full", "Full"), ("core", "Core")]
CLASSES = [("benign", "benign", "Benign"),
           ("flooding", "flooding", "Flooding"),
           ("ghost_aircraft", "ghost", "Ghost aircraft"),
           ("gps_spoof", "gps", "GPS spoof"),
           ("replay", "replay", "Replay"),
           ("velocity_spoof", "velocity", "Velocity spoof")]
CSHORT = {c[0]: c[1] for c in CLASSES}
CABBR = ["Ben.", "Flood", "Ghost", "GPS", "Repl.", "Vel."]
FEATURES = ["implied_speed_mps", "speed_consistency_delta", "vrate_consistency_delta",
            "inter_arrival_s", "turn_rate_deg_s", "track_message_count",
            "is_physically_implausible", "squawk_changed", "icao24_valid_format", "is_stale",
            "region_avg_inter_arrival", "is_low_coverage_region", "is_climbing_or_descending"]
CORE = ["implied_speed_mps", "speed_consistency_delta", "vrate_consistency_delta",
        "inter_arrival_s", "turn_rate_deg_s", "region_avg_inter_arrival", "is_stale"]
METRICS = [("acc", "accuracy"), ("macrof1", "macro_f1"), ("far", "false_alarm_rate"),
           ("mar", "missed_attack_rate"), ("atkrecall", "avg_attack_recall"),
           ("alertreal", "alert_real_share")]
CLASS_METRICS = [f"{k}-{c[1]}" for c in CLASSES for k in ("rec", "prec", "f1")]
ALL_METRICS = [m[0] for m in METRICS] + CLASS_METRICS
STRENGTHS = [round(i / 10, 1) for i in range(11)]
RHOS = [0.001, 0.01]
NDAYS_MIN = 4
DASH = "--"
PM = "{\\,}$\\pm${\\,}"


def skey(s):
    return f"s{int(round(s * 10)):02d}"


def rkey(rho):
    """0.001 -> rho01 (0.1%), 0.01 -> rho1 (1%)."""
    return "rho" + f"{rho * 100:g}".replace(".", "")


def fkey(feat):
    return feat.replace("_", "-")


def tex_escape(s):
    s = str(s)
    rep = {"\\": "\\textbackslash{}", "&": "\\&", "%": "\\%", "$": "\\$", "#": "\\#",
           "_": "\\_", "{": "\\{", "}": "\\}", "~": "\\textasciitilde{}", "^": "\\^{}"}
    return "".join(rep.get(ch, ch) for ch in s)


def thousands(n):
    if n is None:
        return DASH
    return f"{int(round(n)):,}".replace(",", "{,}")


def _clean(vals):
    return [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]


def fmt_pct(vals, dec=1):
    """mean +- std over seeds, in percent; single seed -> mean only."""
    vals = _clean(vals)
    if not vals:
        return DASH
    a = np.array(vals, float) * 100
    if len(a) == 1:
        return f"{a[0]:.{dec}f}"
    return f"{a.mean():.{dec}f}{PM}{a.std(ddof=1):.{dec}f}"


def fmt_num(vals, dec=1):
    vals = _clean(vals)
    if not vals:
        return DASH
    a = np.array(vals, float)
    if len(a) == 1 or a.std() == 0:
        return f"{a.mean():.{dec}f}"
    return f"{a.mean():.{dec}f}{PM}{a.std(ddof=1):.{dec}f}"


def signed(v, dec=1):
    return f"${v:.{dec}f}$" if v < 0 else f"{v:.{dec}f}"


def t975(df):
    try:
        from scipy.stats import t
        return float(t.ppf(0.975, df))
    except Exception:
        table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
                 8: 2.306, 9: 2.262, 10: 2.228}
        return table.get(df, 1.96)


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

class Results:
    def __init__(self, runs_dirs):
        self.runs = {}        # (model, fs, seed) -> json
        self.clients = {}     # seed -> clients json
        for d in runs_dirs or []:
            rd = os.path.join(d, "runs") if os.path.isdir(os.path.join(d, "runs")) else d
            for p in sorted(glob.glob(os.path.join(rd, "*__*__seed*.json"))):
                r = json.load(open(p))
                self.runs[(r["model"], r["feature_set"], int(r["seed"]))] = r
            for p in sorted(glob.glob(os.path.join(rd, "clients__seed*.json"))):
                c = json.load(open(p))
                self.clients[int(c["seed"])] = c
        self.seeds = sorted({k[2] for k in self.runs})
        self.any = next(iter(self.runs.values()), None)
        self.days = sorted({d for r in self.runs.values() for d in r["holdout"].get("main_days", [])})

    def group(self, model, fset):
        return [self.runs[k] for k in sorted(self.runs) if k[0] == model and k[1] == fset]

    def by_seed(self, model, fset):
        return {k[2]: self.runs[k] for k in self.runs if k[0] == model and k[1] == fset}

    def metric(self, model, fset, s, name, where=("pooled",)):
        """list over seeds. where: ('pooled',), ('days', date), ('unseen',),
        ('prev', rho, 'mismatched'|'oracle')."""
        return [v for v in (metric_of(r, s, name, where) for r in self.group(model, fset))
                if v is not None]

    def training(self, model, fset, key):
        return [r["training"].get(key) for r in self.group(model, fset)
                if r["training"].get(key) is not None]


def select(entry, where):
    if entry is None:
        return None
    if where[0] == "pooled":
        return entry.get("pooled")
    if where[0] == "days":
        return entry.get("days", {}).get(where[1])
    if where[0] == "unseen":
        return entry.get("unseen")
    if where[0] == "prev":
        p = entry.get("prevalence", {}).get(f"{where[1]:g}")
        return p.get(where[2]) if p else None
    raise ValueError(where)


def metric_of(run, s, name, where=("pooled",)):
    m = select(run["metrics"].get(f"{s:.2f}"), where)
    return extract(m, name) if m is not None else None


def extract(m, name):
    for short, long_ in METRICS:
        if name == short:
            return m.get(long_)
    kind, cls = name.split("-", 1)
    full = {c[1]: c[0] for c in CLASSES}[cls]
    return m["per_class"][full][{"rec": "recall", "prec": "precision", "f1": "f1"}[kind]]


def jsd(p, q):
    p = np.asarray(p, float) / max(np.sum(p), 1e-12)
    q = np.asarray(q, float) / max(np.sum(q), 1e-12)
    m = (p + q) / 2

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def client_stats(res):
    """Seed-independent client facts (total rows and label counts per client)."""
    src = None
    if res.clients:
        src = res.clients[sorted(res.clients)[0]]["clients"]
    else:
        for r in res.runs.values():
            if r.get("clients"):
                src = r["clients"]
                break
    if not src:
        return None
    names = [c[0] for c in CLASSES]
    sizes = np.array([c["n_train"] + c["n_val"] for c in src])
    lab = np.array([[c["label_counts"].get(n, 0) for n in names] for c in src], float)
    j = np.array([jsd(row, lab.sum(0)) for row in lab])
    return {"n": len(src), "sizes": np.sort(sizes)[::-1], "min": int(sizes.min()),
            "median": float(np.median(sizes)), "max": int(sizes.max()),
            "jsd_mean": float(j.mean()), "jsd_max": float(j.max()),
            "missing_share": float((lab == 0).any(1).mean())}


def paired_gap(res, fset, s, name):
    """fed_cnn minus central_cnn, paired by seed (same clients, same rows)."""
    f, c = res.by_seed("fed_cnn", fset), res.by_seed("central_cnn", fset)
    d = []
    for seed in sorted(set(f) & set(c)):
        a, b = metric_of(f[seed], s, name), metric_of(c[seed], s, name)
        if a is not None and b is not None:
            d.append((a - b) * 100)
    if not d:
        return None
    d = np.array(d)
    out = {"n": len(d), "mean": float(d.mean())}
    if len(d) > 1:
        sd = float(d.std(ddof=1))
        h = t975(len(d) - 1) * sd / np.sqrt(len(d))
        out.update(sd=sd, lo=out["mean"] - h, hi=out["mean"] + h)
    return out


def mean_confusion(res, model, fset, s=1.0):
    mats = []
    for r in res.group(model, fset):
        m = select(r["metrics"].get(f"{s:.2f}"), ("pooled",))
        if m is None:
            continue
        cm = np.array(m["confusion_matrix"], float)
        mats.append(cm / np.maximum(cm.sum(1, keepdims=True), 1e-12) * 100)
    return np.mean(mats, 0) if mats else None


def rule_metrics(hold, rule, cls):
    """recall / precision / FAR of a test-set rule that predicts class `cls`."""
    if not hold or not hold.get("rules") or rule not in hold["rules"]:
        return None
    fires = hold["rules"][rule]["fires_by_class"]
    sup = hold["support"]
    tp = fires.get(cls, 0)
    allf = sum(fires.values())
    return {"recall": tp / max(1, sup.get(cls, 0)),
            "precision": tp / allf if allf else 0.0,
            "far": fires.get("benign", 0) / max(1, sup.get("benign", 0))}


# ---------------------------------------------------------------------------
# macros
# ---------------------------------------------------------------------------

def build_macros(res, v2, placeholder):
    M = {}
    ph = placeholder

    def put(k, v):
        M[k] = DASH if ph or v is None or v == DASH else v

    a = res.any
    d = a["data"] if a else {}
    h = a["holdout"] if a else {}
    ndays = max(NDAYS_MIN, len(res.days))

    # ---- per-result metrics ----
    for model, mk, _ in MODELS:
        for fset, _ in FSETS:
            for s in STRENGTHS:
                base = f"{mk}-{fset}-{skey(s)}"
                for name in ALL_METRICS:
                    put(f"{base}-{name}", fmt_pct(res.metric(model, fset, s, name)))
                    put(f"{base}-unseen-{name}", fmt_pct(res.metric(model, fset, s, name, ("unseen",))))
                for di in range(ndays):
                    day = res.days[di] if di < len(res.days) else None
                    for name, _ in METRICS:
                        put(f"{base}-{name}-d{di + 1}",
                            fmt_pct(res.metric(model, fset, s, name, ("days", day))) if day else None)
                for rho in RHOS:
                    for kind, pre in (("mismatched", ""), ("oracle", "oracle-")):
                        for name in ALL_METRICS:
                            put(f"{base}-{pre}{rkey(rho)}-{name}",
                                fmt_pct(res.metric(model, fset, s, name, ("prev", rho, kind))))

    # ---- v2 comparison ----
    for model, mk, _ in MODELS:
        for s in STRENGTHS:
            for name in ALL_METRICS:
                put(f"v2-{mk}-full-{skey(s)}-{name}",
                    fmt_pct(v2.metric(model, "full", s, name)) if v2 else None)
        put(f"v2-nseeds-{mk}", str(len(v2.group(model, "full"))) if v2 and v2.group(model, "full") else None)

    # ---- days ----
    for di in range(ndays):
        put(f"day{di + 1}", res.days[di] if di < len(res.days) else None)
        put(f"day{di + 1}rows", thousands(h.get("day_counts", {}).get(res.days[di]))
            if a and di < len(res.days) else None)

    # ---- data facts ----
    put("nclients", str(d.get("n_clients")) if a else None)
    put("ndroppedregions", str(d.get("n_dropped_regions")) if a else None)
    put("nregions", str(d.get("n_regions_in_sample")) if a else None)
    put("trainrows", thousands(d.get("train_sample_rows")) if a else None)
    put("trainrawrows", thousands(d.get("train_raw_rows")) if a and d.get("train_raw_rows") else None)
    put("trainsplitrows", thousands(d.get("train_split_rows")) if a else None)
    put("valrows", thousands(d.get("val_rows")) if a else None)
    put("maxtotalrows", thousands(d.get("max_total_rows")) if a else None)
    put("minclientrows", thousands(d.get("min_client_rows")) if a else None)
    put("samplecap", thousands(d.get("cap_per_region_class")) if a and d.get("cap_per_region_class") else None)
    put("sampler", {"uniform_bottom_k": "uniform per (region, class)",
                    "legacy_load_data": "legacy load\\_data"}.get(d.get("sampler"), d.get("sampler"))
        if a else None)
    put("holdoutrows", thousands(h.get("rows")) if a else None)
    put("nseeds", str(len(res.seeds)) if a else None)
    put("seeds", ", ".join(map(str, res.seeds)) if a else None)
    put("aircraft-trainraw", thousands(d.get("train_raw_aircraft")) if a and d.get("train_raw_aircraft") else None)
    put("aircraft-trainsample", thousands(d.get("train_sample_aircraft"))
        if a and d.get("train_sample_aircraft") else None)
    ov = (h.get("overlap") or {}) if a else {}
    put("aircraft-test", thousands(ov.get("test_aircraft")) if ov.get("has_icao24") else None)
    put("overlap-rows", f"{ov['share_rows_seen'] * 100:.1f}" if ov.get("has_icao24") else None)
    put("overlap-aircraft", f"{ov['share_aircraft_seen'] * 100:.1f}" if ov.get("has_icao24") else None)
    put("unseenrows", thousands(ov.get("unseen_rows")) if ov.get("has_icao24") else None)
    tot_sup = sum(h.get("support", {}).values()) if a else 0
    for full, short, _ in CLASSES:
        sup = h.get("support", {}).get(full) if a else None
        put(f"support-{short}", thousands(sup) if sup is not None else None)
        put(f"testmix-{short}", f"{sup / tot_sup * 100:.2f}" if sup is not None and tot_sup else None)
        us = (ov.get("unseen_support") or {}).get(full)
        put(f"unseen-support-{short}", thousands(us) if us is not None else None)
        pr = d.get("natural_prior", {}).get(full) if a else None
        put(f"prior-{short}", f"{pr * 100:.2f}" if pr is not None else None)
        bl = (h.get("blank_share_by_class") or {}).get(full) if a else None
        put(f"blank-{short}", f"{bl * 100:.2f}" if bl is not None else None)
        cw = d.get("class_weights", {}).get(full) if a else None
        put(f"classweight-{short}", f"{cw:.2f}" if cw is not None else None)
        n = d.get("train_sample_class_counts", {}).get(full) if a else None
        tot = d.get("train_sample_rows") if a else None
        put(f"trainmix-{short}", f"{n / tot * 100:.2f}" if n is not None and tot else None)

    # training sample by day
    comp = (d.get("train_sample_composition") or {}).get("per_day", {}) if a else {}
    tot = sum(v["rows"] for v in comp.values()) if comp else 0
    tdays = sorted(k for k, v in comp.items() if tot and v["rows"] >= 0.01 * tot)
    for i in range(max(NDAYS_MIN, len(tdays))):
        dd = tdays[i] if i < len(tdays) else None
        put(f"trainday{i + 1}", dd)
        put(f"trainday{i + 1}-share", f"{comp[dd]['rows'] / tot * 100:.1f}" if dd else None)
        put(f"trainday{i + 1}-rows", thousands(comp[dd]["rows"]) if dd else None)

    # rules
    for rule, rk in (("ghost_aircraft", "ghost"), ("flooding", "flooding")):
        rm = rule_metrics(h, rule, rule) if a else None
        for k in ("recall", "precision", "far"):
            put(f"rule-{rk}-{k}", f"{rm[k] * 100:.1f}" if rm else None)

    # model size, communication, training cost
    fed = res.group("fed_cnn", "full")
    mb = fed[0]["training"]["model_bytes"] if fed else None
    cb = fed[0]["training"]["comm_bytes"] if fed else None
    put("modelbytes", thousands(mb) if mb else None)
    put("modelkb", f"{mb / 1000:.1f}~kB" if mb else None)
    put("modelparams", thousands(fed[0]["training"]["n_params"]) if fed else None)
    put("commbytes", f"{cb / 1e6:.1f}~MB" if cb else None)
    put("commbytesexact", thousands(cb) if cb else None)
    put("commperround", f"{cb / fed[0]['training']['rounds'] / 1e6:.2f}~MB" if cb else None)
    fedc = res.group("fed_cnn", "core")
    put("modelbytes-core", thousands(fedc[0]["training"]["model_bytes"]) if fedc else None)
    put("modelkb-core", f"{fedc[0]['training']['model_bytes'] / 1000:.1f}~kB" if fedc else None)
    put("commbytes-core", f"{fedc[0]['training']['comm_bytes'] / 1e6:.1f}~MB" if fedc else None)
    for fset, _ in FSETS:
        put(f"bestround-fedcnn-{fset}", fmt_num(res.training("fed_cnn", fset, "best_round")))
        put(f"bestepoch-centralcnn-{fset}", fmt_num(res.training("central_cnn", fset, "best_epoch")))
        put(f"epochsrun-centralcnn-{fset}", fmt_num(res.training("central_cnn", fset, "epochs_run")))
        put(f"bestiter-lgbm-{fset}", fmt_num(res.training("lgbm", fset, "best_iteration"), 0))
        for model, mk, _ in MODELS:
            put(f"valacc-{mk}-{fset}", fmt_pct(res.training(model, fset, "best_val_acc")))
            put(f"trainmin-{mk}-{fset}",
                fmt_num([t / 60 for t in res.training(model, fset, "train_seconds")]))
            st = res.training(model, fset, "gradient_steps")
            put(f"steps-{mk}-{fset}", thousands(np.mean(st)) if st else None)
        for s in (0.0, 1.0):
            for name in ("macrof1", "far", "atkrecall"):
                g = None if ph else paired_gap(res, fset, s, name)
                base = f"gap-{fset}-{skey(s)}-{name}"
                put(f"{base}-mean", signed(g["mean"]) if g else None)
                put(f"{base}-sd", f"{g['sd']:.1f}" if g and "sd" in g else None)
                put(f"{base}-ci", f"$[{g['lo']:.1f}, {g['hi']:.1f}]$" if g and "lo" in g else None)
                put(f"{base}-n", str(g["n"]) if g else None)
        for feat in FEATURES:
            gains = [r["training"]["feature_importance_gain_share"].get(feat)
                     for r in res.group("lgbm", fset)
                     if r["training"].get("feature_importance_gain_share")]
            put(f"gain-{fset}-{fkey(feat)}", fmt_pct([g for g in gains if g is not None]))

    cfg = a["config"] if a else {}
    put("nrounds", str(cfg.get("rounds")) if a else None)
    put("localepochs", str(cfg.get("local_epochs")) if a else None)
    put("centralmaxepochs", str(cfg.get("central_max_epochs")) if a else None)
    put("centralpatience", str(cfg.get("central_patience")) if a else None)
    put("lgbmmaxtrees", thousands(cfg.get("lgbm_n_estimators")) if a else None)
    put("lgbmleaves", str(cfg.get("lgbm_num_leaves")) if a else None)
    put("lgbmlr", str(cfg.get("lgbm_learning_rate")) if a else None)
    put("lgbmes", str(cfg.get("lgbm_early_stopping")) if a else None)
    vers = (a.get("versions") or {}) if a else {}
    for k in ("python", "torch", "lightgbm", "sklearn", "pandas", "numpy"):
        put(f"ver-{k}", tex_escape(vers[k]) if vers.get(k) else None)
    cs = client_stats(res)
    put("clientrows-min", thousands(cs["min"]) if cs else None)
    put("clientrows-median", thousands(cs["median"]) if cs else None)
    put("clientrows-max", thousands(cs["max"]) if cs else None)
    put("clientjsd-mean", f"{cs['jsd_mean']:.3f}" if cs else None)
    put("clientjsd-max", f"{cs['jsd_max']:.3f}" if cs else None)
    put("clientmissing-share", f"{cs['missing_share'] * 100:.1f}" if cs else None)
    return M, cs, max(NDAYS_MIN, len(tdays))


def write_macros(M, path):
    lines = ["% Generated by experiments/aggregate_results.py. Do not edit by hand.",
             "% Usage: \\R{fedcnn-full-s10-macrof1}. Percentages are mean $\\pm$ std over seeds,",
             "% without the % sign. Undefined keys print a bold ??.",
             "\\providecommand{\\R}[1]{\\ifcsname res@#1\\endcsname\\csname res@#1\\endcsname"
             "\\else\\textbf{??}\\fi}"]
    for k in sorted(M):
        lines.append(f"\\expandafter\\def\\csname res@{k}\\endcsname{{{M[k]}}}")
    open(path, "w").write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------

def V(M, key):
    return M.get(key, DASH)


def table(env, caption, label, colspec, header_rows, body_rows, size="\\scriptsize",
          tabcolsep="3pt", note=None):
    out = [f"\\begin{{{env}}}[t]", "\\centering", size,
           f"\\caption{{{caption}}}", f"\\label{{{label}}}",
           f"\\setlength{{\\tabcolsep}}{{{tabcolsep}}}",
           f"\\begin{{tabular}}{{{colspec}}}", "\\toprule"]
    out += header_rows
    out.append("\\midrule")
    out += body_rows
    out += ["\\bottomrule", "\\end{tabular}"]
    if note:
        out += ["\\\\[2pt]", f"\\parbox{{\\linewidth}}{{\\scriptsize {note}}}"]
    out.append(f"\\end{{{env}}}")
    return "\n".join(out) + "\n"


SEED_NOTE = "Mean $\\pm$ standard deviation over \\R{nseeds} seeds, in \\%."


def tab_headline(M):
    rows = []
    for i, (_, mk, mname) in enumerate(MODELS):
        if i:
            rows.append("\\midrule")
        for s in (0.0, 0.5, 1.0):
            b = f"{mk}-full-{skey(s)}"
            rows.append(f"{mname if s == 0 else ''} & {s:g} & {V(M, b + '-acc')} & {V(M, b + '-far')} & "
                        f"{V(M, b + '-atkrecall')} & {V(M, b + '-macrof1')} \\\\")
    return table("table", "Headline Results on the Held-Out Days (Full Feature Set)", "tab:headline",
                 "llrrrr", ["Model & $s$ & Acc. & FAR & Atk.\\ rec. & Macro F1 \\\\"], rows,
                 tabcolsep="2.5pt", note=SEED_NOTE + " FAR: false alarm rate; Atk.\\ rec.: "
                 "average recall over the five attack classes; $s$: correction strength.")


def tab_fed_vs_central(M):
    rows = []
    for fi, (fset, fname) in enumerate(FSETS):
        if fi:
            rows.append("\\midrule")
        for mi, (_, mk, mname) in enumerate(MODELS):
            for s in (0.0, 1.0):
                b = f"{mk}-{fset}-{skey(s)}"
                lead = fname if (mi == 0 and s == 0) else ""
                rows.append(f"{lead} & {mname if s == 0 else ''} & {s:g} & {V(M, b + '-acc')} & "
                            f"{V(M, b + '-far')} & {V(M, b + '-mar')} & {V(M, b + '-atkrecall')} & "
                            f"{V(M, b + '-macrof1')} \\\\")
        for s in (0.0, 1.0):
            g = f"gap-{fset}-{skey(s)}"
            cell = lambda n: f"{V(M, f'{g}-{n}-mean')} {V(M, f'{g}-{n}-ci')}"  # noqa: E731
            rows.append(f" & \\emph{{Fed.$-$central}} & {s:g} & & {cell('far')} & & "
                        f"{cell('atkrecall')} & {cell('macrof1')} \\\\")
    return table("table*", "Federated versus Centralized Training on Identical Rows",
                 "tab:fedcentral", "lllrrrrr",
                 ["Features & Model & $s$ & Accuracy & FAR & Missed attacks & Attack recall & "
                  "Macro F1 \\\\"], rows,
                 note=SEED_NOTE + " Fed.\\ CNN and Central CNN share architecture, scaler, class "
                 "weights, optimizer, training rows and validation rows; only the training "
                 "procedure differs. Fed.$-$central: paired difference per seed in percentage "
                 "points, mean and 95\\% $t$-interval over seeds. Missed attacks: attack rows "
                 "predicted benign over attack rows.")


def tab_ablation(M):
    rows = []
    cls_cols = " & ".join(c[2] for c in CLASSES)
    for si, s in enumerate((1.0, 0.5)):
        if si:
            rows.append("\\midrule")
        for mi, (_, mk, mname) in enumerate(MODELS):
            for fset, fname in FSETS:
                b = f"{mk}-{fset}-{skey(s)}"
                rec = " & ".join(V(M, f"{b}-rec-{c[1]}") for c in CLASSES)
                lead_s = f"{s:g}" if (mi == 0 and fset == "full") else ""
                rows.append(f"{lead_s} & {mname if fset == 'full' else ''} & {fname} & "
                            f"{V(M, b + '-macrof1')} & {V(M, b + '-far')} & {V(M, b + '-atkrecall')} & "
                            f"{rec} \\\\")
    hdr = ["& & & & & & \\multicolumn{6}{c}{Recall} \\\\ \\cmidrule(lr){7-12}",
           f"$s$ & Model & Features & Macro F1 & FAR & Atk.\\ rec. & {cls_cols} \\\\"]
    return table("table*", "Full versus Core Feature Set", "tab:ablation", "lll" + "r" * 9,
                 hdr, rows, tabcolsep="2pt",
                 note=SEED_NOTE + " The core set keeps implied speed, speed and vertical-rate "
                 "consistency, inter-arrival time, turn rate, regional inter-arrival baseline and "
                 "the stale flag.")


def tab_perclass(M):
    rows = []
    for full, short, cname in CLASSES:
        cells = []
        for _, mk, _ in MODELS:
            b = f"{mk}-full-s10"
            cells += [V(M, f"{b}-prec-{short}"), V(M, f"{b}-rec-{short}"), V(M, f"{b}-f1-{short}")]
        rows.append(f"{cname} & {V(M, 'support-' + short)} & " + " & ".join(cells) + " \\\\")
    hdr = ["& & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{m[2]}}}" for m in MODELS) + " \\\\",
           "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\\cmidrule(lr){9-11}",
           "Class & Support & " + " & ".join(["P & R & F1"] * 3) + " \\\\"]
    return table("table*", "Per-Class Precision, Recall and F1 at Full Correction ($s=1$, Full "
                 "Feature Set)", "tab:perclass", "lr" + "r" * 9, hdr, rows, tabcolsep="2pt",
                 note=SEED_NOTE + " Support: held-out rows of the class.")


def tab_perday(M, ndays):
    rows = []
    for di in range(ndays):
        cells = []
        for _, mk, _ in MODELS:
            b = f"{mk}-full-s10"
            cells += [V(M, f"{b}-macrof1-d{di + 1}"), V(M, f"{b}-far-d{di + 1}"),
                      V(M, f"{b}-atkrecall-d{di + 1}")]
        rows.append(f"{V(M, f'day{di + 1}')} & {V(M, f'day{di + 1}rows')} & " + " & ".join(cells) + " \\\\")
    hdr = ["& & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{m[2]}}}" for m in MODELS) + " \\\\",
           "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\\cmidrule(lr){9-11}",
           "Day & Rows & " + " & ".join(["Macro F1 & FAR & Atk.\\ rec."] * 3) + " \\\\"]
    return table("table*", "Results per Held-Out Day at Full Correction ($s=1$, Full Feature Set)",
                 "tab:perday", "lr" + "r" * 9, hdr, rows, tabcolsep="2pt",
                 note=SEED_NOTE + " The day of a row is the UTC date of its last position "
                 "update (the replay attack shifts the time field but not this one).")


def tab_prevalence(M):
    rows = []
    rk = [rkey(r) for r in (0.01, 0.001)]
    for mi, (_, mk, mname) in enumerate(MODELS):
        if mi:
            rows.append("\\midrule")
        for s in (0.0, 0.5, 1.0):
            b = f"{mk}-full-{skey(s)}"
            cells = [V(M, f"{b}-macrof1"), V(M, f"{b}-alertreal")]
            for r in rk:
                cells += [V(M, f"{b}-{r}-macrof1"), V(M, f"{b}-{r}-alertreal")]
            rows.append(f"{mname if s == 0 else ''} & {s:g} & " + " & ".join(cells) + " \\\\")
        b = f"{mk}-full-s10"
        cells = [DASH, DASH]
        for r in rk:
            cells += [V(M, f"{b}-oracle-{r}-macrof1"), V(M, f"{b}-oracle-{r}-alertreal")]
        rows.append(" & oracle & " + " & ".join(cells) + " \\\\")
    hdr = ["& & \\multicolumn{2}{c}{Test mix} & \\multicolumn{2}{c}{$\\rho=1\\%$} & "
           "\\multicolumn{2}{c}{$\\rho=0.1\\%$} \\\\",
           "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}",
           "Model & $s$ & " + " & ".join(["Macro F1 & Real alerts"] * 3) + " \\\\"]
    return table("table*", "Results under Assumed Deployment Attack Prevalence (Full Feature Set)",
                 "tab:prevalence", "ll" + "r" * 6, hdr, rows, tabcolsep="3pt",
                 note=SEED_NOTE + " Test rows are reweighted (not resampled) so that attacks make "
                 "up a share $\\rho$ of traffic, the five attack classes keeping their test "
                 "proportions; models are not retrained. Rows $s$: correction with the training-day "
                 "prior, as a deployment would apply it. Oracle: correction at $s=1$ with the true "
                 "deployment prior. Real alerts: share of rows flagged as any attack that are "
                 "attacks. Recall and false alarm rate do not change under reweighting.")


def tab_unseen(M):
    rows = []
    for _, mk, mname in MODELS:
        b = f"{mk}-full-s10"
        rows.append(f"{mname} & " + " & ".join(
            [V(M, f"{b}-macrof1"), V(M, f"{b}-far"), V(M, f"{b}-atkrecall"),
             V(M, f"{b}-unseen-macrof1"), V(M, f"{b}-unseen-far"), V(M, f"{b}-unseen-atkrecall")])
            + " \\\\")
    sup = ", ".join(f"{c[2].lower()} \\R{{unseen-support-{c[1]}}}" for c in CLASSES)
    hdr = ["& \\multicolumn{3}{c}{All test rows} & \\multicolumn{3}{c}{Unseen aircraft} \\\\",
           "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
           "Model & Macro F1 & FAR & Atk.\\ rec. & Macro F1 & FAR & Atk.\\ rec. \\\\"]
    return table("table*", "Test Rows from Aircraft Never Seen in Training ($s=1$, Full Feature Set)",
                 "tab:unseen", "l" + "r" * 6, hdr, rows, tabcolsep="3pt",
                 note=SEED_NOTE + " \\R{overlap-rows}\\% of test rows and \\R{overlap-aircraft}\\% of "
                 "test aircraft addresses also occur in the training files. Unseen-aircraft subset: "
                 f"\\R{{unseenrows}} rows ({sup}). Ghost rows carry fabricated addresses and are "
                 "therefore almost all in this subset.")


def tab_mix(M, ntdays):
    rows = [f"{c[2]} & {V(M, 'prior-' + c[1])} & {V(M, 'trainmix-' + c[1])} & "
            f"{V(M, 'testmix-' + c[1])} \\\\" for c in CLASSES]
    rows += ["\\midrule",
             f"Rows & {V(M, 'trainrawrows')} & {V(M, 'trainrows')} & {V(M, 'holdoutrows')} \\\\",
             f"Aircraft (icao24) & {V(M, 'aircraft-trainraw')} & {V(M, 'aircraft-trainsample')} & "
             f"{V(M, 'aircraft-test')} \\\\",
             "\\midrule"]
    for i in range(ntdays):
        rows.append(f"Sample from {V(M, f'trainday{i + 1}')} & & {V(M, f'trainday{i + 1}-share')} & \\\\")
    return table("table", "Class Mix and Split Sizes", "tab:mix", "lrrr",
                 ["& Training days & Training sample & Test \\\\"], rows,
                 size="\\footnotesize", tabcolsep="4pt",
                 note="Class shares in \\%. Training days: all rows of the four training files "
                 "(the prior used for correction). Training sample: the capped sample models are "
                 "trained on (\\R{sampler} sampler). Test: the held-out sample. The last rows give "
                 "the share of the training sample drawn from each training day.")


def tab_rules(M):
    rows = [f"Ghost aircraft & All four rate features missing & {V(M, 'rule-ghost-recall')} & "
            f"{V(M, 'rule-ghost-precision')} & {V(M, 'rule-ghost-far')} \\\\",
            f"Flooding & Inter-arrival time below 1~s & {V(M, 'rule-flooding-recall')} & "
            f"{V(M, 'rule-flooding-precision')} & {V(M, 'rule-flooding-far')} \\\\"]
    out = ["\\begin{table}[t]", "\\centering", "\\footnotesize",
           "\\caption{Single-Rule Baselines on the Test Set}", "\\label{tab:rules}",
           "\\setlength{\\tabcolsep}{3pt}",
           "\\begin{tabularx}{\\columnwidth}{@{}l>{\\raggedright\\arraybackslash}Xrrr@{}}", "\\toprule",
           "Class & Rule & Recall & Prec. & FAR \\\\", "\\midrule"] + rows + [
           "\\bottomrule", "\\end{tabularx}", "\\\\[2pt]",
           "\\parbox{\\linewidth}{\\scriptsize In \\%, no training. Recall and precision for the "
           "rule's class; FAR: benign rows the rule fires on over benign rows. Rate features: "
           "implied speed, speed and vertical-rate consistency, turn rate. Missing inter-arrival "
           "times do not fire the flooding rule.}", "\\end{table}"]
    return "\n".join(out) + "\n"


def tab_confusion(res, placeholder):
    blocks = [("fed_cnn", None), ("central_cnn", None)]
    if not placeholder:
        blocks = [(m, mean_confusion(res, m, "full")) for m, _ in blocks]
    rows = []
    for i, (full, short, cname) in enumerate(CLASSES):
        cells = []
        for _, cm in blocks:
            cells += [f"{cm[i, j]:.1f}" if cm is not None else DASH for j in range(len(CLASSES))]
        rows.append(f"{cname} & " + " & ".join(cells) + " \\\\")
    hdr = ["& " + " & ".join(f"\\multicolumn{{6}}{{c}}{{{MNAME[m]} (predicted)}}" for m, _ in blocks)
           + " \\\\", "\\cmidrule(lr){2-7}\\cmidrule(lr){8-13}",
           "Actual & " + " & ".join(CABBR * 2) + " \\\\"]
    return table("table*", "Confusion Matrices at Full Correction ($s=1$, Full Feature Set)",
                 "tab:confusion", "l" + "r" * 12, hdr, rows, tabcolsep="2.5pt",
                 note="Row-normalised (each row sums to 100\\%), mean over \\R{nseeds} seeds.")


def tab_importance(M):
    rows = []
    for feat in FEATURES:
        core = V(M, f"gain-core-{fkey(feat)}") if feat in CORE else "n/a"
        rows.append(f"{tex_escape(feat)} & {V(M, f'gain-full-{fkey(feat)}')} & {core} \\\\")
    return table("table", "LightGBM Feature Importance (Gain Share)", "tab:importance", "lrr",
                 ["Feature & Full set & Core set \\\\"], rows, size="\\footnotesize",
                 tabcolsep="4pt",
                 note=SEED_NOTE + " Share of total split gain; n/a: feature not in the core set. "
                 "Correlated features can substitute for each other, so a low share does not rule "
                 "out leakage.")


def tab_v2v3(M):
    rows = []
    cols = ["macrof1", "far", "atkrecall", "rec-flooding", "rec-ghost"]
    for mi, (model, mk, mname) in enumerate([MODELS[2], MODELS[0]]):
        if mi:
            rows.append("\\midrule")
        for s in (0.0, 1.0):
            for ver, pre in (("v2", "v2-"), ("v3", "")):
                b = f"{pre}{mk}-full-{skey(s)}"
                lead = mname if (s == 0 and ver == "v2") else ""
                scol = f"{s:g}" if ver == "v2" else ""
                rows.append(f"{lead} & {scol} & {ver} & " +
                            " & ".join(V(M, f"{b}-{c}") for c in cols) + " \\\\")
    return table("table*", "Old (v2) versus Corrected (v3) Pipeline (Full Feature Set)", "tab:v2v3",
                 "lllrrrrr",
                 ["Model & $s$ & Data & Macro F1 & FAR & Atk.\\ rec. & Flooding rec. & "
                  "Ghost rec. \\\\"], rows, tabcolsep="3pt",
                 note="Mean $\\pm$ standard deviation over seeds, in \\%. v2: the earlier labelled "
                 "files and test sample; v3: the corrected pipeline. Each version is evaluated on "
                 "its own test sample, drawn the same way.")


def tab_clients(M):
    rows = [f"Clients (region cells) & {V(M, 'nclients')} \\\\",
            f"Region cells dropped ($<${V(M, 'minclientrows')} rows) & {V(M, 'ndroppedregions')} \\\\",
            f"Rows per client, min & {V(M, 'clientrows-min')} \\\\",
            f"Rows per client, median & {V(M, 'clientrows-median')} \\\\",
            f"Rows per client, max & {V(M, 'clientrows-max')} \\\\",
            f"Label JSD to global, mean & {V(M, 'clientjsd-mean')} \\\\",
            f"Label JSD to global, max & {V(M, 'clientjsd-max')} \\\\",
            f"Clients missing $\\geq$1 class (\\%) & {V(M, 'clientmissing-share')} \\\\"]
    return table("table", "Client Population", "tab:clients", "lr", ["Quantity & Value \\\\"],
                 rows, size="\\footnotesize", tabcolsep="4pt",
                 note="Rows per client count training plus validation rows of the capped sample. "
                 "JSD: Jensen-Shannon divergence (base 2, range 0 to 1) between a client's label "
                 "distribution and the pooled one.")


def tab_config(M):
    r = [
        ("Training sample", f"{V(M, 'trainrows')} rows from {V(M, 'trainrawrows')} raw rows; "
                            f"{V(M, 'sampler')} sampler, cap {V(M, 'samplecap')} rows per "
                            f"(cell, class), budget {V(M, 'maxtotalrows')}, seed 42"),
        ("Train / validation", f"80/20 per client; {V(M, 'trainsplitrows')} / {V(M, 'valrows')} rows"),
        ("Clients", f"{V(M, 'nclients')} region cells with $\\geq${V(M, 'minclientrows')} rows"),
        ("Seeds", f"{V(M, 'nseeds')} ({V(M, 'seeds')}): init, splits, shuffling, bagging"),
        ("CNN", f"Conv1d 32, 64 ch., $k=3$; {V(M, 'modelparams')} params, "
                f"{V(M, 'modelbytes')} bytes ({V(M, 'modelkb')})"),
        ("Optimizer", "Adam, lr $10^{-3}$, batch 512, balanced class weights"),
        ("Scaler", "Standardization fitted on the pooled training sample"),
        ("FedAvg", f"{V(M, 'nrounds')} rounds, {V(M, 'localepochs')} local epochs, participation 1.0"),
        ("Communication", f"{V(M, 'commbytes')} in total (up and down), "
                          f"{V(M, 'commperround')} per round"),
        ("Central CNN", f"$\\leq${V(M, 'centralmaxepochs')} epochs, early stop patience "
                        f"{V(M, 'centralpatience')}"),
        ("LightGBM", f"{V(M, 'lgbmleaves')} leaves, lr {V(M, 'lgbmlr')}, $\\leq${V(M, 'lgbmmaxtrees')} "
                     f"trees, early stop {V(M, 'lgbmes')} rounds, subsample 0.8, colsample 0.9"),
        ("CNN selection", "Round or epoch with the best unweighted validation accuracy of the "
                          "uncorrected prediction ($s=0$)"),
        ("LightGBM selection", "Early stopping on class-weighted multi\\_logloss of the "
                               "validation rows"),
        ("Training cost (full set)",
         f"Fed.\\ CNN {V(M, 'steps-fedcnn-full')} steps, {V(M, 'trainmin-fedcnn-full')} min; "
         f"Central CNN {V(M, 'steps-centralcnn-full')} steps, {V(M, 'trainmin-centralcnn-full')} min; "
         f"LightGBM {V(M, 'steps-lgbm-full')} iterations, {V(M, 'trainmin-lgbm-full')} min"),
        ("Prior correction", "$s\\in\\{0,0.1,\\ldots,1\\}$, prior from all training-file rows"),
        ("Selection data", "Training-day validation rows only; test days never used"),
        ("Software", f"Python {V(M, 'ver-python')}, PyTorch {V(M, 'ver-torch')}, LightGBM "
                     f"{V(M, 'ver-lightgbm')}, scikit-learn {V(M, 'ver-sklearn')}, pandas "
                     f"{V(M, 'ver-pandas')}, NumPy {V(M, 'ver-numpy')}"),
    ]
    rows = [f"{a} & {b} \\\\" for a, b in r]
    out = ["\\begin{table}[t]", "\\centering", "\\footnotesize",
           "\\caption{Training Configuration}", "\\label{tab:config}",
           "\\setlength{\\tabcolsep}{3pt}",
           "\\begin{tabularx}{\\columnwidth}{@{}l>{\\raggedright\\arraybackslash}X@{}}", "\\toprule",
           "Setting & Value \\\\", "\\midrule"] + rows + ["\\bottomrule", "\\end{tabularx}",
                                                          "\\\\[2pt]",
                                                          "\\parbox{\\linewidth}{\\scriptsize "
                                                          "Training cost: mean over seeds; steps "
                                                          "are optimizer updates summed over "
                                                          "clients.}",
                                                          "\\end{table}"]
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def write_dat(res, tex_dir, placeholder, cs):
    for model, mk, _ in MODELS:
        for fset, _ in FSETS:
            lines = ["s far far_sd atkrecall atkrecall_sd macrof1 macrof1_sd"]
            for s in STRENGTHS:
                vals = []
                for name in ("far", "atkrecall", "macrof1"):
                    v = [] if placeholder else res.metric(model, fset, s, name)
                    v = [x * 100 for x in v if x is not None]
                    if v:
                        vals += [np.mean(v), np.std(v, ddof=1) if len(v) > 1 else 0.0]
                    else:
                        vals += [np.nan, np.nan]
                if placeholder:  # dummy, non-constant so pgfplots axes are not empty
                    vals = [40 - 35 * s, 0.0, 90 - 20 * s, 0.0, 70 + 10 * s, 0.0]
                if any(np.isnan(vals)):
                    continue
                lines.append(f"{s:.1f} " + " ".join(f"{v:.4f}" for v in vals))
            if len(lines) == 1:  # no data for this combination: one dummy row keeps pgfplots happy
                lines.append("0.0 nan 0 nan 0 nan 0")
            open(os.path.join(tex_dir, f"sweep_{mk}_{fset}.dat"), "w").write("\n".join(lines) + "\n")
    lines = ["rank rows"]
    sizes = list(range(10000, 0, -1000)) if (placeholder or cs is None) else list(cs["sizes"])
    lines += [f"{i + 1} {int(v)}" for i, v in enumerate(sizes)]
    open(os.path.join(tex_dir, "clientsizes.dat"), "w").write("\n".join(lines) + "\n")


def fig_sweep(prefix):
    panels = [("far", "False alarm rate (\\%)"), ("atkrecall", "Avg.\\ attack recall (\\%)"),
              ("macrof1", "Macro F1 (\\%)")]
    styles = {"fedcnn": "blue,mark=o", "centralcnn": "red,mark=square*",
              "lgbm": "black!70!green,mark=triangle*"}
    out = ["\\begin{figure*}[t]", "\\centering",
           "\\pgfplotsset{sweepax/.style={scale only axis,width=0.24\\textwidth,height=2.6cm,"
           "xmin=-0.05,xmax=1.05,xtick={0,0.25,0.5,0.75,1},grid=both,"
           "tick label style={font=\\scriptsize},label style={font=\\scriptsize},"
           "title style={font=\\scriptsize,yshift=-4pt}}}"]
    for ri, (fset, fname) in enumerate(FSETS):
        for pi, (col, ylab) in enumerate(panels):
            out.append("\\begin{tikzpicture}[baseline]")
            extra = (",legend to name=sweeplegend,legend columns=3,"
                     "legend style={font=\\scriptsize,draw=none,/tikz/every even column/.append "
                     "style={column sep=6pt}}") if (ri == 0 and pi == 0) else ""
            xl = ",xlabel={Correction strength $s$}" if ri == 1 else ""
            out.append(f"\\begin{{axis}}[sweepax,ylabel={{{ylab}}},title={{{fname} feature set}}"
                       f"{xl}{extra}]")
            for _, mk, mname in MODELS:
                out.append(f"\\addplot[{styles[mk]},thick,mark size=1.3pt,error bars/.cd,"
                           f"y dir=both,y explicit] table[x=s,y={col},y error={col}_sd] "
                           f"{{{prefix}sweep_{mk}_{fset}.dat}};")
                if ri == 0 and pi == 0:
                    out.append(f"\\addlegendentry{{{mname}}}")
            out += ["\\end{axis}", "\\end{tikzpicture}"]
            if pi < 2:
                out.append("\\hfill")
        out.append("\\\\[2pt]")
    out += ["\\ref{sweeplegend}",
            "\\caption{False alarm rate, average attack recall and macro F1 against the "
            "class-prior correction strength $s$ (pooled held-out days). Top: full feature set; "
            "bottom: core feature set. Error bars: standard deviation over seeds.}",
            "\\label{fig:sweep}", "\\end{figure*}"]
    return "\n".join(out) + "\n"


def fig_clients(prefix):
    return "\n".join([
        "\\begin{figure}[t]", "\\centering", "\\begin{tikzpicture}",
        "\\begin{axis}[width=\\columnwidth,height=4.2cm,ymode=log,xlabel={Client rank},"
        "ylabel={Rows per client},grid=both,tick label style={font=\\scriptsize},"
        "label style={font=\\scriptsize}]",
        f"\\addplot[ycomb,blue,thick] table[x=rank,y=rows] {{{prefix}clientsizes.dat}};",
        "\\end{axis}", "\\end{tikzpicture}",
        "\\caption{Rows per client (training plus validation) in the capped sample, sorted in "
        "decreasing order.}", "\\label{fig:clients}", "\\end{figure}"]) + "\n"


# ---------------------------------------------------------------------------
# audit table
# ---------------------------------------------------------------------------

def audit_placeholder():
    return "\n".join([
        "\\begin{table*}[t]", "\\centering", "\\scriptsize",
        "\\caption{Features and Leakage Audit}", "\\label{tab:audit}",
        "\\begin{tabular}{llllll}", "\\toprule",
        "Feature & Causal & Test-day data & Construction artifact & Max AUC & Decision \\\\",
        "\\midrule", "-- & -- & -- & -- & -- & -- \\\\", "\\bottomrule", "\\end{tabular}",
        "\\end{table*}"]) + "\n"


def audit_from_file(path):
    """Copy leakage_audit.py's table, keeping its own caption and label. A bare
    tabular is wrapped in table*; a missing label gets tab:audit."""
    t = open(path).read()
    if not re.search(r"\\begin\{table\*?\}", t):
        t = ("\\begin{table*}[t]\n\\centering\n\\scriptsize\n\\caption{Features and Leakage Audit}"
             "\\label{tab:audit}\n" + t.rstrip() + "\n\\end{table*}\n")
    if "\\label{" not in t:
        m = re.search(r"\\begin\{table\*?\}(\[[^\]]*\])?", t)
        t = t[:m.end()] + "\n\\label{tab:audit}" + t[m.end():]
    return t


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dirs", nargs="*", default=[],
                    help="run_experiments.py --out-dir folder(s); several are merged "
                         "(e.g. results from two machines)")
    ap.add_argument("--v2-runs-dir", nargs="*", default=[],
                    help="run folder(s) produced on the old (v2) labelled files and test sample")
    ap.add_argument("--tex-dir", default=os.path.join(REPO, "paper", "generated"))
    ap.add_argument("--dat-prefix", default="generated/",
                    help="Path of --tex-dir relative to the main .tex file, used for .dat files")
    ap.add_argument("--placeholder", action="store_true")
    ap.add_argument("--audit-table", default=None,
                    help="table written by leakage_audit.py (copied as tab_audit.tex)")
    args = ap.parse_args()
    os.makedirs(args.tex_dir, exist_ok=True)

    res = Results([] if args.placeholder else args.runs_dirs)
    v2 = Results(args.v2_runs_dir) if (args.v2_runs_dir and not args.placeholder) else None
    if not args.placeholder and not res.runs:
        sys.exit("no run JSONs found; pass --runs-dirs or use --placeholder")
    for label, r in (("v3", res), ("v2", v2)):
        if r and r.runs:
            print(f"{label}: {len(r.runs)} run(s), seeds {r.seeds}, days {r.days}")
            for model, _, _ in MODELS:
                for fset, _ in FSETS:
                    n = len(r.group(model, fset))
                    if n:
                        print(f"  {model:<12} {fset:<5} {n} seed(s)")
    M, cs, ntdays = build_macros(res, v2, args.placeholder)
    ndays = max(NDAYS_MIN, len(res.days))
    write_macros(M, os.path.join(args.tex_dir, "results_macros.tex"))
    files = {
        "tab_headline.tex": tab_headline(M),
        "tab_fed_vs_central.tex": tab_fed_vs_central(M),
        "tab_ablation.tex": tab_ablation(M),
        "tab_perclass.tex": tab_perclass(M),
        "tab_perday.tex": tab_perday(M, ndays),
        "tab_clients.tex": tab_clients(M),
        "tab_config.tex": tab_config(M),
        "tab_prevalence.tex": tab_prevalence(M),
        "tab_unseen.tex": tab_unseen(M),
        "tab_mix.tex": tab_mix(M, ntdays),
        "tab_rules.tex": tab_rules(M),
        "tab_confusion.tex": tab_confusion(res, args.placeholder),
        "tab_importance.tex": tab_importance(M),
        "tab_v2v3.tex": tab_v2v3(M),
        "fig_sweep.tex": fig_sweep(args.dat_prefix),
        "fig_clients.tex": fig_clients(args.dat_prefix),
    }
    audit_out = os.path.join(args.tex_dir, "tab_audit.tex")
    if args.audit_table:
        files["tab_audit.tex"] = audit_from_file(args.audit_table)
    elif not os.path.exists(audit_out):
        files["tab_audit.tex"] = audit_placeholder()
    else:
        print(f"keeping existing {audit_out}")
    for name, text in files.items():
        if "\u2014" in text:
            raise ValueError(f"em dash in {name}")
        open(os.path.join(args.tex_dir, name), "w").write(text)
    write_dat(res, args.tex_dir, args.placeholder, cs)
    print(f"wrote {len(M)} macros and {len(files)} tex files + .dat files to {args.tex_dir}")


if __name__ == "__main__":
    main()
