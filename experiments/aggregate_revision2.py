r"""
aggregate_revision2.py
-------------------------------------------------------------------
Revision-2 tables and macros, built from the run folders written by
run_revision2.sh (<root>/main, var_<variant>, ls_<variant>, fedprox_mu<mu>).
Called by aggregate_results.py --rev2-root <root> (also in --placeholder mode).

  tab_ablation_corrections.tex  one correction at a time (leave-one-out and add-one),
                                the sampler, and the reconstructed pre-floor extra row
  tab_unified_pipeline.tex      earlier (legacy) vs corrected pipeline, one protocol,
                                same test rows, paired differences
  tab_selected_s.tex            validation-selected correction strength s*
  tab_baselines.tex             MLP / CNN central and federated, LightGBM, FedProx,
                                paired differences with 95% t-intervals
  tab_prauc.tex                 PR-AUC (average precision) per class
Plain booktabs tabulars; no IEEE-specific macros. Macro keys start with r2-.
"""

import glob
import json
import os

import numpy as np

DASH = "--"
PM = "{\\,}$\\pm${\\,}"
MODELS = [("lgbm", "lgbm", "LightGBM"), ("central_cnn", "centralcnn", "Central CNN"),
          ("fed_cnn", "fedcnn", "Fed.\\ CNN"), ("central_mlp", "centralmlp", "Central MLP"),
          ("fed_mlp", "fedmlp", "Fed.\\ MLP")]
MK = {m[0]: m[1] for m in MODELS}
MN = {m[0]: m[2] for m in MODELS}
ABL = ["C1", "C2", "C3", "C5"]
CDESC = {"C1": "causal message count", "C2": "squawk change needs both codes",
         "C3": "no benign-only speed filter", "C5": "flooding originals benign"}
CLASSES = [("benign", "Benign"), ("flooding", "Flooding"), ("ghost_aircraft", "Ghost"),
           ("gps_spoof", "GPS"), ("replay", "Replay"), ("velocity_spoof", "Velocity")]
PREFLOOR_NOTE = ("\\emph{Pre-floor}: the earlier pipeline without the 0.1\\,s rate-denominator "
                 "floor, reconstructed from the documented rule (rate features missing when the "
                 "gap is below 1\\,s), not from the original code")


def t975(df):
    try:
        from scipy.stats import t
        return float(t.ppf(0.975, df))
    except Exception:
        return {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}.get(df, 1.96)


def fmt(vals, scale=100.0, dec=1):
    vals = [v for v in vals if v is not None]
    if not vals:
        return DASH
    a = np.array(vals, float) * scale
    if len(a) == 1:
        return f"{a[0]:.{dec}f}"
    return f"{a.mean():.{dec}f}{PM}{a.std(ddof=1):.{dec}f}"


def fmt_diff(d, dec=1):
    """paired differences (already in points): mean [95% t-interval]."""
    d = [v for v in d if v is not None]
    if not d:
        return DASH
    a = np.array(d, float)
    m = a.mean()
    sm = f"${m:+.{dec}f}$"
    if len(a) < 2:
        return sm
    h = t975(len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a))
    return f"{sm} [${m - h:+.{dec}f}$, ${m + h:+.{dec}f}$]"


class R2:
    def __init__(self, root):
        self.runs = {}
        if root:
            for p in sorted(glob.glob(os.path.join(root, "*", "runs", "*__*__seed*.json"))):
                d = os.path.basename(os.path.dirname(os.path.dirname(p)))
                try:
                    r = json.load(open(p))
                except Exception:
                    continue
                self.runs[(d, r["model"], r["feature_set"], int(r["seed"]))] = r

    def by_seed(self, d, model, fset="full"):
        return {k[3]: r for k, r in self.runs.items() if k[:3] == (d, model, fset)}

    def vals(self, d, model, getter, fset="full"):
        out = []
        for _, r in sorted(self.by_seed(d, model, fset).items()):
            try:
                out.append(getter(r))
            except (KeyError, TypeError):
                pass
        return out

    def paired(self, a, b, getter, scale=100.0):
        """a, b: (dir, model, fset). Returns per-seed getter(a) - getter(b)."""
        A, B = self.by_seed(*a), self.by_seed(*b)
        out = []
        for s in sorted(set(A) & set(B)):
            try:
                out.append((getter(A[s]) - getter(B[s])) * scale)
            except (KeyError, TypeError):
                pass
        return out


def g_metric(name, s=1.0, test="own"):
    key = "metrics" if test == "own" else "metrics_corrected_test"

    def get(r):
        m = r.get(key) if key in r else (r["metrics"] if r.get("variant") == "corrected" else None)
        return m[f"{s:.2f}"]["pooled"][name]
    return get


def g_at_star(name, test="own"):
    key = "metrics" if test == "own" else "metrics_corrected_test"

    def get(r):
        m = r.get(key) if key in r else (r["metrics"] if r.get("variant") == "corrected" else None)
        return m[f"{r['s_star']:.2f}"]["pooled"][name]
    return get


def table(env, caption, label, colspec, header_rows, body_rows, note=None, size="\\scriptsize",
          tabcolsep="3pt"):
    out = [f"\\begin{{{env}}}[t]", "\\centering", size, f"\\caption{{{caption}}}",
           f"\\label{{{label}}}", f"\\setlength{{\\tabcolsep}}{{{tabcolsep}}}",
           f"\\begin{{tabular}}{{{colspec}}}", "\\toprule"] + header_rows + ["\\midrule"] \
        + body_rows + ["\\bottomrule", "\\end{tabular}"]
    if note:
        out += ["\\\\[2pt]", f"\\parbox{{\\linewidth}}{{\\scriptsize {note}}}"]
    out.append(f"\\end{{{env}}}")
    return "\n".join(out) + "\n"


def vdir(variant):
    return "main" if variant == "corrected" else "var_" + variant.replace(":", "_")


def build(root, placeholder=False):
    """Returns (files: {name: tex}, macros: {key: value})."""
    R = R2(None if placeholder else root)
    M = {}

    def put(k, v):
        M[k] = DASH if placeholder or v is None else v

    f1 = g_metric("macro_f1")
    abl_models = ["lgbm", "central_cnn", "fed_cnn"]

    # ---------------- ablation ----------------
    def abl_cells(d, ref_pair=None, sign=1.0, key=None):
        cells = []
        for m in abl_models:
            own = fmt(R.vals(d, m, g_metric("macro_f1", 1.0, "own")))
            cor = fmt(R.vals(d, m, g_metric("macro_f1", 1.0, "corr")))
            cells += [own, cor]
            if key:
                put(f"r2-abl-{key}-{MK[m]}-own-s10-macrof1", own)
                put(f"r2-abl-{key}-{MK[m]}-corr-s10-macrof1", cor)
        for m in abl_models[:2]:
            if ref_pair is None:
                eff = ""
            else:
                a, b = ref_pair
                dd = R.paired((a, m, "full"), (b, m, "full"), g_metric("macro_f1", 1.0, "own"))
                eff = fmt_diff([sign * v for v in dd])
                dc = R.paired((a, m, "full"), (b, m, "full"), g_metric("macro_f1", 1.0, "corr"))
                if key:
                    put(f"r2-abl-{key}-{MK[m]}-effect", eff)
                    put(f"r2-abl-{key}-{MK[m]}-effect-corrtest", fmt_diff([sign * v for v in dc]))
            cells.append(eff)
        return " & ".join(cells)

    rows = [f"Corrected pipeline (all corrections) & {abl_cells('main', key='corrected')} \\\\",
            f"Earlier pipeline (none of C1, C2, C3, C5) & "
            f"{abl_cells('var_legacy', ('main', 'var_legacy'), key='legacy')} \\\\",
            "\\midrule",
            "\\multicolumn{9}{l}{\\emph{Leave one out: all corrected except the named correction}} \\\\"]
    for c in ABL:
        d = f"var_loo_{c}"
        rows.append(f"\\quad without {c} ({CDESC[c]}) & {abl_cells(d, ('main', d), key='loo-' + c.lower())} \\\\")
    rows += ["\\midrule",
             "\\multicolumn{9}{l}{\\emph{Add one: earlier pipeline plus the named correction}} \\\\"]
    for c in ABL:
        d = f"var_add_{c}"
        rows.append(f"\\quad only {c} ({CDESC[c]}) & {abl_cells(d, (d, 'var_legacy'), key='add-' + c.lower())} \\\\")
    rows += ["\\midrule", "\\multicolumn{9}{l}{\\emph{Sampler S: legacy loader instead of the uniform capped sampler}} \\\\",
             f"\\quad corrected pipeline, legacy loader & {abl_cells('ls_corrected', ('main', 'ls_corrected'), key='ls-corrected')} \\\\",
             f"\\quad earlier pipeline, legacy loader & {abl_cells('ls_legacy', ('var_legacy', 'ls_legacy'), key='ls-legacy')} \\\\",
             "\\midrule",
             f"Pre-floor (reconstructed; extra) & {abl_cells('var_prefloor', ('var_legacy', 'var_prefloor'), key='prefloor')} \\\\"]
    files = {}
    files["tab_ablation_corrections.tex"] = table(
        "table*", "Pipeline Corrections One at a Time (Macro F1 at $s=1$, Full Feature Set)",
        "tab:ablationcorrections", "l" + "r" * 8,
        ["& \\multicolumn{2}{c}{LightGBM} & \\multicolumn{2}{c}{Central CNN} & "
         "\\multicolumn{2}{c}{Fed.\\ CNN} & \\multicolumn{2}{c}{Effect (own test)} \\\\",
         "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}",
         "Training data variant & own test & corr.\\ test & own test & corr.\\ test & own test & "
         "corr.\\ test & LightGBM & Central CNN \\\\"], rows,
        note="Mean $\\pm$ standard deviation over seeds, in \\%. Own test: the test sample built "
             "like the training data of that row; corr.\\ test: the fully corrected test sample "
             "(same sampled rows; rows removed by the earlier filter exist only there). Effect: "
             "paired difference per seed in percentage points, each side on its own test sample, mean and "
             "95\\% $t$-interval; for the earlier pipeline row: corrected minus earlier; leave-one-out "
             "rows: corrected minus row (what the correction contributes last); add-one rows: row "
             "minus earlier pipeline (what it contributes first); sampler rows: uniform minus legacy "
             "loader; pre-floor row: earlier pipeline minus pre-floor. The federated CNN was run for "
             "the corrected, earlier and leave-one-out rows only. " + PREFLOOR_NOTE + ".")

    # ---------------- unified ----------------
    rows = []
    for mi, m in enumerate(abl_models):
        if mi:
            rows.append("\\midrule")
        for label, d in (("earlier", "var_legacy"), ("corrected", "main")):
            c = [fmt(R.vals(d, m, g_metric("macro_f1", 1.0, "own"))),
                 fmt(R.vals(d, m, g_metric("macro_f1", 1.0, "corr"))),
                 fmt(R.vals(d, m, g_at_star("macro_f1", "own"))),
                 fmt(R.vals(d, m, g_metric("false_alarm_rate", 1.0, "own"))),
                 fmt(R.vals(d, m, g_metric("avg_attack_recall", 1.0, "own")))]
            k = f"r2-unified-{'legacy' if d != 'main' else 'corrected'}-{MK[m]}"
            for name, v in zip(("own-s10-macrof1", "corr-s10-macrof1", "own-sstar-macrof1",
                                "own-s10-far", "own-s10-atkrecall"), c):
                put(f"{k}-{name}", v)
            put(f"{k}-nseeds", str(len(R.by_seed(d, m))) if R.by_seed(d, m) else None)
            rows.append(f"{MN[m] if label == 'earlier' else ''} & {label} & " + " & ".join(c) + " \\\\")
        d_own = fmt_diff(R.paired(("main", m, "full"), ("var_legacy", m, "full"), g_metric("macro_f1", 1.0, "own")))
        d_cor = fmt_diff(R.paired(("main", m, "full"), ("var_legacy", m, "full"), g_metric("macro_f1", 1.0, "corr")))
        put(f"r2-unified-diff-{MK[m]}-own", d_own)
        put(f"r2-unified-diff-{MK[m]}-corr", d_cor)
        rows.append(f" & \\emph{{corr.$-$earlier}} & {d_own} & {d_cor} & & & \\\\")
    files["tab_unified_pipeline.tex"] = table(
        "table*", "Earlier versus Corrected Pipeline under One Protocol (Full Feature Set)",
        "tab:unifiedpipeline", "llrrrrr",
        ["Model & Pipeline & Macro F1, own test & Macro F1, corr.\\ test & Macro F1 at $s^{*}$ & "
         "FAR & Atk.\\ rec. \\\\"], rows,
        note="Mean $\\pm$ standard deviation over seeds, in \\%; $s=1$ unless stated. Both "
             "pipelines come from one multi-variant reprocessing of the same raw data, the same "
             "sampler, the same protocol (aircraft-disjoint validation, statistics fitted on training "
             "rows) and the same hash-selected test rows. Own test: each pipeline scored on its own "
             "version of the test sample. corr.$-$earlier: paired difference per seed in percentage "
             "points with 95\\% $t$-interval.")

    # ---------------- selected s ----------------
    rows = []
    for fi, fset in enumerate(("full", "core")):
        if fi:
            rows.append("\\midrule")
        for mi, m in enumerate(abl_models):
            runs = [r for _, r in sorted(R.by_seed("main", m, fset).items())]
            ss = [r["s_star"] for r in runs if r.get("s_star") is not None]
            sv = (f"{np.mean(ss):.2f} ({', '.join(f'{v:g}' for v in ss)})" if ss else DASH)
            topt = [r["test_summary"]["test_optimal_s_observation_only"]["s"] for r in runs
                    if r.get("test_summary")]
            c = [sv,
                 fmt([r["selection"]["val_macro_f1_at_s_star"] for r in runs if r.get("selection")]),
                 fmt(R.vals("main", m, g_at_star("macro_f1"), fset)),
                 fmt(R.vals("main", m, g_metric("macro_f1", 0.0), fset)),
                 fmt(R.vals("main", m, g_metric("macro_f1", 0.5), fset)),
                 fmt(R.vals("main", m, g_metric("macro_f1", 1.0), fset)),
                 (f"{np.mean(topt):.2f}" if topt else DASH)]
            base = f"r2-sstar-{MK[m]}-{fset}"
            for name, v in zip(("values", "val-macrof1", "test-macrof1", "test-s00-macrof1",
                                "test-s05-macrof1", "test-s10-macrof1", "testopt"), c):
                put(f"{base}-{name}", v)
            put(f"{base}-mean", f"{np.mean(ss):.2f}" if ss else None)
            put(f"{base}-far", fmt(R.vals("main", m, g_at_star("false_alarm_rate"), fset)))
            put(f"{base}-atkrecall", fmt(R.vals("main", m, g_at_star("avg_attack_recall"), fset)))
            rows.append(f"{'Full' if fset == 'full' and mi == 0 else ('Core' if mi == 0 else '')} & "
                        f"{MN[m]} & " + " & ".join(c) + " \\\\")
    files["tab_selected_s.tex"] = table(
        "table*", "Correction Strength Selected on Validation Data", "tab:selecteds", "llrrrrrrr",
        ["& & & Val.\\ F1 & \\multicolumn{4}{c}{Test macro F1 at} & Test-opt. \\\\",
         "\\cmidrule(lr){5-8}",
         "Features & Model & $s^{*}$ (per seed) & at $s^{*}$ & $s^{*}$ & $s=0$ & $s=0.5$ & $s=1$ & $s$ \\\\"],
        rows,
        note="$s^{*}$: the strength in $\\{0,0.1,\\dots,1\\}$ with the highest macro F1 on the "
             "validation rows (aircraft disjoint from the training rows) reweighted to the natural "
             "class prior of the training days; ties go to the smaller $s$; fixed for each run "
             "before the test sample is scored. Mean $\\pm$ standard deviation over seeds, in \\%. "
             "Test-opt.\\ $s$: the strength that would have been best on the test sample, mean over "
             "seeds, shown as an observation only.")

    # ---------------- baselines ----------------
    entries = [("LightGBM", "main", "lgbm", "lgbm"), ("Central CNN", "main", "central_cnn", "centralcnn"),
               ("Fed.\\ CNN (FedAvg)", "main", "fed_cnn", "fedcnn"),
               ("Central MLP", "main", "central_mlp", "centralmlp"),
               ("Fed.\\ MLP (FedAvg)", "main", "fed_mlp", "fedmlp"),
               ("Fed.\\ CNN (FedProx, $\\mu=0.01$)", "fedprox_mu0.01", "fed_cnn", "fedprox001"),
               ("Fed.\\ CNN (FedProx, $\\mu=0.1$)", "fedprox_mu0.1", "fed_cnn", "fedprox01")]
    rows = []
    for name, d, m, k in entries:
        npar = R.vals(d, m, lambda r: r["training"].get("n_params"))
        npar = f"{int(npar[0]):,}".replace(",", "{,}") if npar and npar[0] else DASH
        c = [npar, fmt(R.vals(d, m, g_at_star("macro_f1"))), fmt(R.vals(d, m, f1)),
             fmt(R.vals(d, m, g_at_star("false_alarm_rate"))),
             fmt(R.vals(d, m, g_at_star("avg_attack_recall"))),
             fmt(R.vals(d, m, lambda r: r["pr_auc"]["macro"]))]
        for nm, v in zip(("nparams", "sstar-macrof1", "s10-macrof1", "sstar-far", "sstar-atkrecall",
                          "prauc-macro"), c):
            put(f"r2-base-{k}-{nm}", v)
        put(f"r2-base-{k}-nseeds", str(len(R.by_seed(d, m))) if R.by_seed(d, m) else None)
        rows.append(f"{name} & " + " & ".join(c) + " \\\\")
    rows.append("\\midrule")
    diffs = [("Fed.$-$central, CNN", ("main", "fed_cnn", "full"), ("main", "central_cnn", "full"), "cnn"),
             ("Fed.$-$central, MLP", ("main", "fed_mlp", "full"), ("main", "central_mlp", "full"), "mlp"),
             ("FedProx $\\mu=0.01$ $-$ FedAvg", ("fedprox_mu0.01", "fed_cnn", "full"), ("main", "fed_cnn", "full"), "prox001"),
             ("FedProx $\\mu=0.1$ $-$ FedAvg", ("fedprox_mu0.1", "fed_cnn", "full"), ("main", "fed_cnn", "full"), "prox01"),
             ("CNN$-$MLP, central", ("main", "central_cnn", "full"), ("main", "central_mlp", "full"), "archcentral"),
             ("CNN$-$MLP, federated", ("main", "fed_cnn", "full"), ("main", "fed_mlp", "full"), "archfed")]
    for name, a, b, k in diffs:
        c = [fmt_diff(R.paired(a, b, g_at_star("macro_f1"))), fmt_diff(R.paired(a, b, f1)),
             fmt_diff(R.paired(a, b, g_at_star("false_alarm_rate"))),
             fmt_diff(R.paired(a, b, g_at_star("avg_attack_recall"))),
             fmt_diff(R.paired(a, b, lambda r: r["pr_auc"]["macro"]))]
        for nm, v in zip(("sstar-macrof1", "s10-macrof1", "sstar-far", "sstar-atkrecall", "prauc-macro"), c):
            put(f"r2-diff-{k}-{nm}", v)
        rows.append(f"\\emph{{{name}}} & & " + " & ".join(c) + " \\\\")
    files["tab_baselines.tex"] = table(
        "table*", "Baselines on the Corrected Pipeline (Full Feature Set)", "tab:baselines", "lrrrrrr",
        ["Model & Parameters & Macro F1 at $s^{*}$ & Macro F1 at $s=1$ & FAR at $s^{*}$ & "
         "Atk.\\ rec.\\ at $s^{*}$ & PR-AUC \\\\"], rows,
        note="Mean $\\pm$ standard deviation over seeds, in \\%. The MLP (13--190--44--6) has the "
             "same number of parameters as the CNN and is trained through the same centralized and "
             "FedAvg code paths. PR-AUC: macro average over the six classes of the one-vs-rest "
             "average precision. Lower block: paired differences per seed in percentage points, "
             "mean and 95\\% $t$-interval over the seeds both runs share.")

    # ---------------- PR-AUC ----------------
    rows = []
    for name, d, m, k in entries:
        c = [fmt(R.vals(d, m, lambda r, cc=cls: r["pr_auc"]["per_class"][cc])) for cls, _ in CLASSES]
        c.append(fmt(R.vals(d, m, lambda r: r["pr_auc"]["macro"])))
        for (cls, _), v in zip(CLASSES + [("macro", "")], c):
            put(f"r2-prauc-{k}-{cls.replace('_', '-')}", v)
        rows.append(f"{name} & " + " & ".join(c) + " \\\\")
    files["tab_prauc.tex"] = table(
        "table*", "PR-AUC per Class on the Test Sample (Full Feature Set)", "tab:prauc", "l" + "r" * 7,
        ["Model & " + " & ".join(n for _, n in CLASSES) + " & Macro \\\\"], rows,
        note="One-vs-rest average precision of the class probability, from the saved test "
             "probabilities; mean $\\pm$ standard deviation over seeds, in \\%. The value does not "
             "depend on the correction strength, because a per-class factor does not change a "
             "one-vs-rest ranking. The test sample keeps the natural class mix of the generator.")

    # split report macros
    any_main = next((r for k, r in sorted(R.runs.items()) if k[0] == "main"), None)
    sr = (any_main or {}).get("data", {}).get("split_report") or {}
    put("r2-val-aircraft", f"{sr['aircraft_in_validation']:,}".replace(",", "{,}") if sr else None)
    put("r2-val-rowshare", f"{sr['val_row_share'] * 100:.1f}" if sr else None)
    put("r2-val-clients-losing-class-train", str(sr["clients_losing_a_class_in_train"]) if sr else None)
    put("r2-val-clients-losing-class-val", str(sr["clients_losing_a_class_in_val"]) if sr else None)
    put("r2-nruns", str(len(R.runs)) if R.runs else None)
    return files, M
