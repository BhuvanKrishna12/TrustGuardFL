#!/usr/bin/env bash
# Smoke test of the experiment harness on small synthetic data (CPU, about 20 min):
# round-1 checks, then the revision-2 checks (variant equivalence, sampling, protocol v1
# regression, FedProx mu=0, resume after stop and after kill -9, quick runs of every new
# model / variant / protocol path, aggregator, and a simulated full pass of
# run_revision2.sh with stops and resumes).
# Usage: bash experiments/smoke_test.sh [WORK_DIR]      (SMOKE_SKIP_SIM=1 skips the simulation)
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
W="${1:-$(mktemp -d)}"
[ -n "$W" ] && [ "$W" != "/" ] || { echo "empty work dir"; exit 2; }
case "$W" in /*) ;; *) W="$PWD/$W" ;; esac
mkdir -p "$W"
echo "work dir: $W"
cd "$REPO"
python3 experiments/make_synthetic_data.py --out-dir "$W/synth"
python3 experiments/test_sampler.py "$W/sampler" | grep -v "^  "
python3 experiments/check_equivalence.py "$W/synth/train_day1.csv" "$W/synth/train_day2.csv" \
    --max-total-rows 200000 --rounds 3
COMMON=(--train-files "$W/synth/train_day1.csv" "$W/synth/train_day2.csv"
        --holdout "$W/synth/holdout.csv" --max-total-rows 200000 --device cpu)
python3 experiments/run_experiments.py "${COMMON[@]}" --out-dir "$W/runs" --quick
python3 experiments/run_experiments.py "${COMMON[@]}" --out-dir "$W/runs" --quick | grep "already done"
python3 experiments/run_experiments.py --out-dir "$W/runs" --holdout-only --quick \
    --strengths 0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 --device cpu
# stand-in for the v2 comparison runs: legacy sampler, full set, LightGBM and fed CNN
python3 experiments/run_experiments.py "${COMMON[@]}" --out-dir "$W/runs_v2" --quick \
    --legacy-sampler --feature-sets full --models lgbm,fed_cnn
python3 experiments/make_holdout_sample.py "$W/synth/train_day1.csv" --out "$W/hs.csv" >/dev/null

# ---------------------------------------------------------------- revision 2
echo "== revision 2: pipeline variants"
python3 experiments/test_variants.py "$W/variants"
echo "== revision 2: sampling passes"
python3 experiments/test_sampling_mv.py "$W/smv"
MVT="$W/smv/mv_opensky_2019-01-07_labeled.csv"; MVT2="$W/smv/mv_opensky_2019-04-01_labeled.csv"
python3 experiments/make_holdout_sample.py --method hash --frac 0.3 "$MVT2" --out "$W/hold_mv.csv" >/dev/null
python3 experiments/make_holdout_sample.py --method hash --frac 0.3 "$W/smv/v3_opensky_2019-04-01_labeled.csv" \
    --out "$W/hold_v3.csv" >/dev/null
python3 experiments/make_holdout_sample.py --method hash --frac 0.3 "$W/smv/v2_opensky_2019-04-01_labeled.csv" \
    --out "$W/hold_v2.csv" >/dev/null
python3 - "$W" <<'PY'
# the hash sample selects the same rows from the mv, the v3 and the v2 file of a day
import sys
import pandas as pd
w = sys.argv[1]
key = ["icao24", "time", "lat", "lon", "lastposupdate"]
mv, v3, v2 = (pd.read_csv(f"{w}/hold_{k}.csv", low_memory=False) for k in ("mv", "v3", "v2"))
assert len(mv) == len(v3) and (mv[key].astype(str).values == v3[key].astype(str).values).all()
kept = mv[~mv["legacy_filter_drop"].astype(str).str.lower().eq("true")]
assert len(kept) == len(v2) and (kept[key].astype(str).values == v2[key].astype(str).values).all()
assert (kept["attack_type_legacy"].values == v2["attack_type"].values).all()
print(f"HASH TEST SAMPLE OK: same {len(mv):,} rows from the mv and v3 files; v2 file gives the "
      f"{len(v2):,} rows not removed by the legacy filter, with the legacy labels")
PY
echo "== revision 2: protocol v1 regression"
python3 experiments/test_protocol_v1_regression.py "$W/smv/v3_opensky_2019-01-07_labeled.csv" "$MVT" \
    "$W/hold_v3.csv" "$W/hold_mv.csv" "$W/reg"
echo "== revision 2: FedProx mu=0 and resume (stop, kill -9)"
python3 experiments/test_fedprox_resume.py "$MVT" "$W/hold_mv.csv" "$W/fr"
echo "== revision 2: quick runs of every model / variant / protocol path"
R2=(--train-files "$MVT" "$MVT2" --holdout "$W/hold_mv.csv" --cache-dir "$W/r2/cache"
    --max-total-rows 60000 --device cpu --quick --min-client-rows 1000 --seeds 1 --feature-sets full)
python3 experiments/run_experiments.py "${R2[@]}" --out-dir "$W/r2/main" \
    --models lgbm,central_cnn,fed_cnn,central_mlp,fed_mlp | grep "<<<"
for v in legacy prefloor loo:C1 loo:C2 loo:C3 loo:C5 add:C1 add:C2 add:C3 add:C5; do
  python3 experiments/run_experiments.py "${R2[@]}" --variant "$v" --out-dir "$W/r2/var_${v/:/_}" \
      --models lgbm,central_cnn | grep -c "<<<" | sed "s/^/  $v: runs finished: /"
done
python3 experiments/run_experiments.py "${R2[@]}" --variant loo:C3 --out-dir "$W/r2/var_loo_C3" \
    --models fed_cnn | grep -c "<<<" >/dev/null
for v in corrected legacy; do
  python3 experiments/run_experiments.py "${R2[@]}" --variant "$v" --legacy-sampler \
      --out-dir "$W/r2/ls_$v" --models lgbm | grep -c "<<<" >/dev/null
done
python3 experiments/run_experiments.py "${R2[@]}" --fedprox-mu 0.01 --out-dir "$W/r2/fedprox_mu0.01" \
    --models fed_cnn | grep -c "<<<" >/dev/null
python3 experiments/run_experiments.py "${R2[@]}" --protocol v1 --out-dir "$W/r2/v1" --models lgbm | grep -c "<<<" >/dev/null
python3 experiments/run_experiments.py --out-dir "$W/r2/var_legacy" --cache-dir "$W/r2/cache" --holdout-only \
    --quick --seeds 1 --feature-sets full --models lgbm --device cpu | tail -1
python3 - "$W/r2" <<'PY'
import glob, json, sys
n = 0
for p in glob.glob(sys.argv[1] + "/*/runs/*__*__seed*.json"):
    r = json.load(open(p))
    for k in ("variant", "protocol", "s_star", "selection", "pr_auc", "false_alerts", "per_client_test",
              "test_summary", "metrics"):
        assert k in r, (p, k)
    assert r["s_star"] in [i / 10 for i in range(11)]
    assert f"{r['s_star']:.2f}" in r["metrics"]
    if r["variant"] != "corrected":
        assert "metrics_corrected_test" in r, p
    if r["protocol"] == "v2":
        assert r["data"]["split_report"]["split"].startswith("aircraft-disjoint")
    n += 1
print(f"RUN JSON CHECK OK: {n} revision-2 run files carry variant, protocol, s*, validation "
      f"metrics, PR-AUC, false-alert counts and per-client test metrics")
PY
echo "== revision 2: run_revision2.sh"
bash -n "$REPO/run_revision2.sh"
REV2_W="$W/dry" REV2_CODE="$REPO" bash "$REPO/run_revision2.sh" dry-run > "$W/dry_run.txt"
echo "  dry-run lists $(grep -c run_experiments.py "$W/dry_run.txt") harness commands"
if [ "${SMOKE_SKIP_SIM:-0}" != "1" ]; then
  bash experiments/test_run_revision2.sh "$W/sim" | tail -12
  R2ROOT="$W/sim/W/runs"
else
  R2ROOT="$W/r2"
fi

mkdir -p "$W/tex/generated"
python3 experiments/aggregate_results.py --runs-dirs "$W/runs" --v2-runs-dir "$W/runs_v2" \
    --rev2-root "$R2ROOT" --tex-dir "$W/tex/generated"
mkdir -p "$W/texph/generated"
python3 experiments/aggregate_results.py --placeholder --tex-dir "$W/texph/generated"
for d in tex texph; do
cat > "$W/$d/test.tex" <<'EOF'
\documentclass[conference]{IEEEtran}
\usepackage{amsmath,amssymb,booktabs,tabularx,array,graphicx,pgfplots}
\input{generated/results_macros.tex}
\begin{document}
Macro F1 \R{fedcnn-full-s10-macrof1}\%, \R{nclients} clients, \R{commbytes},
overlap \R{overlap-rows}\%, gap \R{gap-full-s10-macrof1-mean} \R{gap-full-s10-macrof1-ci}.
\input{generated/tab_headline.tex}\input{generated/tab_fed_vs_central.tex}
\input{generated/tab_ablation.tex}\input{generated/tab_perclass.tex}
\input{generated/tab_perday.tex}\input{generated/tab_clients.tex}
\input{generated/tab_config.tex}\input{generated/tab_audit.tex}
\input{generated/tab_prevalence.tex}\input{generated/tab_unseen.tex}
\input{generated/tab_mix.tex}\input{generated/tab_rules.tex}
\input{generated/tab_confusion.tex}\input{generated/tab_importance.tex}
\input{generated/tab_v2v3.tex}
\input{generated/tab_ablation_corrections.tex}\input{generated/tab_unified_pipeline.tex}
\input{generated/tab_selected_s.tex}\input{generated/tab_baselines.tex}
\input{generated/tab_prauc.tex}
s star \R{r2-sstar-fedcnn-full-mean}, \R{r2-unified-diff-lgbm-own}.
\input{generated/fig_sweep.tex}\input{generated/fig_clients.tex}
\end{document}
EOF
if command -v pdflatex >/dev/null; then
  if (cd "$W/$d" && pdflatex -interaction=nonstopmode -halt-on-error test.tex >/dev/null 2>&1 \
      && pdflatex -interaction=nonstopmode -halt-on-error test.tex >/dev/null 2>&1); then
    echo "LaTeX compile check OK for $d"
  else
    # The compile is an optional bonus check. It needs the IEEEtran class and some
    # LaTeX packages, which a data/ML environment may not have. Not having them does
    # not mean the revision code is wrong, so warn and carry on.
    echo "note: optional LaTeX compile check skipped for $d (pdflatex is present but a LaTeX class or package such as IEEEtran is missing); this does not affect the results"
  fi
else
  echo "pdflatex not installed: skipping the LaTeX compile check for $d"
fi
done
for f in tab_ablation_corrections tab_unified_pipeline tab_selected_s tab_baselines tab_prauc; do
  [ -s "$W/tex/generated/$f.tex" ] && [ -s "$W/texph/generated/$f.tex" ]
  ! grep -q "IEEE" "$W/tex/generated/$f.tex"
done
grep -q "res@r2-sstar-fedcnn-full-mean" "$W/tex/generated/results_macros.tex"
echo "SMOKE TEST OK: generated tables in $W/tex and $W/texph"
