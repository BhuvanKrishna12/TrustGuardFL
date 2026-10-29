#!/usr/bin/env bash
# Smoke test of the experiment harness on small synthetic data (CPU, about 5 min).
# Usage: bash experiments/smoke_test.sh [WORK_DIR]
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
W="${1:-$(mktemp -d)}"
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
mkdir -p "$W/tex/generated"
python3 experiments/aggregate_results.py --runs-dirs "$W/runs" --v2-runs-dir "$W/runs_v2" \
    --tex-dir "$W/tex/generated"
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
\input{generated/fig_sweep.tex}\input{generated/fig_clients.tex}
\end{document}
EOF
if command -v pdflatex >/dev/null; then
(cd "$W/$d" && pdflatex -interaction=nonstopmode -halt-on-error test.tex >/dev/null \
    && pdflatex -interaction=nonstopmode -halt-on-error test.tex >/dev/null)
else
  echo "pdflatex not installed: skipping the LaTeX compile check for $d"
fi
done
echo "SMOKE TEST OK: generated tables in $W/tex and $W/texph"
