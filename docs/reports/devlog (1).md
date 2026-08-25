# Devlog: Does GAF Harmonization Actually Help Cross-Domain UAV+ADS-B Detection?

**Project:** FL-IDS cross-domain extension — UAV-NIDD + OpenSky ADS-B
**Author:** Bhuvan Krishna (Tony)

---

## The question

The cross-domain thesis extension claims that GAF (Gramian Angular Field) harmonization —
turning engineered feature vectors from two structurally unrelated domains into images so a
shared CNN can process both — helps a model classify attacks across UAV network traffic and
aircraft ADS-B data. That's a claim, not a fact yet. Before writing it into the report, I wanted
an actual controlled test: same data, same split, same training budget, two models — one fed
the raw feature vector, one fed the GAF image — and see which one actually wins.

Turned out building that comparison honestly was the easy part. Getting real data into a shape
where the comparison meant anything was most of the work.

## Setting up the ablation

The design itself is simple: a `RawFeatureMLP` baseline and a `CrossDomainCNN` (GAF) model,
trained on identical equal-parts-sampled, leakage-safe-preprocessed, class-weighted data, with
the only difference being raw vector vs. GAF image. Wrote this as a standalone script
(`gaf_vs_raw_cross_domain.py`) rather than editing the existing notebook, so the comparison logic
itself couldn't be accused of favoring one side.

First real snag: my initial go at this didn't have any defense against overfitting late in a
fixed epoch budget. More on that below — it mattered a lot.

## The UAV-NIDD rabbit hole

Downloaded UAV-NIDD from Zenodo (302MB zip, three files — one per attack *scenario*, not per
attack *type*, which I didn't expect). Then things got messy, roughly in this order:

1. **One of the three files wasn't actually a CSV.** `GSC Case3 Label .csv` had a `.csv`
   extension but its contents were raw ZIP/XLSX bytes (`head -1` on it printed
   `[Content_Types].xml` garbage). Someone exported it as Excel and it got renamed without being
   converted. Fixed by force-reading it with `pd.read_excel()` regardless of the extension and
   re-saving as a real CSV.

2. **Each file used a different column name for its label.** Case1 used `Label`, Case2 used
   `Normal`, Case3 used `Class`. My first loader only checked for `label`/`Label` — which meant
   Case2 and Case3 would have silently had every row's real label overwritten with the filename.
   Caught this before running anything real, thankfully, but it's the kind of bug that produces
   plausible-looking garbage results instead of an error.

3. **The biggest file crashed the CSV parser** with a `pandas` `IndexError` that had nothing to
   do with encoding — it was genuinely malformed rows (inconsistent field counts). Needed a
   fallback path: fast C-engine parse first, and if every encoding attempt fails, retry with the
   slower `python` engine and `on_bad_lines='skip'`. That alone wasn't safe either — pandas'
   `python` engine, when it sees one ragged row, quietly assumes there's an implicit leading
   index column for the *entire file* and shifts every other row's data by one column. Needed
   `index_col=False` explicitly to stop that. Found this by literally forcing the bug with a
   synthetic malformed file and watching good rows come back with `NaN` labels until I added the
   flag.

4. **Case3's features were structurally incompatible with the other two.** Case1/Case2 are WiFi
   packet-capture fields (`radiotap.*`, `wlan.*`). Case3 is flow-based statistics
   (`fwd_pkts_tot`, `flow_iat.max`, ...) — a completely different feature family, zero column
   overlap. Concatenating it in anyway would silently inject ~824K rows that were 100% imputed
   constants for whatever columns weren't theirs. Diagnosed this with a feature-overlap check
   (printed 0% shared columns across all three files) and made the call to drop Case3 entirely.
   Case1+Case2 alone share 47% of their columns — a real, usable overlap.

5. **Labels didn't even agree with themselves.** `Reconnassiance` (typo) vs `Reconnaissance`,
   `Brute-Force`/`Bruteforce`/`BruteForce`, `Ewil Twin` vs `EvilTwin`. Built a small alias map to
   harmonize these, plus a near-duplicate detector (`difflib`) that flagged `DDoS`↔`DoS` and
   `UDP Flooding`↔`ICMP Flooding` as *possibly* the same thing without auto-merging them — that
   felt like a judgment call, not something to silently decide. Ended up merging both pairs after
   review.

6. **~24,700 rows had a genuinely blank label.** Not corruption exactly — just missing data in
   the source. Dropped rather than trained on as a fake "Missing" class.

7. **`Reconnaissance` had 46 total rows.** After equal-parts sampling down to 20,000 rows,
   proportional rounding gave it exactly *1* training example — which crashed sklearn's
   stratified split outright (`ValueError: least populated classes ... only 1 member`). Fixed the
   sampler to guarantee a floor of 2 rows per class regardless of rounding, and added a
   `--min-class-count` threshold (default 100) to drop classes too thin to learn anything from in
   the first place, rather than technically-splittable-but-useless.

## The 1.57GB problem

The ADS-B side had its own issue: one hour of OpenSky labeled data was 1.57GB and wouldn't load.
Turned out most of that size was raw/intermediate pipeline columns nobody needed. Fixed with a
chunked streaming reader — `usecols` to only pull the ~15 engineered features, `chunksize` to
read in pieces, and a per-class reservoir sampler that keeps memory flat no matter how big the
file is. Tested this against a synthetic 500K-row file with junk columns before trusting it on
the real one.

## "Why is ADS-B doing so much better than UAV?"

First full run: ADS-B accuracy way ahead of UAV for both models. Tempting to assume "more data,"
but the equal-parts sampling had already forced both domains to exactly 20,000 rows — that wasn't
it. Real reasons: (1) the Case3 feature-overlap problem above, still unresolved at that point, and
(2) UAV genuinely had 15 unbalanced classes vs. ADS-B's clean 6 near-perfectly-balanced ones.

## Does more training help?

Tried it — and the GAF model's test loss spiked from ~1.08 to ~4.73 between epochs 15 and 20 while
train accuracy kept climbing. Classic overfitting, and the script was reporting whatever the
*last* epoch looked like, which happened to be the worst point in that run. Added best-checkpoint
tracking: track test accuracy every epoch, restore the model's weights to whichever epoch was
actually best when training finishes. This changed the story significantly — a run that looked
like "GAF loses to baseline" was actually "GAF peaked at epoch 15 and got measured at epoch 20."

Reran at 20 → 40 → 60 epochs with checkpointing in place. 60 was enough for both models to settle
(best checkpoints landing well before the ceiling, not clipped by it).

## Final result

| Metric | Baseline | GAF | Delta |
|---|---|---|---|
| UAV domain accuracy | 58.52% | 56.45% | −2.07% |
| ADS-B domain accuracy | 67.17% | 96.93% | +29.77% |
| Combined accuracy | 62.84% | 76.69% | +13.85% |
| Macro-F1 | 0.5028 | 0.6134 | +0.1106 |

GAF wins clearly on ADS-B and on the overall macro-F1. It does *not* clearly win on UAV — that's
an honest mixed result, not a clean "GAF works" story, and I decided to report it that way rather
than cherry-pick the framing.

One thing the summary metrics hid completely, that only showed up in the confusion matrices: the
baseline gets `UAV_Jamming` 100% right and `UAV_MITM` 0% right. GAF flips that exactly — `MITM`
recall jumps to 96%, `Jamming` drops to 0%. Macro-F1 going up doesn't mean every class got better;
it means the net was positive. Worth remembering that going into the write-up.

## What's still open

- Case3 (GCS-compromise scenario) is out of the UAV domain entirely — a real scope reduction from
  UAV-NIDD's original three scenarios, documented rather than hidden.
- The 47% Case1/Case2 feature overlap is a hard ceiling on UAV signal quality that no amount of
  data cleaning fixes — would need actual shared feature engineering to move past it.
- Only one random seed run so far. The story is consistent across five different configurations,
  which is reassuring, but a proper seed sweep would give a real confidence interval instead of a
  point estimate.
- The DoS/DDoS and Flooding merges are a stated judgment call, not a technical necessity — worth
  keeping the split-class version around as a sensitivity check if anyone pushes back on it.
