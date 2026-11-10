# Citation verification notes: AeroGuard (FL-IDS on ADS-B)

Verification date: 2026-10-27. Sources: only pages actually opened with WebFetch (listed in Section 6). Bash access to Crossref was blocked by the network proxy, so Crossref was used only where a Crossref URL surfaced in search. Publisher pages for IET/Wiley, ACM DL, IEEE Xplore, dblp and PMC refused automated access; equivalent records (author repositories, ResearchGate, mlanthology, PMLR, arXiv, USENIX, JAIR, institutional repositories) were used instead.

All 40 entries in `refs.bib` were verified this way. Where an opened page did not show a field (DOI, pages, volume, given names), the field is left out and a comment in `refs.bib` says so.

Confidence levels: **High** means an opened page directly supports the claim. **Medium** means the source fits the claim, but the supporting sentence was not quoted, or support comes from an abstract or summary only. **Low/None** means the source does not support the claim, or none was found.

---

## 1. TODO notes in `main_original.tex` mapped to references

There are 33 TODO notes in total: 24 concern the literature and 9 concern the authors' own code or data. The revised `main.tex` already cites the keys listed in the coordinator's message. All of those keys verified, with the identity notes in Section 1b.

### 1a. Literature TODOs

| # | Line | TODO (quoted) | Recommended keys | Claim supported | Confidence |
|---|---|---|---|---|---|
| 1 | 47 | `CITE: ADS-B deployment mandates, 1090 MHz Extended Squitter, and protocol security properties` | strohmeier2014, strohmeier2015, faa91225 | strohmeier2014 (IEEE Commun. Mag.) describes ADS-B/1090ES, says "ADS-B messages are sent unencrypted", and notes that the FAA and EUROCONTROL mandated deployment. faa91225 (14 CFR 91.225) requires 1090ES (TSO-C166b / DO-260B) or UAT equipment "After January 1, 2020" in listed US airspace. strohmeier2015 states ADS-B "has not been developed with security in mind". | High (security, 1090ES, US mandate). The EU mandate has no verified source yet. |
| 2 | 51 | `CITE: aviation surveillance data-sharing and regulatory constraints` | li2020 (partial) | The li2020 abstract says real ADS-B attack data are hard to obtain because of aviation regulations. **No source found** for the claim that pooling surveillance feeds raises commercial or regulatory concerns. | Low. Rephrase or leave uncited as motivation (Section 5). |
| 3 | 51 | `CITE: OpenSky data provenance and ADS-B attack-data availability` | schafer2014, li2020 | schafer2014: "OpenSky is a participatory sensor network of ADS-B sensors"; data "accessible to the volunteers ... and to anyone else on request". li2020: real attack data are hard to obtain. | High for provenance and scarcity. schafer2014 does **not** document the weekly Monday state-vector files (Section 5). |
| 4 | 55 | `CITE: ADS-B authentication deployment and residual threat models` | strohmeier2015, ngamboe2025cabba | strohmeier2015 surveys countermeasures. ngamboe2025cabba (IJCIP 2025) proposes a backward-compatible authenticated ADS-B broadcast protocol, which shows that authentication is still a research proposal. | Medium. The claim that deployment "is not immediate across long-lived infrastructure" has no quoted sentence behind it. |
| 5 | 67 | `CITE: ADS-B message format and threat taxonomy` | strohmeier2015, schafer2013, costin2012 | strohmeier2015 (arXiv version read) lists eavesdropping, jamming, message injection (ghost targets, flooding), message deletion, message modification (trajectory modification) and aircraft spoofing. schafer2013 shows the attacks are inexpensive and highly successful in practice. costin2012 shows eavesdropping, jamming, replay and injection with USRP/GNU Radio. | High. **Meaconing** does not appear in that taxonomy. main.tex already uses "jamming" instead, which is correct. |
| 6 | 73 | `CITE: horizontal federated learning and privacy limitations` | yang2019, zhu2019, bonawitz2017, abadi2016, zhao2018, li2020fedprox | Horizontal FL definition (yang2019). Gradient leakage (zhu2019). Secure aggregation (bonawitz2017). DP-SGD (abadi2016). Non-IID degradation (zhao2018, li2020fedprox). | High for identity and topic. Specific claims were not quote-checked. |
| 7 | 79 | `CITE: synthetic attack generation, GANs, SMOTE, and synthetic data in network IDS` | chawla2002, goodfellow2014, li2020, ahmed2025 | SMOTE and GANs as minority-class synthesis. li2020 is an ADS-B attack-data construction strategy. ahmed2025 used the OpenScope simulator to generate malicious ADS-B messages. | High for SMOTE/GAN/ADS-B. No *network-IDS-specific* GAN/SMOTE paper was verified. |
| 8 | 87 | `CITE: ADS-B LSTM and autoencoder detectors` | habler2018, wang2020, fried2021 | LSTM encoder-decoder (habler2018), LSTM predictor with residual threshold (wang2020), autoencoder classifier (fried2021). | High |
| 9 | 87 | `CITE: SODA-style DNN ADS-B detectors` | ying2019 | ying2019 *is* SODA, a two-stage DNN on physical-layer features. | High |
| 10 | 87 | `CITE: representative centralized ADS-B anomaly-detection studies and dataset limitations` | ahmed2025, cevik2025, ouldslimane2022, khoei2024, li2019ijcip, ngamboe2025 | All are centralized and all use synthetic or simulated attacks (Section 4). | High |
| 11 | 93 | `CITE: federated IDS papers in IoT, vehicular networks, and smart grids` | khraisat2024 (survey), nguyen2019diot (IoT), ruzafa2023 (IIoT), abdelbasset2022 (smart transportation/vehicular), kesici2024 (smart grid) | Representative FL-IDS in each domain. kesici2024 uses *vertical* FL. | High for identity |
| 12 | 97 | `VERIFY: verify Li et al. attack mechanisms, 300-s replay delay, 360-s denial-of-service window, and 0.05\% injection rate against the paper` | li2020 | Mechanisms partly confirmed: the authors' code repository for this paper has modules for data manipulation, ghost injection, replay and data block (DoS), with OpenSky data collection. **300 s, 360 s and 0.05% could NOT be confirmed** (Section 2a). | Mechanisms: Medium. Numbers: None. |
| 13 | 97 | `CITE: Wang, Zou, and Ding 2020 ADS-B attack-generation study` | wang2020 | Identified as EURASIP JWCN 2020:160. | High |
| 14 | 97 | `VERIFY: verify attack details and quantitative values attributed to Wang, Zou, and Ding against the primary source` | wang2020 | Attack list confirmed. "Heading **reversal**" is not supported: the paper says "heading change". Parameter values (Table 2) were not accessible. | Medium (Section 2b) |
| 15 | 103 | `CITE: systematic and comprehensive search of ADS-B FL-IDS literature` | khan2022flguard (+ near misses) | Not a citation slot. Report the search protocol instead (Section 3). | n/a |
| 16 | 103 | `VERIFY: complete a systematic literature search before retaining this research-gap statement` | khan2022flguard | **FL-Guard (NDSS 2022 poster) applies FedAvg to ADS-B spoofing/alteration detection.** The gap statement must cite it and be narrowed (Section 3). | High |
| 17 | 153 | `VERIFY: verify Li et al. injection-rate value and comparability with the per-type 2\% rate` | li2020 | 0.05% not confirmed. | None (remove number, Section 5) |
| 18 | 203 | `VERIFY: verify Li et al. replay delay and document the parameter comparison` | li2020 | 300 s not confirmed. | None |
| 19 | 213 | `VERIFY: compare flooding with Li et al.'s time-window denial-of-service method` | li2020 | A "data block (DoS)" module exists in the authors' code. The 360 s window is not confirmed. | Low |
| 20 | 217 | `VERIFY: verify Li et al. label use for training versus evaluation` | li2020 | Not confirmable from the abstract or preview. | None |
| 21 | 479 | `VERIFY: confirm Li et al. parameter values and flooding comparison` | li2020 | Same as rows 17 to 19. | None |
| 22 to 24 | 497 | `VERIFY authors / title / pages against doi:10.1049/iet-ifs.2018.5635` | li2020 | Tengyao Li, Buhong Wang, Fute Shang, Jiwei Tian, Kunrui Cao. "Threat Model and Construction Strategy on ADS-B Attack Data". IET Inf. Secur. 14(5):542-552, 2020. | High (issue: see 2a) |

**Tally of the 24 literature TODOs:**

- **Resolved (12):** rows 1, 3, 5, 6, 8, 9, 10, 11, 13, and 22 to 24.
- **Partially resolved (6):** rows 2, 4, 7, 12, 14 and 16. Row 16 is resolved as a finding: the search found prior work, so the gap statement must change.
- **Not resolvable from accessible sources (5):** rows 17 to 21, all Li et al. numeric or label claims. The text should change.
- **Not a citation slot (1):** row 15.

### 1b. Key identity notes for `main.tex`

- `strohmeier2014` is IEEE Communications Magazine 52(5):111-118 (2014), "Realities and Challenges of NextGen Air Traffic Management: The Case of ADS-B". This fits main.tex lines 43 and 60 (ADS-B broadcast and 1090ES). The Oxford BibTeX key `strohmeier2014surveys` is used for the **COMST 2015** paper, which is `strohmeier2015` here. Do not mix them up.
- `strohmeier2015` is IEEE COMST 17(2):1066-1087, doi 10.1109/COMST.2014.2365951. The print year is 2015, even though the DOI string contains 2014.
- `costin2012` is a Black Hat USA 2012 paper/talk, not a peer-reviewed venue. It is acceptable alongside schafer2013, but a reviewer may object if it is cited alone.
- `zhao2018` is an arXiv preprint only (no journal-ref).
- `li2020` (IET) and `li2020fedprox` (MLSys) are different works, and both keys are correct.
- `ying2019` now points to the peer-reviewed IEEE CNS 2019 version.
- `ngamboe2025` now points to DASC 2025, with the arXiv id kept in a note.
- **main.tex line 80 / 296:** `schafer2014` is cited for "historical state vectors, released as hourly files for one Monday per week". The IPSN 2014 paper predates that release and does not describe it. Cite the OpenSky dataset documentation or terms of use instead (not verified here), or limit the schafer2014 citation to "OpenSky Network".

### 1c. Internal TODOs (need the authors' code or data, not literature)

| Line | TODO | Action |
|---|---|---|
| 111 | grid resolution for region_cell | main.tex now states 10 x 10 degrees. Confirm it in the code. |
| 243, 263, 417 | squawk_changed null and first-row behavior | Run the feature code on a sample. |
| 293 | LightGBM hyperparameters, class weights, prior correction on LightGBM | Read from the training script. |
| 333 | 2 or 3 local epochs | Read from the run config or log. |
| 361 | missed-attack-rate denominator | Read `evaluate_holdout`. |
| 407 | identity of the 1,143,664-row flooding set | Trace the ablation script. |
| 491 | repository link | Authors to supply. |

---

## 2. Specific verifications

### 2a. doi:10.1049/iet-ifs.2018.5635 (li2020)

- **Authors (confirmed, ResearchGate record):** Tengyao Li, Buhong Wang, Fute Shang, Jiwei Tian, Kunrui Cao.
- **Title:** "Threat Model and Construction Strategy on ADS-B Attack Data". The title was also confirmed by the Wiley search-result title "Threat model and construction strategy on ADS-B attack data - Li - 2020 - IET Information Security" and by the authors' GitHub README ("the simulation codes for the paper 'Threat Model and Construction Strategy on ADS-B Attack Data'").
- **Volume/issue/pages:** 14(5), 542-552, published online 13 May 2020, issue dated September 2020. One discrepancy: ResearchGate's metadata field says 14(6), but the article header it reproduces says "Vol. 14, Iss. 5". Issue 5 is retained.
- **Access:** The Wiley, IET Digital Library and ACM DL pages all refused automated access, and Crossref was blocked by the proxy. Only the abstract (via ResearchGate) and the GitHub README were readable. **The paper is effectively paywalled for this check.**
- **What the abstract supports (paraphrased, because the fetch tool would not return it verbatim):**
  - ADS-B data are broadcast in plain text without adequate security.
  - Real ADS-B attack data are hard to obtain because of aviation regulations.
  - The paper models ADS-B data features and classic attack patterns as threat models.
  - It proposes a construction strategy covering target selection, key parameter choice, and mixing attack data with normal data.
  - It validates the strategy by simulation on real ADS-B data.
  - The only quotable fragment returned was that ADS-B data can be "jammed, tampered, replayed or deleted".
- **What the GitHub README supports:** four attack modules: data manipulation (tampering), ghost injection, replay, and data block (DoS). It also has an `attack_choice` target-selection module and a `data_collector` that "collects data from OpenSky-Network" via `opensky_api`. This supports the paper's statement that Li et al. include replay and ghost-aircraft patterns, and suggests Li et al. also used OpenSky data.

**Claims that could NOT be confirmed:**

- **300 s replay delay:** not in the abstract or README.
- **360 s DoS window:** not in the abstract or README. A "data block (DoS)" module exists.
- **0.05% injection rate:** not in the abstract or README.
- **Labels used for evaluation only:** not stated in any accessible text.

Searches for secondary papers quoting these values found none. PeerJ ahmed2025 and the two Kaabouch-group papers do not cite Li et al. for parameters.

**Recommendation:** obtain the PDF through a university library (IET/Wiley) and check the experimental-setup section. Until then, remove the numbers from the text (Section 5).

### 2b. Wang, Zou and Ding 2020 (wang2020)

- **Record:** J. Wang, Y. Zou, J. Ding, "ADS-B spoofing attack detection method based on LSTM", EURASIP Journal on Wireless Communications and Networking, 2020, article 160, doi:10.1186/s13638-020-01756-8 (open access, SpringerOpen). The publisher page gives initials only.
- **Data:** "decoded from real ADS-B messages with a total length of approximately 220,000", obtained from the GitHub project junzis/flight-data-processor. This is not stated as OpenSky.
- **Attack list (Section 4.2, quoted):** "random noise, fixed offset (+), fixed offset (-), route replacement, altitude offset (+), height offset (-), speed offset (+), speed offset (-), heading change, and climb rate change". The paper groups them as "10 different types of attack data for jamming, modification, and injection". Attacks were injected into sequence segments [100, 105] (and [140, 145]).
- **Check of the attributed attack types:**

| Attributed type | Status |
|---|---|
| Positional perturbation | Supported, as random noise and fixed offsets. Which features each offset targets is in Table 2, which could not be viewed. |
| Route replacement | Supported |
| Altitude and speed changes | Supported, as +/- offsets |
| Climb-rate manipulation | Supported ("climb rate change") |
| **Heading reversal** | **Not supported.** The paper says "heading change". Change the wording. |

- **Quantitative values:** Table 2 (attack parameters) was not reachable. Tables 3 and 5 were (for example, climb-rate F1 0.3032 and heading F1 0.5914 in Table 3). The original manuscript attributes no numbers to Wang, so nothing else needs to change.
- **Model:** one LSTM layer of 14 units, a dense layer of 7 units, next-step prediction, and a 3-sigma residual threshold per feature.
- **Oddity:** Wang et al. say their simulation builds on reference [10], but [10] in their list is Graves and Schmidhuber's BiLSTM phoneme paper. This looks like a citation error in their paper and does not affect yours.

### 2c. Ying et al. and Ngamboe et al.

- **Ying et al.:**
  - arXiv:1904.09969 title: "Detecting ADS-B Spoofing Attacks using Deep Neural Networks".
  - Authors (ar5iv): Xuhang Ying, Joanna Mazer, Linda Bushnell, Radha Poovendran (University of Washington); Giuseppe Bernieri, Mauro Conti (University of Padua).
  - **Peer-reviewed version (University of Padua repository):** X. Ying, J. Mazer, G. Bernieri, M. Conti, L. Bushnell, R. Poovendran, "Detecting ADS-B Spoofing Attacks Using Deep Neural Networks", 2019 IEEE Conference on Communications and Network Security (CNS), pp. 187-195, doi:10.1109/CNS.2019.8802732. **Cite this version** (author order as in the repository).
- **Ngamboe et al.:**
  - arXiv:2510.08333v1 (9 Oct 2025, only version) title: "New Machine Learning Approaches for Intrusion Detection in ADS-B".
  - Authors: Mikaëla Ngamboé, Jean-Simon Marrocco, Jean-Yves Ouattara, José M. Fernandez, Gabriela Nicolescu. No journal-ref.
  - The Polytechnique Montréal repository lists it as a conference paper at the **44th Digital Avionics Systems Conference (DASC 2025), Montreal, September 2025** (10 pages, no DOI). No IEEE Xplore record was found yet. Cite DASC 2025 and keep the arXiv id; add pages and DOI when Xplore lists it.

### 2d. McMahan 2017 and Ke 2017

- **mcmahan2017 (PMLR page):**
  - Authors: Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, Blaise Aguera y Arcas.
  - Venue: Proceedings of the 20th International Conference on Artificial Intelligence and Statistics, PMLR vol. 54, pp. 1273-1282, 2017, eds. Aarti Singh and Jerry Zhu.
  - The original bibitem is correct. Note that PMLR writes "Aguera" without an accent.
- **ke2017 (ML Anthology):**
  - Authors: Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, Tie-Yan Liu.
  - Venue: NeurIPS (NIPS) 2017, pp. 3146-3154.
  - The volume number (30) was not displayed on the opened page, so it is left out of the bib.

---

## 3. Search for federated learning on ADS-B and aviation surveillance

**Queries run:**

- "federated learning ADS-B intrusion detection"
- "federated learning aviation surveillance anomaly detection ADS-B spoofing privacy-preserving"
- "\"federated\" \"ADS-B\" anomaly detection 2024 OR 2025"
- "federated learning ADS-B spoofing detection OpenSky clients receivers"
- "federated learning air traffic management aircraft trajectory privacy ADS-B data sharing"
- "Khan Braeken Kumar Gurtov federated learning aviation ADS-B"

This was a web search, not a database-level systematic review. Scopus, IEEE Xplore and Web of Science were not queried.

### Direct hit (must be cited)

**khan2022flguard:** S. Khan, A. Braeken, P. Kumar, A. Gurtov, "POSTER: FL-Guard: A Federated Learning Based Ground-Air Secure Communication Model For Future Aviation Network", NDSS 2022 poster (3 pages).

- **Data:** normal messages collected at Arlanda Airport, Sweden. Attack messages come from an ADS-B simulator and are synthetic.
- **Size:** 4,000 samples across 4 aircraft, split 90:10.
- **Attacks evaluated:** alteration and spoofing only.
- **FL setup:** 4 virtual-aircraft clients, each with 1,000 samples and 25 local epochs, aggregated by FedAvg.
- **Model:** an MLP.
- **Result:** 89% test accuracy on the attack class. Results are preliminary, with no centralized baseline.

### Near misses (not ADS-B intrusion detection)

- **Chandar 2023, UND MS thesis**, "Benchmarking Federated Learning Frameworks For Aviation Network Data": FL for ADS-B/GPS *missing-data imputation* and systems benchmarking, not intrusion detection.
- **Liu and Papadimitratos 2025**, arXiv:2505.06171, "Self-supervised federated GNSS spoofing detection with opportunistic data": FedAvg on smartphone GNSS data. It is not about ADS-B.
- **li2019ijcip** (Li and Wang 2019): *collaborative* ground-to-ground, ground-to-air and air-to-air detection that fuses detector outputs. It is not federated learning, but it is the closest conceptual precedent for multi-receiver collaboration.
- A search result pointed to a CORDIS/SESAR project on federated learning in aviation. It was not opened and is not cited.

### Suggested research-gap sentence

> Federated learning has received little attention for ADS-B intrusion detection. The only federated ADS-B detector we found, FL-Guard \cite{khan2022flguard}, is a preliminary poster that trains a small network with FedAvg across four simulated aircraft on 4,000 messages from one airport, evaluates alteration and spoofing attacks, and reports no centralized baseline. To our knowledge, no prior study trains a federated detector over geographically partitioned receiver data at OpenSky scale, compares it with a centralized model of identical architecture, or audits synthetic-attack features for label leakage. Our search covered web and repository sources rather than a formal database review, so this describes the literature we found rather than a claim of priority.

---

## 4. Related-work comparison table

"n/r" means not reported on the pages opened. Cells reflect only what was verified.

| Key | Year | Data source | Attack types | Attack labels | Model | Federated | Evaluation notes |
|---|---|---|---|---|---|---|---|
| habler2018 | 2018 | Real FlightRadar24 recordings, 6 routes (54-80 flights each), Feb-Jul 2017. Not OpenSky. | Random noise, route replacement, gradual altitude shift up/down | Synthetic (segment of 70 messages modified) | LSTM encoder-decoder, cosine reconstruction error, 95th-percentile threshold | N | All injected attacks detected; average FPR about 4.3% |
| ying2019 | 2019 | Own RTL-SDR receivers in Seattle. Not OpenSky. | Message/IQ replay, ghost injection (aircraft spoofing described but not generated) | Real RF transmissions from a USRP spoofer (wired/anechoic chamber) | SODA: two-stage DNN on physical-layer features | N | 99.34% detection, 0.43% false alarm; aircraft classifier F-score 96.68% over 238 aircraft |
| li2019ijcip | 2019 | Real ADS-B data | Random deviation injection, constant deviation injection, DoS | n/r | Sequential collaborative detection (flight-plan validation, single-node, group) | N (collaborative detection, not FL) | n/r on abstract page |
| wang2020 | 2020 | Real decoded ADS-B, about 220k points (junzis flight-data-processor) | 10 types: noise, +/- fixed offset, route replacement, +/- altitude, +/- speed, heading change, climb-rate change | Synthetic (injected segments) | LSTM (14 units) next-step predictor, 3-sigma residual | N | Per-attack F1 0.30 (climb rate) to 0.94 (fixed offset -) |
| fried2021 | 2021 | "Large-scale and publicly available flight recordings" (source not named in abstract) | Trajectory modification | Synthetic (simulated) | Non-recurrent autoencoder after differencing | N | ROC AUC 0.935-0.951 on flights unseen in training |
| ouldslimane2022 | 2022 | **OpenSky**, 1 h within 50 km of JFK, 22,315 instances | Path modification, ghost injection, velocity drift | Synthetic | Linear SVM, C-SVM, nu-SVM | N | C-SVM 95.32% accuracy, 1.86% false alarm |
| khan2022flguard | 2022 | Real messages from Arlanda Airport plus simulator, 4,000 samples | Alteration, spoofing | Synthetic (simulator) | MLP | **Y** (FedAvg, 4 virtual aircraft) | 89% attack-class test accuracy; preliminary; no baseline |
| khoei2024 | 2024 | ADS-B from receivers at various locations (same 22,315-sample set as their earlier work) | Path modification, ghost injection, velocity drift | Synthetic | LSTM, BiLSTM, GRU, BiGRU | N | GRU 94.61% accuracy; 67/33 split |
| ahmed2025 | 2025 | **OpenSky** benign (180,716) plus OpenScope-simulator malicious (163,490) | 9 types incl. message delay, transponder-code change, ghost injection, trajectory modification, standing still, aircraft spoofing, jumping aircraft | Synthetic (simulator) | TabNet, NODE, DeepGBM | N | DeepGBM about 98.07% test accuracy (the paper's own comparison table says 98.7%); 80/20 split |
| cevik2025 | 2025 | n/r in abstract | 8 scenarios incl. spoofing, message injection, virtual trajectory change | Generated by authors | RF, extra trees, DT, MLP | N | RF/ET 0.999 accuracy |
| ngamboe2025 | 2025 | **OpenSky** state vectors at 10 s, 3 datasets | Gradual altitude (+82 ft/msg), groundspeed (+1.9 kn), heading (+1 deg); standing-still for generalization | Synthetic | xLSTM and transformer encoder (pretrained on benign, fine-tuned) | N | xLSTM F1 98.9%; about 7.26 s inference |

**Take-away for the prose:** every ADS-B detector found, including the only federated one, trains on synthetic or simulated attack labels. Only ying2019 uses real over-the-air RF attack transmissions, and that was under lab conditions. Three detectors (ouldslimane2022, ahmed2025, ngamboe2025) inject attacks into OpenSky data, as AeroGuard does.

---

## 5. Claims that could not be supported, and suggested text changes

1. **Li et al. numbers (original lines 97, 153, 203, 213, 217, 479; check whether main.tex still has them).** 300 s replay delay, 360 s DoS window, 0.05% injection rate, and "labels for evaluation only" were not confirmable. Suggested wording:

   > Li et al. propose a threat model and construction strategy for ADS-B attack data covering tampering, ghost injection, replay and data-block (denial-of-service) patterns injected into real traffic \cite{li2020}. Our generation rules were designed independently, and their parameters are not intended to match theirs.

   Remove every numeric comparison with Li et al. unless the full text is checked.
2. **Wang et al. "heading reversal".** Change to "heading changes". Also call the positional perturbations "random-noise and fixed-offset perturbations".
3. **Research-gap statement (original line 103).** It must cite FL-Guard and be narrowed (Section 3).
4. **"Pooling surveillance feeds raises commercial and regulatory concerns"** (main.tex line 45) is uncited. No verified source was found. Either soften it to a motivation ("may raise") or find an ICAO, EUROCONTROL or ANSP data-sharing policy source.
5. **"OpenSky ... historical state vectors, released as hourly files for one Monday per week" cited to schafer2014** (main.tex line 80). schafer2014 does not document this release, and strohmeier2021 (ESSD) describes a different, flight-level dataset. Cite the OpenSky dataset page or terms of use (not verified here) or reword.
6. **"Cryptographic remedies require changes to long-lived avionics and ground infrastructure" cited to strohmeier2015** (main.tex line 43). The topic fits, but no supporting sentence was quoted. Check section VI of strohmeier2015 when editing.
7. **The EU ADS-B mandate** (if mentioned) has no verified source. Only the US rule (faa91225) and the FAA/EUROCONTROL statement in strohmeier2014 were confirmed.
8. **Meaconing** (original line 67) is not in the verified taxonomy. main.tex already removed it.
9. **The "17th IJCAI" edition number, the NeurIPS volume numbers and the Neural Computation issue number** were not displayed on opened pages and were left out of the bib. Add them only after checking.
10. **Prior correction** (main.tex section 4): main.tex correctly cites the Saerens ratio form and states that a uniform effective training prior reduces it to pi^s. Make sure the "uniform effective prior" assumption is stated as an approximation, since the class weights (0.80 to 1.12) do not make it exactly uniform.

---

## 6. URLs opened, by reference

**ADS-B background and security**

- strohmeier2014: https://www.cs.ox.ac.uk/publications/publication8579.bib ; https://ora.ox.ac.uk/objects/uuid:dae3c7a3-e624-46c4-97d7-91eb69f1c14b ; https://www.lenders.ch/publications/journals/COMMAG14.pdf
- strohmeier2015: https://www.cs.ox.ac.uk/publications/publication8644.bib ; https://arxiv.org/pdf/1307.3664
- costin2012: https://www.eurecom.fr/publication.bib/3788 ; https://www.eurecom.fr/publication/3788
- schafer2013: https://link.springer.com/doi/10.1007/978-3-642-38980-1_16 ; https://cs.ox.ac.uk/publications/publication8023-abstract.html
- schafer2014: https://www.cs.ox.ac.uk/publications/publication8453.bib ; https://www.cs.ox.ac.uk/files/7238/ipsn2014.pdf
- strohmeier2021: https://essd.copernicus.org/articles/13/357/2021/essd-13-357-2021.html
- faa91225: https://www.law.cornell.edu/cfr/text/14/91.225
- ahmed2026survey: https://research.universityofgalway.ie/en/publications/towards-secure-air-traffic-surveillance-a-survey-on-ads-b-threats/
- ngamboe2025cabba: https://publications.polymtl.ca/view/year/2025/Ngamboe_Mvogo,_Mikaela_St=E9phanie.html

**Federated learning, privacy, synthetic data and leakage**

- mcmahan2017: https://proceedings.mlr.press/v54/mcmahan17a.html
- yang2019: https://api.crossref.org/works/10.1145%2F3298981
- zhao2018: https://arxiv.org/abs/1806.00582v1
- li2020fedprox: https://proceedings.mlsys.org/paper_files/paper/2020/file/1f5fe83998a09396ebe6477d9475ba0c-Metadata.json
- zhu2019: https://mlanthology.org/neurips/2019/zhu2019neurips-deep
- bonawitz2017: https://eprint.iacr.org/2017/281 ; https://research.google/pubs/pub47246/ ; https://bibbase.org/network/publication/bonawitz-ivanov-kreuter-marcedone-mcmahan-patel-ramage-segal-etal-practicalsecureaggregationforprivacypreservingmachinelearning-2017
- abadi2016: https://arxiv.org/abs/1607.00133 ; https://research.google/pubs/pub45428/
- chawla2002: https://jair.org/index.php/jair/article/view/10302
- goodfellow2014: https://mlanthology.org/neurips/2014/goodfellow2014neurips-generative
- kapoor2023: https://reproducible.cs.princeton.edu/citation.bib ; https://collaborate.princeton.edu/en/publications/leakage-and-the-reproducibility-crisis-in-machine-learning-based-/
- arp2022: https://usenix.org/conference/usenixsecurity22/presentation/arp
- saerens2002: https://mlanthology.org/neco/2002/saerens2002neco-adjusting
- elkan2001: https://mlanthology.org/ijcai/2001/elkan2001ijcai-foundations
- ke2017: https://mlanthology.org/neurips/2017/ke2017neurips-lightgbm

**ADS-B attack data and detectors**

- li2020: https://www.researchgate.net/publication/340509261_Threat_Model_and_Construction_Strategy_on_ADS-B_Attack_Data ; https://github.com/litengyao/adsb-attack-data-generator (the Wiley, IET Digital Library and ACM DL pages were attempted and refused)
- wang2020: https://jwcn-eurasipjournals.springeropen.com/articles/10.1186/s13638-020-01756-8 ; https://link.springer.com/article/10.1186/s13638-020-01756-8 ; .../tables/3 ; .../tables/5
- ying2019: https://ar5iv.labs.arxiv.org/html/1904.09969 ; https://www.research.unipd.it/handle/11577/3340669
- ngamboe2025: https://arxiv.org/abs/2510.08333v1 ; https://arxiv.org/html/2510.08333v1 ; https://publications.polymtl.ca/view/year/2025/Ngamboe_Mvogo,_Mikaela_St=E9phanie.html
- habler2018: https://cris.bgu.ac.il/en/publications/using-lstm-encoder-decoder-algorithm-for-detecting-anomalous-ads/ ; https://arxiv.org/pdf/1711.10192
- fried2021: https://www.bgu.ac.il/en/researcher/mark-last/publications/318287601/
- li2019ijcip: https://ideas.repec.org/a/eee/ijocip/v24y2019icp78-99.html
- ouldslimane2022: https://par.nsf.gov/servlets/purl/10354439
- khoei2024: https://par.nsf.gov/servlets/purl/10564790
- ahmed2025: https://peerj.com/articles/cs-2886.pdf
- cevik2025: https://acikerisim.istinye.edu.tr/items/a0e8c4df-51ba-4716-9366-c4c0cbedd634/full

**Federated detection in aviation and other domains**

- khan2022flguard: https://www.ndss-symposium.org/wp-content/uploads/NDSS2022Poster_paper_26.pdf
- khraisat2024: https://research.torrens.edu.au/en/publications/survey-on-federated-learning-for-intrusion-detection-system-conce/
- nguyen2019diot: https://tubiblio.ulb.tu-darmstadt.de/112157/ ; https://arxiv.org/abs/1804.07474v4
- ruzafa2023: https://portalinvestigacion.um.es/documentos/63b0d86d0f8bcd1826cffc5b?lang=en
- abdelbasset2022: https://researchprofiles.canberra.edu.au/en/publications/federated-intrusion-detection-in-blockchain-based-smart-transport/
- kesici2024: https://spiral.imperial.ac.uk/entities/publication/e7842b26-f2bb-4a6a-9db6-8a3016226d0b

**Near misses (opened, not in bib)**

- https://commons.und.edu/theses/5705
- https://arxiv.org/html/2505.06171v1
- https://preprints.org/manuscript/202405.0586/v1
