# 15-Qubit Quantum Language Distribution Experiment

[![Python](https://img.shields.io/badge/python-3.13-blue.svg)](environment_info.json)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository accompanies Part 3 of my quantum series on Medium.

Part 3 article URL: `PART_3_MEDIUM_URL`.

This repository contains the complete, standalone research artifact for the 15-qubit quantum language distribution experiment proposed in Abhinay Sharma's article [*Stop Porting GPU Training to Quantum Computers*](https://medium.com/@sharmaabhinay28/stop-porting-gpu-training-to-quantum-computers-fd7cf5cf3bdb).

The study evaluates an Instantaneous Quantum Polynomial-time (**IQP**) Born machine against classical generative controls on unconditional 3-character text generation from Project Gutenberg Ebook #100 (*The Complete Works of William Shakespeare*).

---

## Quickstart: Reproduction in 4 Shell Commands

Verify all 180 candidate checkpoints, 60 frozen selections, SHA-256 content hashes, and numerical test results from scratch:

```bash
git clone https://github.com/sharmaabhinay28/quantum-shakespeare-15q-ready.git && cd quantum-shakespeare-15q-ready
python3.13 -m venv venv
venv/bin/pip install -r requirements.txt
venv/bin/python verify_artifacts.py
```

---

## 1. What This Experiment Evaluates

- **Task:** Unconditional discrete generative modeling of 3-character text chunks ($N=15$ bits, $2^{15} = 32,768$ states).
- **Core Question:** Does an IQP Born machine demonstrate an empirical distribution learning advantage over classical parametric controls of matched parameter count (120 parameters) and an empirical frequency baseline under finite optimization?
- **Execution:** CPU noiseless statevector simulation in software using Xanadu's `iqpopt` (pinned commit `18ae04255ed80a274d8c49ed91cd4e6214757ccd`) and PennyLane.

---

## 2. Dataset & Encoding Bijections

- **Corpus:** Project Gutenberg Ebook #100 (*The Complete Works of William Shakespeare*).
  - Exact Raw File: [`data/gutenberg_100_raw.txt`](data/gutenberg_100_raw.txt) (5,638,481 bytes, CRLF preserved).
  - SHA-256: `eb61e33cab23dbe712ad57091ded6c7292ba91ac3bb2764f51a3223ec0068fe5`.
- **Alphabet:** 32 symbols (26 lowercase English letters, space, period, comma, semicolon, question mark, exclamation mark).
- **State Space:** 3 characters $\times$ 5 bits = 15 bits ($2^{15} = 32,768$ discrete states).
- **Splits:** Disjoint whole-passage splits partitioned by deduplicated text group SHA-256 hashes:
  - Train: 13,667 passages (70%)
  - Validation: 2,928 passages (15%)
  - Test: 2,929 passages (15%)
- **Encodings:** 3 deterministic bijections ([`data/encoding_maps.json`](data/encoding_maps.json)):
  - `primary`: Standard alphabetical order ($a=0 \dots \text{punct}=31$).
  - `alt1`: Pseudorandom permutation (seed 1729).
  - `alt2`: Pseudorandom permutation (seed 2718).

---

## 3. Models & Parameter Capacity

1. **IQP Born Machine (Quantum, 120 Parameters):**
   - 15 single-qubit $X$ rotations + 105 pairwise $XX$ interactions (implemented generator list: 15 $X$ + 105 $XX$).
   - Evaluated via Fast Walsh-Hadamard Transform (FWHT) statevector probabilities.
   - Trained on Maximum Mean Discrepancy (MMD) with multi-scale Gaussian kernels ($\sigma \in \{1.0, 2.0, 4.0\}$).
2. **Stochastic Bitflip (Classical Control, 120 Parameters):**
   - Correlated XOR error channels with matched gate structure (15 single + 105 pairwise).
   - Trained on MMD loss.
3. **Autoregressive Bernoulli (Classical Parametric Baseline, 120 Parameters):**
   - Strictly lower-triangular weight matrix (105 weights) + 15 bit biases = 120 parameters.
   - Trained on negative log-likelihood (cross-entropy).
4. **Frequency Table (Empirical Baseline, 32,768 States / 32,767 Independent Probabilities):**
   - Smoothed empirical histogram with Laplace add-$\alpha$ smoothing ($\alpha \in \{0.01, 0.1, 1.0\}$).
   - Deterministic closed-form MLE fit (re-evaluated identically across seed slots; deterministic repeat, not independent random fits).

> **Capacity Note:** Matched parameter count (120 parameters across IQP, Bitflip, and Autoregressive) does not denote identical expressive capacity. IQP explores quantum probability distributions over commuting Pauli observables, Bitflip samples classical XOR noise, and Autoregressive Bernoulli parameterizes a directed acyclic probabilistic network with explicit conditional factorization.

---

## 4. Hyperparameter Selection & Validation

For each model, encoding, and seed, 3 candidate configurations were fitted on the training split and selected by chunk negative log-likelihood on the validation split:
- **Parametric models (IQP, Bitflip, Autoregressive):** Adam learning rates $\eta \in \{0.01, 0.03, 0.1\}$, 500 updates, batch size 512.
- **Frequency Table:** Smoothing parameter $\alpha \in \{0.01, 0.1, 1.0\}$.
- Total candidate fits: 4 models $\times$ 3 encodings $\times$ 5 seeds $\times$ 3 hyperparameter configurations = **180 candidate checkpoints**.
- The 60 validation-selected models are frozen in [`results/selection_manifest.json`](results/selection_manifest.json).

---

## 5. Primary Scientific Results (Held-Out Test Split)

Metrics evaluated on the held-out test split (2,929 passages, 680,787 three-character chunks) for the primary encoding across five fit seeds:

| Model | Parameters | Training Loss | Test NLL (nats/chunk) | Test NLL (nats/char) | Seed SD | 95% Bootstrap CI* | Test MMD$^2$ | Unseen Valid Mass | Invalid Mass |
|---|---|---|---|---|---|---|---|---|---|
| **IQP Born Machine** | 120 | MMD | **$10.465410$** | **$3.488470$** | $\pm 0.040213$ | $[3.4876, 3.4894]$ | $0.003532$ | $0.701708$ | $0.000000$ |
| **Stochastic Bitflip** | 120 | MMD | **$10.185880$** | **$3.395293$** | $\pm 0.002227$ | $[3.3946, 3.3960]$ | $0.004151$ | $0.733700$ | $0.000000$ |
| **Autoregressive Bernoulli** | 120 | NLL | **$9.349823$** | **$3.116608$** | $\pm 0.000353$ | $[3.1148, 3.1185]$ | $0.000861$ | $0.579407$ | $0.000000$ |
| **Frequency Table** | 32,767 | MLE + $\alpha$ | **$7.205917$** | **$2.401972$** | $\pm 0.000000$ | $[2.3995, 2.4043]$ | $0.000001$ | $0.000083$ | $0.000000$ |

*\*Passage bootstrap intervals: 2,000 paired whole-passage resamples conditioned on the five selected checkpoints.*

### Key Observations
1. **Rank Ordering:** Across all three tested encodings, the mean model ranking is identical:
   $$\text{Frequency Table} < \text{Autoregressive Bernoulli} < \text{Stochastic Bitflip} < \text{IQP Born Machine}$$
2. **Individual IQP Seed Differences:** While IQP underperforms on mean log loss, under alternative encoding 1 (`alt1`), 3 individual IQP seeds (seeds 23, 37, 71) beat their matched Bitflip fits on log loss.
3. **Unseen Valid Chunk Mass:** The models allocate probability mass to 3-character sequences formed from the 32-symbol alphabet that did not appear in the training split:
   - IQP Born Machine: **$70.1708\%$**
   - Stochastic Bitflip: **$73.3700\%$**
   - Autoregressive Bernoulli: **$57.9407\%$**
   - Frequency Table: **$0.008297\%$**
   *(Note: Alphabet-valid chunks consist of symbols from the defined alphabet, but are not necessarily English words.)*
4. **Zero Invalid Mass:** Because 15 bits biject exactly to $32^3 = 32,768$ alphabet triples, invalid chunk mass is strictly $0.000000$ for all models.

---

## 6. Methodological Limits & Nuances

- **MMD vs. NLL objective mismatch:** IQP and bitflip trained on MMD. The classical generator trained directly on negative log likelihood. On primary test squared MMD, IQP beat bitflip, 0.00353159 versus 0.00415051. The classical generator did better still, at 0.00086118. Under alternative encoding 1, IQP did not beat bitflip on squared MMD: 0.00780339 versus 0.00759448.
- **Seed Variance vs. Passage Bootstrap Interval:**
  - Across-seed standard deviation ($\pm 0.0402$ for IQP) captures variability over initial parameter weights and mini-batch sample paths.
  - The paired passage bootstrap interval ($[3.4876, 3.4894]$) is conditional on the five fitted checkpoints and captures text passage sampling variance. Neither is a confidence interval over new model architectures.
- **Toy Scale Chunking:** This is a small, unconditional 3-character distribution model ($2^{15}$ states), **not** a large language model (LLM) or generative sequence predictor.
- **No Physical Hardware / No Speedup Claim:** All simulations were executed via noiseless classical linear algebra on CPU. No physical quantum processor was used; no claims of physical quantum coherence, fault tolerance, or quantum computational speedup are made.

---

## 7. Provenance & Archival Clarification

- **Frozen Protocol:** Defined in [`PROTOCOL.md`](PROTOCOL.md) (SHA-256 `64086fe2e51c747d39aaba3fe17856583e05f93f6428ea20fb160c305b473dac`).
- **Archival Chronology:** The experimental sequence (protocol freeze $\to$ candidate training $\to$ selection $\to$ test evaluation) was recorded in private git history. In this standalone public export, this sequence is an **archival research assertion**; local git history cannot externally prove the original timeline, and no claim of external pre-registration is made.
- For complete details, see [`PROVENANCE.md`](PROVENANCE.md).

---

## 8. Research Credits & AI Assistance Disclosure

The tested environment is Python 3.13 on Apple Silicon macOS. Other platforms have not been independently tested. The divided log-loss score is chunk loss divided by three, not a long-context entropy rate. Resource memory values use MiB despite legacy MB labels in the raw keys and charts.

- **Experiment Conception & Direction:** Conceived, scoped, authored, and directed by **Abhinay Sharma**.
- **Implementation:** **Gemini 3.8 Flash** implemented the code, mathematical modules, and automated test harnesses under instruction.
- **Reproduction Architecture & Verification:** **GPT 6.1 Sol** planned the reproduction architecture, reviewed the implementation, independently verified numerical parity, and edited the research artifacts.
- **Testing Scope:** Automated test suites and isolated worker replays verify internal computational consistency; they do not constitute external independent replication.

---

## 9. Optional: Two-Stage Retraining into a NEW Output Directory

To retrain candidate models from scratch without modifying frozen artifacts:

```bash
# 1. Run candidate training into a fresh, non-existent run directory
venv/bin/python src/train_and_select.py --run-dir runs/reproduction

# 2. Evaluate newly trained candidates on held-out test split
venv/bin/python src/evaluate_test.py   --selection-manifest runs/reproduction/results/selection_manifest.json   --checkpoint-dir runs/reproduction/checkpoints   --out-dir runs/reproduction/evaluation
```

All mutating scripts strictly guard against modifying frozen publication artifacts (`results/`, `checkpoints/`, `data/`). The legacy monolithic script `src/run_experiment.py` is safely disabled with an informative error message.

---

## 10. Repository Structure

```
quantum-shakespeare-15q-ready/
├── PROTOCOL.md                  # Frozen protocol specification (byte-identical)
├── PROVENANCE.md                # Research provenance, redaction log & audit records
├── README.md                    # Overview, reproduction guide & scientific findings
├── REVIEWED_RESULTS.md          # Comprehensive reviewed results and discussion
├── SOURCES.md                   # Complete citations for Project Gutenberg & Xanadu iqpopt
├── LICENSE                      # MIT License (copyright 2026 Abhinay Sharma)
├── export_manifest.json         # Non-self-recursive SHA-256 and byte manifest of all files
├── verify_artifacts.py          # Standalone verification script (10 numerical checks + provenance)
├── environment_info.json        # Platform, Python, JAX, and package versions
├── requirements.txt             # Pinned install requirements copied from dependency_lock.txt
├── dependency_lock.txt          # Frozen pip dependencies (CPU packages, pinned iqpopt)
├── .gitignore                   # Ignores venvs, pycaches, and new run directories
├── .gitattributes              # Preserves CRLF on gutenberg source and marks binaries
├── data/
│   ├── gutenberg_100_raw.txt    # Verbatim Shakespeare text (5,638,481 bytes, CRLF preserved)
│   ├── gutenberg_source_metadata.json # Transfer metadata and retrieval timestamps
│   ├── encoding_maps.json       # Primary, Alt1, and Alt2 5-bit character bijections
│   └── splits_manifest.json     # Disjoint train/val/test passage partitions
├── src/
│   ├── data.py                  # Normalization, passage grouping, chunk extraction
│   ├── models.py                # IQP, Bitflip, Autoregressive, Frequency Table models
│   ├── evaluation.py            # FWHT, exact probabilities, exact squared MMD, log loss
│   ├── resolver.py              # Portable checkpoint path resolver without external fallback
│   ├── guard.py                 # Guard against overwriting frozen publication artifacts
│   ├── train_and_select.py      # Candidate training & selection manifest constructor
│   ├── evaluate_test.py         # Test evaluation, 2,000 passage bootstrap, plotting
│   ├── plot_reviewed.py         # Reviewed publication plots generator
│   ├── replay_resources.py      # Isolated-process CPU wall time & RSS profiling
│   └── run_experiment.py        # Safely disabled legacy pipeline runner
├── tests/
│   ├── test_units.py            # Complete unit test suite (math, guards, resolver, manifest)
│   └── fixtures/
│       └── corrupt_checkpoint_copy.npy # Retained fixture for controlled negative testing
├── checkpoints/                 # 180 frozen .npy candidate parameter checkpoints (12MB)
└── results/
    ├── candidate_validation_results.json / .csv  # 180 candidate validation scores
    ├── selection_manifest.json                   # 60 selected models manifest
    ├── test_results_summary.json / .csv          # Held-out test metrics for 60 models
    ├── test_per_passage_results.csv              # Per-passage loss across 2,929 test passages
    ├── bootstrap_results.json                    # 2,000 paired whole-passage bootstrap samples
    ├── samples_10k_summary.json                  # 10,000 chunk generation summaries
    ├── resource_accounting.json                  # Original pipeline wall time & cumulative RSS
    ├── resource_replay.json / .csv               # Isolated worker phase wall times & peak RSS
    ├── plot_test_log_loss_reviewed.png           # Reviewed bar chart (means ± seed SD)
    ├── plot_paired_differences_reviewed.png      # Reviewed paired differences (IQP − baseline)
    └── plot_resource_costs_corrected.png         # Corrected isolated phase resource profiles
```
