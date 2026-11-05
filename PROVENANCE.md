# Research Provenance & Archival Record

This repository is a self-contained, standalone public research export of the 15-qubit quantum language distribution experiment conducted by Abhinay Sharma.

---

## 1. Archival Provenance vs. External Pre-Registration

During the original research project, the experimental phases were executed sequentially in private development repositories:
1. Frozen protocol specification ([`PROTOCOL.md`](PROTOCOL.md)) was established before training candidate models.
2. 180 candidate configurations were trained on the training split and validation-selected models were recorded in [`results/selection_manifest.json`](results/selection_manifest.json) before held-out test evaluation.
3. Held-out test evaluation, paired whole-passage bootstrap resampling, and isolated worker resource profiling were executed strictly against frozen checkpoints.

### Provenance Status & Scope Caveats
- **Archival Research Assertion:** The sequential phase progression was recorded in local development version control. In this standalone public release repository, this sequence represents an archival research assertion by the author. Local git commit history cannot externally prove the original chronological timeline from this export alone.
- **No External Pre-Registration Claim:** Because local git commit histories lack third-party independent timestamping or registration prior to execution, no claim of external pre-registration is made.
- **Scientific Reproducibility:** Every scientific check, numerical score, parameter checkpoint array hash (`arr.tobytes()`), and recomputed test metric in this repository is strictly verifiable and reproduces within $\le 10^{-9}$ tolerance using [`verify_artifacts.py`](verify_artifacts.py) and the unit test suite.

---

## 2. Metadata Redaction & Path Sanitization

To ensure this exported repository is portable across different operating systems and computing environments while protecting private filesystem structures:

1. **Checkpoint Paths:**
   - In `results/candidate_validation_results.json`, `results/candidate_validation_results.csv`, and `results/selection_manifest.json`, local absolute filesystem paths were replaced with relative paths conforming to `checkpoints/<basename>`.
   - Zero absolute paths or private username references remain in metadata records.
2. **Numerical & Scientific Invariance:**
   - All 180 checkpoint parameter arrays (`.npy`), hashes, weights, loss values, timing measurements, and evaluation statistics remain strictly identical to the original experimental run.
   - The source text ([`data/gutenberg_100_raw.txt`](data/gutenberg_100_raw.txt)) and protocol ([`PROTOCOL.md`](PROTOCOL.md)) are preserved byte-identically.
3. **Strict Local Resolver:**
   - The checkpoint resolver ([`src/resolver.py`](src/resolver.py)) resolves checkpoints deterministically within the repository checkout (`checkpoints/` or explicit `--checkpoint-dir`). It rejects path traversal (`..`), absolute paths, and contains zero fallback to current working directory or external filesystems.

---

## 3. Public Attribution & Third-Party Licences

- **Author Credit:** Research design, execution, and analysis by Abhinay Sharma.
- **Source Text:** Project Gutenberg Ebook #100 (*The Complete Works of William Shakespeare*). The text contains mandatory historical Project Gutenberg public license headers and contact notices (`gbnewby@pglaf.org`, `business@pglaf.org`, `gutenberg.org`), which are intended public notices and preserved verbatim in [`data/gutenberg_100_raw.txt`](data/gutenberg_100_raw.txt).
- **Upstream Dependencies:** Third-party quantum and scientific computing packages (Xanadu `iqpopt`, PennyLane, JAX, NumPy, SciPy) remain licensed under their respective open-source licenses and are installed directly from upstream package repositories.

---

## 4. Export Integrity Manifest

All material files included in this export are indexed with their SHA-256 content hashes and exact byte counts in [`export_manifest.json`](export_manifest.json). The manifest is non-self-recursive and is verified by [`verify_artifacts.py`](verify_artifacts.py) and `tests/test_units.py`.
