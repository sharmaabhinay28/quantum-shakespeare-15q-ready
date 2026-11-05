# Final Preparation & Verification Report

**Repository:** `quantum-shakespeare-15q-ready`  
**Status:** PUBLICATION-READY STANDALONE RESEARCH ARTIFACT  
**Author / Study Director:** Abhinay Sharma  
**Implementation:** Gemini 3.8 Flash  
**Architecture Review & Verification:** GPT 6.1 Sol  
**Timestamp:** 2026-11-05  

---

## 1. Executive Summary

This report documents the preparation, hardening, and verification of the standalone local research artifact for the 15-qubit quantum language distribution experiment. Following the review of the initial packaging export, all identified issues have been systematically addressed in this clean, standalone repository.

All 180 candidate checkpoints, 60 frozen selections, raw dataset bytes, and exact numerical test metrics remain strictly invariant and identical to the original scientific study.

---

## 2. Review Findings & Implemented Fixes

### Finding A: Sanitized History & Relative-Only Documentation
- Created a fresh, clean repository folder (`quantum-shakespeare-15q-ready`) with a newly initialized, sanitized Git history.
- Generic archival committer identity configured: `Research Archival Contributor <research@example.invalid>`.
- All operational documentation, verification recipes, and reports use relative paths and commands exclusively.
- Scan for private filesystem paths and scratch directories is performed after all final edits across all tracked content and git history.
- Byte-level integrity preserved for Project Gutenberg raw text and public legal notices.

### Finding B: Data Preparation Zero-Write Guarantee
- Updated `src/data.py::prepare_corpus_and_splits` with `write_artifacts=False` as the default argument.
- When `write_artifacts=False`, the function executes entirely in memory with zero filesystem writes, directory creations, or modifications.
- All readers (`verify_artifacts.py`, `src/evaluate_test.py`, `src/replay_resources.py`) invoke data preparation with `write_artifacts=False`.
- `verify_artifacts.py` regenerates splits and encoding maps in memory and strictly verifies exact parity against the frozen JSON files on disk before proceeding.
- Unit tests verify that tracked file hashes and filesystem modification times (`mtime`) remain unchanged after execution.

### Finding C: Guarded Mutating Entry Points & Explicit Reproduction Recipes
- Implemented `src/guard.py::guard_output_dir` protecting frozen artifact directories (`results/`, `checkpoints/`, `data/`).
- Entry points validate output destinations prior to initiating any computation or model initialization:
  - `src/train_and_select.py`: Requires `--run-dir <new_dir>`, creating separate `<run_dir>/checkpoints` and `<run_dir>/results`. Refuses existing non-empty directories and protected directories.
  - `src/evaluate_test.py`: Requires separate explicit parameters:
    `--selection-manifest <path>`, `--checkpoint-dir <dir>`, and `--out-dir <new_dir>`.
    Evaluates strictly against the provided manifest and checkpoint directory rather than hardcoded archived defaults.
  - `src/run_experiment.py`: Legacy monolithic runner is safely disabled at entry with an informative message guiding users to the two-stage reproduction pipeline. The file is preserved for historical audit.
  - `src/plot_reviewed.py` and `src/replay_resources.py`: Guarded with explicit output destination flags.
- Comprehensive unit tests verify that writes into protected directories or existing non-empty paths are refused before computation begins.

### Finding D: Deterministic Checkpoint Resolver
- Hardened `src/resolver.py::resolve_checkpoint_path`:
  - Enforces relative path references; rejects absolute paths and path traversal (`..`).
  - When an explicit `ckpt_dir` is supplied, it acts as the sole authority. Fallbacks to working directory or default repository directories are prohibited.
  - Default resolution is anchored strictly to the repository `checkpoints/` directory.
  - Controlled test fixtures are supported via explicit fixture directories.
- Unit tests verify rejection of path traversal, absolute paths, missing checkpoints in explicit directories, and confirmation that unrelated files in working directories are ignored.

### Finding E: Scientific Descriptions & Capacity Clarifications
- Corrected IQP gate generator description in `README.md` to reflect the actual implemented circuit: 15 single-qubit $X$ rotations + 105 pairwise $XX$ interactions (120 total generators).
- Clarified that matched parameter count (120 parameters) does not denote identical representation capacity across distinct model families (quantum Born machine vs. classical XOR bitflip vs. directed autoregressive network).
- Reported exact unseen valid chunk masses:
  - IQP Born Machine: 70.1708%
  - Stochastic Bitflip: 73.3700%
  - Autoregressive Bernoulli: 57.9407%
  - Frequency Table: 0.008297%
- Clarified that alphabet-validity (strings composed of symbols from the 32-character alphabet) is distinct from natural English vocabulary.
- Documented Frequency Table properties: 32,768 states and 32,767 independent probabilities; identical across seed slots due to deterministic closed-form MLE fit.
- Documented MMD vs. NLL loss nuances, noting that while primary IQP outperformed Bitflip on test MMD, under alternative encoding 1 (`alt1`), IQP test MMD did not beat Bitflip.

### Finding F: Transparent Research Credits & Verification Badges
- Corrected research contributions in `README.md`:
  - Proposal and Study Direction: Abhinay Sharma.
  - Implementation: Gemini 3.8 Flash.
  - Architecture Planning, Review, and Verification: GPT 6.1 Sol.
- Clarified that automated test suites and isolated worker replays verify internal computational consistency and do not constitute external independent replication.
- Removed unverifiable verification badges and overstated claims.
- Separated the unnotarized archival chronology assertion from the count of verifiable numerical checks.

### Finding G: Provenance & Archival Status
- Omitted private development commit hashes from public-facing documentation and verification logs.
- Distinguished the selection manifest file SHA-256 from the protocol SHA-256 in verification reporting.
- Clarified that commit chronology represents an archival assertion from internal research that cannot be independently proven from local git history alone, and that no claim of external pre-registration is made.

### Finding H: Export Manifest Integrity & Tamper Detection
- Extracted a standalone export manifest verification helper: `validate_export_manifest(manifest_or_dict, repo_root)`.
- Verifies the hashes and sizes of every listed material file. Owner review separately compares the manifest inventory with all tracked files, excluding the manifest itself.
- Implemented unit tests verifying tamper detection for modified SHA-256 digests, modified byte counts, missing files, empty manifests, and self-recursive manifests.

---

## 3. Privacy, Secret & Path Audit

An automated scan was conducted across all tracked files, documentation, metadata, and reachable Git commit history.

| Category | Description | Scan Scope | Matches | Result |
|---|---|---|---|---|
| User Home Directories | Absolute user home filesystem paths | All tracked files and git commits | 0 | PASSED |
| System Temp Paths | System temporary and scratch paths | All tracked files and git commits | 0 | PASSED |
| Private Credentials | API keys, secret tokens, bearer credentials | All tracked files and git commits | 0 | PASSED |
| Private Keys | Private SSH, TLS, or cryptographic keys | All tracked files and git commits | 0 | PASSED |
| Private Contact Data | Unintended personal email addresses | All tracked files and git commits | 0 | PASSED |

**Verified Intentional Public Notices:**
- Author Public Attribution: `Abhinay Sharma`
- Archival Generic Email: `research@example.invalid`
- Project Gutenberg Legal Notices: `gbnewby@pglaf.org`, `business@pglaf.org`, `gutenberg.org` (preserved verbatim in source text)

---

## 4. Controlled Negative & Unit Testing

The test suite in `tests/test_units.py` was executed:
- Controlled negative test fixture retained: `tests/fixtures/corrupt_checkpoint_copy.npy`.
- Unit tests cover mathematical parity, PennyLane simulation equivalence, strict resolver rules, guard destination rejection, zero-write data preparation, and full export manifest validation with tamper detection.
- All fixtures are retained on disk without deletion.

---

## 5. Clean-Room Reproduction & Verification Procedure

The complete reproduction verification procedure is executable in 4 relative commands:

```bash
# 1. Navigate to repository root
cd <path-to-repository>

# 2. Initialize clean Python 3.13 virtual environment
python3.13 -m venv ./venv

# 3. Install locked dependencies
./venv/bin/pip install -r dependency_lock.txt

# 4. Execute standalone verification suite
./venv/bin/python verify_artifacts.py
```

### Measured Execution Results
- **Unit Test Suite (`pytest tests/test_units.py`):**
  - Implementation run: 12 passed, 0 failed, 0 skipped. The final owner verification record is maintained separately.
- **Verification Script (`python verify_artifacts.py`):**
  - [1/11] Protocol SHA-256 (`64086fe2...`): VERIFIED IDENTICAL
  - [2/11] Gutenberg Ebook #100 source (5,638,481 bytes, `eb61e33c...`): VERIFIED IDENTICAL
  - [3/11] 180 candidates and 60 selected checkpoints: VERIFIED
  - [4/11] 180 candidate + 60 selected checkpoint content hashes (`arr.tobytes()`): VERIFIED
  - [5/11] 3 encoding bijections in `encoding_maps.json`: VERIFIED
  - [6/11] 5 seeds (`[11, 23, 37, 53, 71]`): VERIFIED
  - [7/11] Selection manifest argmin and tie-breaking rule: VERIFIED
  - [8/11] Disjoint passage groups across train, val, and test: VERIFIED
  - [9/11] Exact recomputation of validation and test metrics from source:
    - Max validation chunk loss diff: $\le 10^{-9}$
    - Max test chunk loss diff: $\le 10^{-9}$
    - Max test squared MMD diff: $\le 10^{-9}$
    - Max unseen valid mass diff: $\le 10^{-9}$
    - Max invalid mass diff: $0.00$
    - Regenerated splits and encoding maps identical to frozen JSON on disk.
  - [10/11] Export integrity manifest validation: All material files verified intact.
  - [11/11] Archival research provenance and pre-registration limits documented.
- **Tracked File Invariance:**
  - Tracked file content hashes and modification times (`mtime`) before and after execution: 100% invariant, 0 writes to disk.
