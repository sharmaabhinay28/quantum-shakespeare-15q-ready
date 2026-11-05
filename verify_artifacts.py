#!/usr/bin/env python3
"""Portable Artifact Verification Script for 15-Qubit Quantum Language Experiment.

Verifies:
1. Protocol SHA-256 matches frozen protocol.
2. Source Gutenberg raw text SHA-256 and byte length.
3. 180 candidates in candidate_validation_results and 60 selections in selection_manifest.
4. Content hashes (arr.tobytes()) for all 180 candidate checkpoints and 60 selected checkpoints.
5. Three bijections in encoding_maps.json (primary, alt1, alt2).
6. 5 seeds (11, 23, 37, 53, 71).
7. Selection manifest argmin and tie-breaking rule from candidate validation results.
8. Strictly disjoint passage groups and IDs across train, val, and test splits.
9. Exact recomputation of validation and test metrics from source counts and frozen checkpoints,
   verifying absolute differences <= 1e-9 against original JSON records.
10. Export integrity manifest verification (export_manifest.json).
11. Explicit provenance documentation distinguishing scientific verification from archival chronology assertions.
12. Read-only operation: performs zero writes to existing artifact or results files.
"""

import os
import sys
import json
import hashlib
import numpy as np

# Ensure repository root is in path
base_dir = os.path.abspath(os.path.dirname(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from src.resolver import resolve_checkpoint_path, sha256_of_array
from src.data import ALPHABET, prepare_corpus_and_splits, extract_chunks_and_counts
from src.evaluation import (
    get_standard_gates,
    get_kernel_hat,
    exact_squared_mmd,
    evaluate_log_loss,
    probs_from_iqp,
    probs_from_bitflip,
    probs_from_autoregressive
)

FROZEN_PROTOCOL_SHA = "64086fe2e51c747d39aaba3fe17856583e05f93f6428ea20fb160c305b473dac"
EXPECTED_SOURCE_SHA = "eb61e33cab23dbe712ad57091ded6c7292ba91ac3bb2764f51a3223ec0068fe5"
EXPECTED_SOURCE_BYTES = 5638481
EXPECTED_SEEDS = [11, 23, 37, 53, 71]
EXPECTED_ENCODINGS = ["primary", "alt1", "alt2"]
EXPECTED_MODELS = ["IQP", "Bitflip", "Autoregressive", "Table"]


def validate_export_manifest(manifest_or_dict, repo_root=None) -> int:
    """Validates that all files listed in an export manifest exist and match byte count and SHA-256.

    Args:
        manifest_or_dict: Filepath string or parsed manifest dict.
        repo_root: Root directory of repository (default: base_dir).

    Returns:
        int: Number of files verified.

    Raises:
        ValueError: If manifest is empty, self-recursive, or contains size/hash mismatches.
        FileNotFoundError: If a listed file does not exist on disk.
    """
    if repo_root is None:
        repo_root = base_dir

    if isinstance(manifest_or_dict, str):
        with open(manifest_or_dict, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    elif isinstance(manifest_or_dict, dict):
        manifest = manifest_or_dict
    else:
        raise TypeError("manifest_or_dict must be a file path string or dict")

    files = manifest.get("files", {})
    if not files:
        raise ValueError("Export manifest is empty or contains no files.")

    if "export_manifest.json" in files:
        raise ValueError("export_manifest.json must not be self-recursive.")

    for rel_path, meta in files.items():
        file_path = os.path.join(repo_root, rel_path)
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File in manifest missing from disk: {rel_path}")

        actual_size = os.path.getsize(file_path)
        if actual_size != meta.get("bytes"):
            raise ValueError(
                f"Byte size mismatch for {rel_path}: actual {actual_size} != expected {meta.get('bytes')}"
            )

        actual_hash = check_file_sha(file_path)
        if actual_hash != meta.get("sha256"):
            raise ValueError(
                f"SHA-256 mismatch for {rel_path}: actual {actual_hash} != expected {meta.get('sha256')}"
            )

    return len(files)


def check_file_sha(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=" * 70)
    print("15-Qubit Experiment Artifact & Reproducibility Verification")
    print("=" * 70)

    data_dir = os.path.join(base_dir, "data")
    results_dir = os.path.join(base_dir, "results")
    ckpt_dir = os.path.join(base_dir, "checkpoints")

    # 1. Validate Protocol SHA-256
    protocol_path = os.path.join(base_dir, "PROTOCOL.md")
    print("\n[1/11] Validating Protocol SHA-256...")
    actual_proto_sha = check_file_sha(protocol_path)
    print(f"  PROTOCOL.md SHA-256: {actual_proto_sha}")
    assert actual_proto_sha == FROZEN_PROTOCOL_SHA, (
        f"Protocol SHA mismatch! Expected {FROZEN_PROTOCOL_SHA}, got {actual_proto_sha}"
    )
    print("  ✓ Protocol SHA-256 verified identical.")

    # 2. Validate Source Gutenberg Text SHA-256 and byte size
    raw_source_path = os.path.join(data_dir, "gutenberg_100_raw.txt")
    print("\n[2/11] Validating Gutenberg Ebook #100 source...")
    actual_source_sha = check_file_sha(raw_source_path)
    actual_source_bytes = os.path.getsize(raw_source_path)
    print(f"  Source bytes: {actual_source_bytes} (expected {EXPECTED_SOURCE_BYTES})")
    print(f"  Source SHA:   {actual_source_sha}")
    assert actual_source_bytes == EXPECTED_SOURCE_BYTES, f"Source byte count mismatch: {actual_source_bytes}"
    assert actual_source_sha == EXPECTED_SOURCE_SHA, f"Source SHA mismatch: {actual_source_sha}"
    print("  ✓ Gutenberg source byte count and SHA-256 verified identical.")

    # 3. Validate Candidate and Selection Manifest Counts
    cand_path = os.path.join(results_dir, "candidate_validation_results.json")
    manifest_path = os.path.join(results_dir, "selection_manifest.json")
    print("\n[3/11] Validating candidate (180) and selection manifest (60) counts...")
    with open(cand_path, "r") as f:
        candidate_records = json.load(f)
    with open(manifest_path, "r") as f:
        selection_manifest = json.load(f)

    assert len(candidate_records) == 180, f"Expected 180 candidates, found {len(candidate_records)}"
    selected_checkpoints = selection_manifest["selected_checkpoints"]
    assert len(selected_checkpoints) == 60, f"Expected 60 selections, found {len(selected_checkpoints)}"
    assert selection_manifest["total_candidates"] == 180
    assert selection_manifest["total_selected"] == 60
    assert selection_manifest["frozen_protocol_sha256"] == FROZEN_PROTOCOL_SHA
    print(f"  ✓ Found {len(candidate_records)} candidates and {len(selected_checkpoints)} selected checkpoints.")

    # 4. Validate All Checkpoint Content Hashes (arr.tobytes())
    print("\n[4/11] Validating content hashes (arr.tobytes()) for all 180 candidate checkpoints...")
    for idx, cand in enumerate(candidate_records):
        arr, res_path = resolve_checkpoint_path(cand, ckpt_dir=ckpt_dir)
        actual_sha = sha256_of_array(arr)
        expected_sha = cand["checkpoint_sha256"]
        assert actual_sha == expected_sha, (
            f"Candidate checkpoint hash mismatch for {res_path}!\n"
            f"  Expected: {expected_sha}\n  Actual:   {actual_sha}"
        )
    print(f"  ✓ All 180 candidate checkpoints verified against recorded SHA-256.")

    print("  Validating content hashes for 60 selected checkpoints...")
    for idx, sel in enumerate(selected_checkpoints):
        arr, res_path = resolve_checkpoint_path(sel, ckpt_dir=ckpt_dir)
        actual_sha = sha256_of_array(arr)
        expected_sha = sel["checkpoint_sha256"]
        assert actual_sha == expected_sha, (
            f"Selected checkpoint hash mismatch for {res_path}!\n"
            f"  Expected: {expected_sha}\n  Actual:   {actual_sha}"
        )
    print("  ✓ All 60 selected checkpoints verified against recorded SHA-256.")

    # 5. Validate Three Bijections in encoding_maps.json
    print("\n[5/11] Validating 3 encoding bijections in encoding_maps.json...")
    maps_path = os.path.join(data_dir, "encoding_maps.json")
    with open(maps_path, "r") as f:
        encoding_maps = json.load(f)

    for enc_name in EXPECTED_ENCODINGS:
        assert enc_name in encoding_maps, f"Missing encoding: {enc_name}"
        c2code = encoding_maps[enc_name]["char_to_code"]
        code2c = encoding_maps[enc_name]["code_to_char"]
        assert len(c2code) == 32, f"Encoding {enc_name} has length {len(c2code)} != 32"
        assert set(c2code.keys()) == set(ALPHABET), f"Encoding {enc_name} keys do not match ALPHABET"
        assert set(c2code.values()) == set(range(32)), f"Encoding {enc_name} values do not span 0..31"
        for ch, code in c2code.items():
            assert code2c[str(code)] == ch, f"Inverse map mismatch in {enc_name} for char '{ch}'"
        print(f"  ✓ Encoding '{enc_name}': 32-symbol bijection verified.")

    # 6. Validate 5 Seeds
    print("\n[6/11] Validating 5 seeds...")
    manifest_seeds = sorted(list(set(r["seed"] for r in candidate_records)))
    assert manifest_seeds == EXPECTED_SEEDS, f"Seed mismatch: {manifest_seeds} != {EXPECTED_SEEDS}"
    print(f"  ✓ Seeds verified: {manifest_seeds}")

    # 7. Validate Selection Argmin and Tie-Breaking Rule
    print("\n[7/11] Validating selection manifest argmin and tie-breaking rule from candidates...")
    expected_selections_map = {}
    for r in candidate_records:
        key = (r["model"], r["encoding"], r["seed"])
        if key not in expected_selections_map:
            expected_selections_map[key] = []
        expected_selections_map[key].append(r)

    reconstructed_selections = []
    for key, cands in expected_selections_map.items():
        cands.sort(key=lambda x: x["config_index"])
        best = cands[0]
        for c in cands[1:]:
            if c["validation_log_loss_chunk"] < best["validation_log_loss_chunk"] - 1e-12:
                best = c
        reconstructed_selections.append(best)

    assert len(reconstructed_selections) == 60
    for sel_manifest in selected_checkpoints:
        key = (sel_manifest["model"], sel_manifest["encoding"], sel_manifest["seed"])
        matching = [r for r in reconstructed_selections if (r["model"], r["encoding"], r["seed"]) == key]
        assert len(matching) == 1, f"Did not find unique candidate for {key}"
        best_cand = matching[0]
        assert sel_manifest["selected_config_index"] == best_cand["config_index"]
        assert sel_manifest["selected_config_value"] == best_cand["config_value"]
        assert abs(sel_manifest["validation_log_loss_chunk"] - best_cand["validation_log_loss_chunk"]) < 1e-12
        assert abs(sel_manifest["validation_log_loss_char"] - best_cand["validation_log_loss_char"]) < 1e-12
        assert sel_manifest["checkpoint_sha256"] == best_cand["checkpoint_sha256"]
    print("  ✓ Selection manifest strictly matches candidate argmin validation log loss and tie rule.")

    # 8. Validate Disjoint Passage Groups Across Splits
    print("\n[8/11] Validating disjoint passage groups in splits_manifest.json...")
    splits_path = os.path.join(data_dir, "splits_manifest.json")
    with open(splits_path, "r") as f:
        splits_manifest = json.load(f)

    all_passages = splits_manifest["passages"]
    train_passages = set(p["passage_id"] for p in all_passages if p["split"] == "train")
    val_passages = set(p["passage_id"] for p in all_passages if p["split"] == "val")
    test_passages = set(p["passage_id"] for p in all_passages if p["split"] == "test")

    train_groups = set(p["group_sha256"] for p in all_passages if p["split"] == "train")
    val_groups = set(p["group_sha256"] for p in all_passages if p["split"] == "val")
    test_groups = set(p["group_sha256"] for p in all_passages if p["split"] == "test")

    print(f"  Train: {len(train_passages)} passages ({len(train_groups)} groups)")
    print(f"  Val:   {len(val_passages)} passages ({len(val_groups)} groups)")
    print(f"  Test:  {len(test_passages)} passages ({len(test_groups)} groups)")

    assert len(train_passages & val_passages) == 0, "Train and Val passages overlap!"
    assert len(train_passages & test_passages) == 0, "Train and Test passages overlap!"
    assert len(val_passages & test_passages) == 0, "Val and Test passages overlap!"

    assert len(train_groups & val_groups) == 0, "Train and Val groups overlap!"
    assert len(train_groups & test_groups) == 0, "Train and Val groups overlap!"
    assert len(val_groups & test_groups) == 0, "Val and Test groups overlap!"

    assert len(train_passages) == splits_manifest["n_train_passages"]
    assert len(val_passages) == splits_manifest["n_val_passages"]
    assert len(test_passages) == splits_manifest["n_test_passages"]
    print(f"  ✓ Train ({len(train_passages)}), Val ({len(val_passages)}), Test ({len(test_passages)}) passages and groups are strictly disjoint.")

    # 9. Recompute Selected Exact Validation & Test Metrics from Frozen Checkpoints & Source
    print("\n[9/11] Recomputing exact validation & test metrics from source and frozen checkpoints...")
    passages, splits, enc_maps, gen_manifest = prepare_corpus_and_splits(raw_source_path, output_dir=None, write_artifacts=False)

    # Compare regenerated manifests with frozen JSON before relying on them
    splits_json_path = os.path.join(data_dir, "splits_manifest.json")
    with open(splits_json_path, "r", encoding="utf-8") as f:
        frozen_splits = json.load(f)
    assert gen_manifest == frozen_splits, "Regenerated splits manifest does not match frozen data/splits_manifest.json"

    maps_json_path = os.path.join(data_dir, "encoding_maps.json")
    with open(maps_json_path, "r", encoding="utf-8") as f:
        frozen_maps = json.load(f)
    assert json.loads(json.dumps(enc_maps)) == frozen_maps, "Regenerated encoding maps do not match frozen data/encoding_maps.json"
    encoded_data = {}
    for enc_name in EXPECTED_ENCODINGS:
        c2code = enc_maps[enc_name]["char_to_code"]
        tr_counts, _ = extract_chunks_and_counts(splits["train"], c2code)
        val_counts, _ = extract_chunks_and_counts(splits["val"], c2code)
        te_counts, _ = extract_chunks_and_counts(splits["test"], c2code)
        encoded_data[enc_name] = {
            "train_counts": tr_counts,
            "val_counts": val_counts,
            "test_counts": te_counts,
        }

    k_hat = get_kernel_hat(sigmas=[1.0, 2.0, 4.0], n_qubits=15)
    gates = get_standard_gates(15)
    all_bits = np.array([[(s >> (14 - i)) & 1 for i in range(15)] for s in range(32768)], dtype=np.float64)

    test_results_path = os.path.join(results_dir, "test_results_summary.json")
    with open(test_results_path, "r") as f:
        original_test_results = json.load(f)

    test_results_lookup = {
        (r["model"], r["encoding"], r["seed"]): r for r in original_test_results
    }

    max_val_chunk_diff = 0.0
    max_val_char_diff = 0.0
    max_test_chunk_diff = 0.0
    max_test_char_diff = 0.0
    max_test_unsmoothed_diff = 0.0
    max_mmd_diff = 0.0
    max_unseen_diff = 0.0
    max_invalid_diff = 0.0

    print("  Evaluating all 60 selected models...")
    for idx, sel in enumerate(selected_checkpoints):
        model_name = sel["model"]
        enc_name = sel["encoding"]
        seed = sel["seed"]

        params, _ = resolve_checkpoint_path(sel, ckpt_dir=ckpt_dir)

        if model_name == "IQP":
            probs_raw = probs_from_iqp(params, gates, 15)
        elif model_name == "Bitflip":
            probs_raw = probs_from_bitflip(params, gates, 15)
        elif model_name == "Autoregressive":
            biases = params[:15]
            weights = params[15:]
            probs_raw = probs_from_autoregressive(biases, weights, all_bits, 15)
        else:
            probs_raw = params

        probs_raw = np.clip(probs_raw, 0.0, None)
        total_p = np.sum(probs_raw)
        probs_raw /= total_p

        val_counts = encoded_data[enc_name]["val_counts"]
        v_loss_chunk, v_loss_char, _ = evaluate_log_loss(probs_raw, val_counts, smoothing_eps=1e-6)

        val_chunk_diff = abs(v_loss_chunk - sel["validation_log_loss_chunk"])
        val_char_diff = abs(v_loss_char - sel["validation_log_loss_char"])
        max_val_chunk_diff = max(max_val_chunk_diff, val_chunk_diff)
        max_val_char_diff = max(max_val_char_diff, val_char_diff)

        assert val_chunk_diff <= 1e-9, (
            f"Validation chunk loss diff for {model_name} {enc_name} s{seed} exceeds 1e-9: {val_chunk_diff}"
        )
        assert val_char_diff <= 1e-9, (
            f"Validation char loss diff for {model_name} {enc_name} s{seed} exceeds 1e-9: {val_char_diff}"
        )

        orig_test = test_results_lookup[(model_name, enc_name, seed)]
        test_counts = encoded_data[enc_name]["test_counts"]
        t_loss_chunk, t_loss_char, t_unsmoothed = evaluate_log_loss(probs_raw, test_counts, smoothing_eps=1e-6)

        test_q_emp = test_counts / np.sum(test_counts)
        recomputed_mmd = exact_squared_mmd(probs_raw, test_q_emp, k_hat=k_hat, n_qubits=15)

        train_absent = (encoded_data[enc_name]["train_counts"] == 0)
        recomputed_unseen = float(np.sum(probs_raw[train_absent]))
        recomputed_invalid = 0.0

        t_chunk_diff = abs(t_loss_chunk - orig_test["test_log_loss_chunk"])
        t_char_diff = abs(t_loss_char - orig_test["test_log_loss_char"])
        t_unsmoothed_diff = abs(t_unsmoothed - orig_test["test_unsmoothed_chunk"])
        mmd_diff = abs(recomputed_mmd - orig_test["test_squared_mmd"])
        unseen_diff = abs(recomputed_unseen - orig_test["unseen_valid_mass"])
        invalid_diff = abs(recomputed_invalid - orig_test["invalid_mass"])

        max_test_chunk_diff = max(max_test_chunk_diff, t_chunk_diff)
        max_test_char_diff = max(max_test_char_diff, t_char_diff)
        max_test_unsmoothed_diff = max(max_test_unsmoothed_diff, t_unsmoothed_diff)
        max_mmd_diff = max(max_mmd_diff, mmd_diff)
        max_unseen_diff = max(max_unseen_diff, unseen_diff)
        max_invalid_diff = max(max_invalid_diff, invalid_diff)

        assert t_chunk_diff <= 1e-9, f"Test chunk loss diff exceeds 1e-9: {t_chunk_diff}"
        assert t_char_diff <= 1e-9, f"Test char loss diff exceeds 1e-9: {t_char_diff}"
        assert t_unsmoothed_diff <= 1e-9, f"Test unsmoothed loss diff exceeds 1e-9: {t_unsmoothed_diff}"
        assert mmd_diff <= 1e-9, f"Test MMD diff exceeds 1e-9: {mmd_diff}"
        assert unseen_diff <= 1e-9, f"Unseen mass diff exceeds 1e-9: {unseen_diff}"
        assert invalid_diff <= 1e-9, f"Invalid mass diff exceeds 1e-9: {invalid_diff}"

    print(f"  ✓ Max validation chunk loss diff:      {max_val_chunk_diff:.2e} (<= 1e-9)")
    print(f"  ✓ Max validation char loss diff:       {max_val_char_diff:.2e} (<= 1e-9)")
    print(f"  ✓ Max test chunk loss diff:            {max_test_chunk_diff:.2e} (<= 1e-9)")
    print(f"  ✓ Max test char loss diff:             {max_test_char_diff:.2e} (<= 1e-9)")
    print(f"  ✓ Max test unsmoothed chunk loss diff: {max_test_unsmoothed_diff:.2e} (<= 1e-9)")
    print(f"  ✓ Max test squared MMD diff:           {max_mmd_diff:.2e} (<= 1e-9)")
    print(f"  ✓ Max unseen valid mass diff:          {max_unseen_diff:.2e} (<= 1e-9)")
    print(f"  ✓ Max invalid mass diff:               {max_invalid_diff:.2e} (<= 1e-9)")

    # 10. Export Integrity Manifest Verification
    print("\n[10/11] Validating export integrity manifest (export_manifest.json)...")
    manifest_file = os.path.join(base_dir, "export_manifest.json")
    assert os.path.exists(manifest_file), "export_manifest.json not found in repository root"
    n_verified = validate_export_manifest(manifest_file, repo_root=base_dir)
    print(f"  ✓ Export manifest verified: {n_verified} files intact, 0 mismatches.")

    # 11. Provenance & Archival Status Documentation (Pre-registration Limits)
    print("\n[11/11] Documenting research provenance and pre-registration limits...")
    sel_manifest_file_sha = check_file_sha(manifest_path)
    ref_proto_sha = selection_manifest.get("frozen_protocol_sha256")
    print("  Archival research metadata:")
    print(f"    - Protocol file SHA-256:            {FROZEN_PROTOCOL_SHA}")
    print(f"    - Selection manifest file SHA-256:  {sel_manifest_file_sha}")
    print(f"    - Embedded protocol reference SHA:  {ref_proto_sha}")
    print()
    print("  [PROVENANCE & INDEPENDENT VERIFICATION CAVEAT]")
    print("  Chronological commit ordering was recorded locally during original research.")
    print("  In this standalone exported repository, git chronology is an archival assertion")
    print("  that cannot be independently proven from local history alone.")
    print("  Because local git commits lack third-party attestation,")
    print("  no claim of external pre-registration is made.")
    print("  All numerical checks, content hashes, and reproducibility tests have PASSED.")

    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE: 10/10 numerical and structural checks passed.")
    print("Scientific Reproducibility: VERIFIED PASS (0 errors, 0 writes over original).")
    print("Archival Chronology:        ASSERTION (unnotarized local research timeline).")
    print("=" * 70)


if __name__ == "__main__":
    main()
