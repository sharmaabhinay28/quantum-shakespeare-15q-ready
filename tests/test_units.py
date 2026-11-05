"""Unit tests for 15-qubit quantum language model and reproduction components."""

import os
import sys
import json
import copy
import hashlib
import tempfile
import pytest
import numpy as np
import jax
import jax.numpy as jnp
from iqpopt import IqpSimulator

# Ensure repo root is on sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.evaluation import (
    fwht,
    gate_to_mask,
    get_standard_gates,
    probs_from_iqp,
    probs_from_bitflip,
    probs_from_autoregressive,
    probs_from_table,
    exact_squared_mmd,
    get_kernel_hat,
    evaluate_log_loss
)
from src.data import ALPHABET, build_encoding_maps, encode_chunk, prepare_corpus_and_splits
from src.models import AutoregressiveBernoulliModel
from src.resolver import resolve_checkpoint_path, sha256_of_array
from src.guard import guard_output_dir, get_protected_dirs
from verify_artifacts import validate_export_manifest, check_file_sha


def test_alphabet_and_encodings():
    """Verify alphabet is 32 chars and encodings are bijections."""
    assert len(ALPHABET) == 32
    maps = build_encoding_maps()
    for enc_name in ["primary", "alt1", "alt2"]:
        c2code = maps[enc_name]["char_to_code"]
        code2c = maps[enc_name]["code_to_char"]
        assert len(c2code) == 32
        assert len(code2c) == 32
        codes = set(c2code.values())
        assert codes == set(range(32))
        for ch, code in c2code.items():
            assert code2c[code] == ch

    # Test chunk encoding
    c0, c1, c2 = "t", "h", "e"
    code = encode_chunk(c0, c1, c2, maps["primary"]["char_to_code"])
    assert 0 <= code < 32768


def test_fwht_vs_pennylane_small():
    """Verify FWHT matches PennyLane default.qubit on small 4-qubit circuit."""
    n_qubits = 4
    gates = [[[i]] for i in range(n_qubits)] + [[[i, j]] for i in range(n_qubits) for j in range(i+1, n_qubits)]
    params = np.array([0.1, -0.2, 0.05, 0.3, -0.15, 0.22, 0.08, -0.12, 0.18, -0.05])

    sim = IqpSimulator(n_qubits, gates, device="default.qubit")
    probs_pl = np.array(sim.probs(jnp.array(params)))
    probs_exact = probs_from_iqp(params, gates, n_qubits=n_qubits)

    diff = np.max(np.abs(probs_pl - probs_exact))
    assert diff < 1e-12, f"FWHT differs from PennyLane: {diff}"
    assert np.isclose(np.sum(probs_exact), 1.0, atol=1e-10)


def test_fwht_vs_pennylane_15q():
    """Verify FWHT matches PennyLane default.qubit on 15 qubits checkpoint."""
    n_qubits = 15
    gates = get_standard_gates(n_qubits)
    rng = np.random.Generator(np.random.PCG64(42))
    params = rng.uniform(-0.1, 0.1, len(gates))

    sim = IqpSimulator(n_qubits, gates, device="default.qubit")
    probs_pl = np.array(sim.probs(jnp.array(params)))
    probs_exact = probs_from_iqp(params, gates, n_qubits=n_qubits)

    diff = np.max(np.abs(probs_pl - probs_exact))
    assert diff < 1e-12, f"15-qubit FWHT differs from PennyLane: {diff}"
    assert np.isclose(np.sum(probs_exact), 1.0, atol=1e-10)


def test_bitflip_analytic_vs_expvals_and_simulation():
    """Verify bitflip exact probabilities against package op_expval and XOR simulation."""
    n_qubits = 4
    gates = [[[i]] for i in range(n_qubits)] + [[[i, j]] for i in range(n_qubits) for j in range(i+1, n_qubits)]
    params = np.array([0.1, -0.2, 0.05, 0.3, -0.15, 0.22, 0.08, -0.12, 0.18, -0.05])

    sim_bf = IqpSimulator(n_qubits, gates, bitflip=True)
    all_ops = np.array([[(s >> (n_qubits - 1 - i)) & 1 for i in range(n_qubits)] for s in range(1 << n_qubits)])

    expvals_pkg, _ = sim_bf.op_expval(jnp.array(params), jnp.array(all_ops), n_samples=1, key=jax.random.PRNGKey(0))
    probs_exact = probs_from_bitflip(params, gates, n_qubits=n_qubits)

    expvals_from_probs = fwht(probs_exact)
    diff_exp = np.max(np.abs(np.array(expvals_pkg) - expvals_from_probs))
    assert diff_exp < 1e-12, f"Bitflip expectations differ: {diff_exp}"
    assert np.isclose(np.sum(probs_exact), 1.0, atol=1e-10)
    assert np.all(probs_exact >= 0.0)


def test_autoregressive_bernoulli():
    """Verify autoregressive Bernoulli exact probabilities sum to 1."""
    n_qubits = 15
    model = AutoregressiveBernoulliModel(n_qubits=n_qubits)
    assert model.n_params == 120
    params = model.init_params(seed=42)
    probs = model.get_probs(params)
    assert len(probs) == 32768
    assert np.isclose(np.sum(probs), 1.0, atol=1e-10)
    assert np.all(probs >= 0.0)


def test_exact_squared_mmd_vs_quadratic_form():
    """Verify exact squared MMD via FWHT matches dense quadratic form on 6 qubits."""
    n = 6
    num_states = 1 << n
    sigmas = [1.0, 2.0, 4.0]

    k_dense = np.zeros((num_states, num_states))
    for x in range(num_states):
        for y in range(num_states):
            h = bin(x ^ y).count('1')
            k_dense[x, y] = np.mean([np.exp(-h / (2.0 * s**2)) for s in sigmas])

    rng = np.random.Generator(np.random.PCG64(99))
    p = rng.dirichlet(np.ones(num_states))
    q = rng.dirichlet(np.ones(num_states))
    delta = p - q

    mmd2_dense = delta.T @ k_dense @ delta
    k_hat = get_kernel_hat(sigmas=sigmas, n_qubits=n)
    mmd2_fwht = exact_squared_mmd(p, q, k_hat=k_hat, n_qubits=n)

    assert abs(mmd2_dense - mmd2_fwht) < 1e-12


def test_frequency_table_smoothing():
    """Verify frequency table smoothing behavior."""
    counts = np.zeros(32768, dtype=np.int64)
    counts[10] = 50
    counts[200] = 50
    p = probs_from_table(counts, alpha=0.1, num_states=32768)
    assert np.isclose(np.sum(p), 1.0, atol=1e-10)
    assert p[10] > p[0]
    assert np.all(p > 0.0)


def test_corrupt_checkpoint_copy_fails_without_fallback():
    """Controlled negative test: a corrupted copy of a checkpoint MUST fail hash validation.

    Ensures that resolve_checkpoint_path raises ValueError and does not fall back
    to the original or any external file. The test fixture is retained without deletion.
    """
    fixtures_dir = os.path.join(os.path.dirname(__file__), 'fixtures')
    corrupt_fixture = os.path.join(fixtures_dir, 'corrupt_checkpoint_copy.npy')
    assert os.path.exists(corrupt_fixture), f"Fixture missing: {corrupt_fixture}"

    original_sha256 = "a408f50b3c6572c206a174b1203cd707df75e035d571395ad564608ca2c96ab7"

    with pytest.raises(ValueError) as exc_info:
        resolve_checkpoint_path("corrupt_checkpoint_copy.npy", expected_sha256=original_sha256, ckpt_dir=fixtures_dir)

    assert "Content hash mismatch" in str(exc_info.value)
    assert os.path.exists(corrupt_fixture)


def test_resolver_deterministic_rules():
    """Validates strict deterministic resolution and rejection of invalid/traversal paths."""
    fixtures_dir = os.path.join(os.path.dirname(__file__), 'fixtures')

    # 1. Absolute paths rejected
    with pytest.raises(ValueError) as exc1:
        resolve_checkpoint_path(os.path.abspath("nonexistent_arbitrary_checkpoint.npy"))
    assert "Absolute checkpoint paths are rejected" in str(exc1.value)

    # 2. Path traversal rejected
    with pytest.raises(ValueError) as exc2:
        resolve_checkpoint_path("../checkpoints/IQP_primary_s11_cfg0.npy")
    assert "Path traversal ('..') is strictly rejected" in str(exc2.value)

    # 3. Empty reference rejected
    with pytest.raises(ValueError) as exc3:
        resolve_checkpoint_path("")
    assert "Empty checkpoint reference" in str(exc3.value)

    # 4. Explicit ckpt_dir is sole authority: missing file raises FileNotFoundError
    with pytest.raises(FileNotFoundError) as exc4:
        resolve_checkpoint_path("nonexistent_checkpoint.npy", ckpt_dir=fixtures_dir)
    assert "not found in authoritative ckpt_dir" in str(exc4.value)


def test_guard_rejects_protected_and_existing_destinations(tmp_path):
    """Validates that guard_output_dir refuses writes to protected or existing directories."""
    # 1. Rejects repo protected directories
    for pdir in ["results", "checkpoints", "data"]:
        target = os.path.join(repo_root, pdir)
        with pytest.raises(ValueError) as exc:
            guard_output_dir(target, repo_root=repo_root)
        assert "inside protected directory" in str(exc.value)

    # Subdirectory of protected
    with pytest.raises(ValueError) as exc:
        guard_output_dir(os.path.join(repo_root, "results", "sub"), repo_root=repo_root)
    assert "inside protected directory" in str(exc.value)

    # 2. Rejects existing non-empty directory
    existing_dir = tmp_path / "existing_run"
    existing_dir.mkdir()
    (existing_dir / "marker.txt").write_text("content")

    with pytest.raises(ValueError) as exc:
        guard_output_dir(str(existing_dir), repo_root=repo_root, allow_existing=False)
    assert "already exists and is non-empty" in str(exc.value)

    # 3. Allows clean non-existent path
    clean_target = tmp_path / "new_run_clean"
    resolved = guard_output_dir(str(clean_target), repo_root=repo_root, allow_existing=False)
    assert resolved == str(clean_target)


def test_data_preparation_zero_writes_and_mtimes_unchanged():
    """Proves prepare_corpus_and_splits with write_artifacts=False performs zero writes."""
    data_dir = os.path.join(repo_root, "data")
    raw_path = os.path.join(data_dir, "gutenberg_100_raw.txt")
    tracked_files = [
        os.path.join(data_dir, "gutenberg_100_raw.txt"),
        os.path.join(data_dir, "splits_manifest.json"),
        os.path.join(data_dir, "encoding_maps.json"),
        os.path.join(data_dir, "gutenberg_source_metadata.json"),
    ]

    before_mtimes = {f: os.path.getmtime(f) for f in tracked_files}
    before_shas = {f: check_file_sha(f) for f in tracked_files}

    passages, splits, maps, manifest = prepare_corpus_and_splits(
        raw_path, output_dir=None, write_artifacts=False
    )

    after_mtimes = {f: os.path.getmtime(f) for f in tracked_files}
    after_shas = {f: check_file_sha(f) for f in tracked_files}

    assert before_mtimes == after_mtimes, "mtimes changed during read-only data preparation!"
    assert before_shas == after_shas, "File content hashes changed during read-only data preparation!"


def test_export_manifest_validator_and_tamper_detection():
    """Validates export manifest integrity helper and verifies tamper detection."""
    manifest_path = os.path.join(repo_root, 'export_manifest.json')
    assert os.path.exists(manifest_path), "export_manifest.json must exist in repository root"

    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    # 1. Real manifest validates completely with full file coverage
    n_verified = validate_export_manifest(manifest, repo_root=repo_root)
    assert n_verified >= 195, f"Expected at least 195 material files in manifest, got {n_verified}"

    # 2. Tamper Test A: Modified SHA-256 triggers ValueError
    tampered_sha_manifest = copy.deepcopy(manifest)
    first_file = next(iter(tampered_sha_manifest["files"]))
    tampered_sha_manifest["files"][first_file]["sha256"] = "0" * 64
    with pytest.raises(ValueError) as exc_sha:
        validate_export_manifest(tampered_sha_manifest, repo_root=repo_root)
    assert "SHA-256 mismatch" in str(exc_sha.value)

    # 3. Tamper Test B: Modified byte count triggers ValueError
    tampered_bytes_manifest = copy.deepcopy(manifest)
    tampered_bytes_manifest["files"][first_file]["bytes"] += 12345
    with pytest.raises(ValueError) as exc_bytes:
        validate_export_manifest(tampered_bytes_manifest, repo_root=repo_root)
    assert "Byte size mismatch" in str(exc_bytes.value)

    # 4. Tamper Test C: Non-existent file triggers FileNotFoundError
    tampered_missing_manifest = copy.deepcopy(manifest)
    tampered_missing_manifest["files"]["nonexistent_file_test.txt"] = {"bytes": 10, "sha256": "a" * 64}
    with pytest.raises(FileNotFoundError) as exc_missing:
        validate_export_manifest(tampered_missing_manifest, repo_root=repo_root)
    assert "missing from disk" in str(exc_missing.value)

    # 5. Tamper Test D: Empty manifest triggers ValueError
    with pytest.raises(ValueError) as exc_empty:
        validate_export_manifest({"files": {}}, repo_root=repo_root)
    assert "empty" in str(exc_empty.value)

    # 6. Tamper Test E: Self-recursive manifest triggers ValueError
    tampered_recursive = copy.deepcopy(manifest)
    tampered_recursive["files"]["export_manifest.json"] = {"bytes": 10, "sha256": "b" * 64}
    with pytest.raises(ValueError) as exc_rec:
        validate_export_manifest(tampered_recursive, repo_root=repo_root)
    assert "must not be self-recursive" in str(exc_rec.value)
