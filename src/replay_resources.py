#!/usr/bin/env python3
"""Instrumented Replay of Frozen Configurations in Isolated Phase Workers.

Measures clean, isolated resource costs (wall time and isolated peak RSS)
for the selected primary-encoding seed-11 configuration across all 4 models:
  - Training excluding get_probs, with cold JIT included but not separately timed
  - Exact evaluation (prob vector construction + exact metrics)
  - 10k sampling (prob construction vs categorical sampling separated)
  - Data preparation baseline in an isolated worker

Saves:
  - results/resource_replay.json
  - results/resource_replay.csv
  - results/plot_resource_costs_corrected.png
"""

import os
import sys
import json
import time
import subprocess
import resource
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from src.guard import guard_output_dir
from src.resolver import resolve_checkpoint_path, sha256_of_array


def get_peak_rss_mb():
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return usage / (1024 * 1024)
    else:
        return usage / 1024


WORKER_SCRIPT = os.path.join(tempfile.mkdtemp(prefix="quantum-replay-worker-"), "worker.py")


def create_worker_script():
    content = '''import os
import sys
import time
import json
import resource
import numpy as np

base_dir = os.environ["QUANTUM_ARTIFACT_DIR"]
sys.path.insert(0, base_dir)

from src.guard import guard_output_dir
from src.resolver import resolve_checkpoint_path, sha256_of_array
from src.data import prepare_corpus_and_splits, extract_chunks_and_counts
from src.models import (
    create_training_sampler,
    IQPModel,
    StochasticBitflipModel,
    AutoregressiveBernoulliModel,
    FrequencyTableModel
)
from src.evaluation import (
    get_standard_gates,
    get_kernel_hat,
    exact_squared_mmd,
    evaluate_log_loss,
    probs_from_iqp,
    probs_from_bitflip,
    probs_from_autoregressive,
    probs_from_table
)

def get_peak_rss_mb():
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return usage / (1024 * 1024)
    else:
        return usage / 1024

def run_data_prep():
    t0 = time.perf_counter()
    raw_path = os.path.join(base_dir, "data", "gutenberg_100_raw.txt")
    data_dir = os.path.join(base_dir, "data")
    passages, splits, maps, manifest = prepare_corpus_and_splits(raw_path, output_dir=None, write_artifacts=False)
    c2code = maps["primary"]["char_to_code"]
    tr_counts, _ = extract_chunks_and_counts(splits["train"], c2code)
    val_counts, _ = extract_chunks_and_counts(splits["val"], c2code)
    te_counts, _ = extract_chunks_and_counts(splits["test"], c2code)
    t1 = time.perf_counter()
    rss = get_peak_rss_mb()
    return {
        "phase": "data_prep",
        "wall_seconds": t1 - t0,
        "peak_rss_mb": rss
    }

def run_pure_train(model_name, enc_name, seed, cfg_idx, cfg_val, ckpt_sha):
    raw_path = os.path.join(base_dir, "data", "gutenberg_100_raw.txt")
    data_dir = os.path.join(base_dir, "data")
    _, splits, maps, _ = prepare_corpus_and_splits(raw_path, output_dir=None, write_artifacts=False)
    c2code = maps[enc_name]["char_to_code"]
    train_counts, _ = extract_chunks_and_counts(splits["train"], c2code)

    models_dict = {
        "IQP": IQPModel(15),
        "Bitflip": StochasticBitflipModel(15),
        "Autoregressive": AutoregressiveBernoulliModel(15),
        "Table": FrequencyTableModel(15)
    }
    model_obj = models_dict[model_name]

    p_init = None
    if model_name != "Table":
        p_init = model_obj.init_params(seed)

    sampler = create_training_sampler(train_counts, seed=seed)
    # Advance sampler for previous configs in loop order
    advance_batches = cfg_idx * 500
    for _ in range(advance_batches):
        _ = sampler(512)

    # Measure pure training
    t0 = time.perf_counter()
    if model_name in ["IQP", "Bitflip", "Autoregressive"]:
        # Time the complete fit; compilation is included, not separately measured.
        train_seed = seed + 1000 * (cfg_idx + 1)
        trained_params, losses = model_obj.train(
            p_init, sampler, lr=cfg_val, n_updates=500, seed=train_seed
        )
    else:
        probs_raw = model_obj.fit(train_counts, alpha=cfg_val)
        trained_params = probs_raw
        losses = []
    t1 = time.perf_counter()
    rss = get_peak_rss_mb()

    # Verify against frozen checkpoint
    actual_sha = sha256_of_array(trained_params)
    assert actual_sha == ckpt_sha, f"Trained params hash mismatch! {actual_sha} != {ckpt_sha}"

    # Load frozen checkpoint to compute exact float tolerance
    ref_rec = {"checkpoint_path": f"{model_name}_{enc_name}_s{seed}_cfg{cfg_idx}.npy", "checkpoint_sha256": ckpt_sha}
    frozen_p, _ = resolve_checkpoint_path(ref_rec, ckpt_dir=os.path.join(base_dir, "checkpoints"))
    max_param_diff = float(np.max(np.abs(trained_params - frozen_p)))

    return {
        "phase": "pure_training",
        "model": model_name,
        "encoding": enc_name,
        "seed": seed,
        "config_index": cfg_idx,
        "config_value": cfg_val,
        "pure_train_wall_seconds": round(t1 - t0, 5),
        "peak_rss_mb": round(rss, 2),
        "max_param_diff_vs_frozen": max_param_diff,
        "sha256_match": bool(actual_sha == ckpt_sha)
    }

def run_exact_eval(model_name, enc_name, seed, ckpt_sha):
    raw_path = os.path.join(base_dir, "data", "gutenberg_100_raw.txt")
    data_dir = os.path.join(base_dir, "data")
    _, splits, maps, _ = prepare_corpus_and_splits(raw_path, output_dir=None, write_artifacts=False)
    c2code = maps[enc_name]["char_to_code"]
    test_counts, _ = extract_chunks_and_counts(splits["test"], c2code)

    ref_rec = {"checkpoint_path": f"{model_name}_{enc_name}_s{seed}_cfg.npy", "checkpoint_sha256": ckpt_sha}
    # Resolve from manifest or basename
    manifest_path = os.path.join(base_dir, "results", "selection_manifest.json")
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    match = [s for s in manifest["selected_checkpoints"] if s["model"] == model_name and s["encoding"] == enc_name and s["seed"] == seed][0]
    params, _ = resolve_checkpoint_path(match, ckpt_dir=os.path.join(base_dir, "checkpoints"))

    k_hat = get_kernel_hat(sigmas=[1.0, 2.0, 4.0], n_qubits=15)
    gates = get_standard_gates(15)
    all_bits = np.array([[(s >> (14 - i)) & 1 for i in range(15)] for s in range(32768)], dtype=np.float64)

    t0_prob = time.perf_counter()
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
    t1_prob = time.perf_counter()

    probs_raw = np.clip(probs_raw, 0.0, None)
    probs_raw /= np.sum(probs_raw)

    t0_metrics = time.perf_counter()
    v_chunk, v_char, unsmoothed = evaluate_log_loss(probs_raw, test_counts, smoothing_eps=1e-6)
    test_q = test_counts / np.sum(test_counts)
    mmd2 = exact_squared_mmd(probs_raw, test_q, k_hat=k_hat, n_qubits=15)
    t1_metrics = time.perf_counter()

    rss = get_peak_rss_mb()
    return {
        "phase": "exact_evaluation",
        "model": model_name,
        "encoding": enc_name,
        "seed": seed,
        "prob_construction_wall_seconds": round(t1_prob - t0_prob, 5),
        "metrics_computation_wall_seconds": round(t1_metrics - t0_metrics, 5),
        "total_eval_wall_seconds": round(t1_metrics - t0_prob, 5),
        "peak_rss_mb": round(rss, 2),
        "test_log_loss_chunk": round(v_chunk, 6),
        "test_log_loss_char": round(v_char, 6),
        "test_squared_mmd": round(mmd2, 8)
    }

def run_sampling(model_name, enc_name, seed, ckpt_sha):
    manifest_path = os.path.join(base_dir, "results", "selection_manifest.json")
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    match = [s for s in manifest["selected_checkpoints"] if s["model"] == model_name and s["encoding"] == enc_name and s["seed"] == seed][0]
    params, _ = resolve_checkpoint_path(match, ckpt_dir=os.path.join(base_dir, "checkpoints"))

    gates = get_standard_gates(15)
    all_bits = np.array([[(s >> (14 - i)) & 1 for i in range(15)] for s in range(32768)], dtype=np.float64)

    t0_prob = time.perf_counter()
    if model_name == "IQP":
        probs_raw = probs_from_iqp(params, gates, 15)
    elif model_name == "Bitflip":
        probs_raw = probs_from_bitflip(params, gates, 15)
    elif model_name == "Autoregressive":
        probs_raw = probs_from_autoregressive(params[:15], params[15:], all_bits, 15)
    else:
        probs_raw = params
    t1_prob = time.perf_counter()

    probs_raw = np.clip(probs_raw, 0.0, None)
    probs_raw /= np.sum(probs_raw)

    rng = np.random.Generator(np.random.PCG64(seed + 100000))
    t0_samp = time.perf_counter()
    samples = rng.choice(32768, size=10000, replace=True, p=probs_raw)
    t1_samp = time.perf_counter()

    rss = get_peak_rss_mb()
    return {
        "phase": "10k_sampling",
        "model": model_name,
        "encoding": enc_name,
        "seed": seed,
        "sample_seed": seed + 100000,
        "prob_construction_wall_seconds": round(t1_prob - t0_prob, 5),
        "categorical_draw_wall_seconds": round(t1_samp - t0_samp, 5),
        "total_sampling_wall_seconds": round(t1_samp - t0_prob, 5),
        "peak_rss_mb": round(rss, 2),
        "n_samples": len(samples),
        "unique_samples": int(len(np.unique(samples)))
    }

if __name__ == "__main__":
    task = sys.argv[1]
    if task == "data_prep":
        res = run_data_prep()
    elif task == "train":
        m, enc, s, idx, val, sha = sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), float(sys.argv[6]), sys.argv[7]
        res = run_pure_train(m, enc, s, idx, val, sha)
    elif task == "eval":
        m, enc, s, sha = sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
        res = run_exact_eval(m, enc, s, sha)
    elif task == "sampling":
        m, enc, s, sha = sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
        res = run_sampling(m, enc, s, sha)
    else:
        raise ValueError(f"Unknown task: {task}")
    print(json.dumps(res))
'''
    with open(WORKER_SCRIPT, "w") as f:
        f.write(content)


def run_worker_cmd(cmd_args):
    proc = subprocess.run(
        [sys.executable, WORKER_SCRIPT] + cmd_args,
        capture_output=True,
        text=True,
        check=True,
        env=dict(os.environ, QUANTUM_ARTIFACT_DIR=base_dir)
    )
    # Parse last line as JSON
    lines = proc.stdout.strip().split("\n")
    return json.loads(lines[-1])


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Instrumented Replay of Frozen Configurations in Isolated Phase Workers.")
    parser.add_argument("--out-dir", required=True, help="Directory where replay results will be saved (must be new/unprotected)")
    args = parser.parse_args()

    out_dir = guard_output_dir(args.out_dir, repo_root=base_dir, allow_existing=False)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("Resource Replay: Instrumented Isolated-Process Resource Profiling")
    print("=" * 70)

    create_worker_script()

    manifest_path = os.path.join(base_dir, "results", "selection_manifest.json")
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    # Target models for representative replay: primary encoding, seed 11
    target_models = ["IQP", "Bitflip", "Autoregressive", "Table"]
    selected_map = {}
    for sel in manifest["selected_checkpoints"]:
        if sel["encoding"] == "primary" and sel["seed"] == 11:
            selected_map[sel["model"]] = sel

    # 1. Isolated Data Prep Baseline
    print("\n[Phase 1] Measuring isolated Data Prep baseline...")
    data_prep_res = run_worker_cmd(["data_prep"])
    print(f"  Data Prep: {data_prep_res['wall_seconds']:.2f}s, Peak RSS: {data_prep_res['peak_rss_mb']:.1f} MB")

    # 2. Pure Training in Isolated Workers
    print("\n[Phase 2] Measuring Pure Training (excluding get_probs) in isolated processes...")
    train_results = []
    for m in target_models:
        sel = selected_map[m]
        print(f"  Running isolated pure training for {m} (cfg_idx={sel['selected_config_index']}, cfg_val={sel['selected_config_value']})...")
        t_res = run_worker_cmd([
            "train",
            m,
            "primary",
            str(sel["seed"]),
            str(sel["selected_config_index"]),
            str(sel["selected_config_value"]),
            sel["checkpoint_sha256"]
        ])
        train_results.append(t_res)
        print(f"    -> Pure Train: {t_res['pure_train_wall_seconds']:.4f}s | Peak RSS: {t_res['peak_rss_mb']:.1f} MB | max_diff: {t_res['max_param_diff_vs_frozen']:.1e}")

    # 3. Exact Evaluation in Isolated Workers
    print("\n[Phase 3] Measuring Exact Evaluation (prob construction + metrics) in isolated processes...")
    eval_results = []
    for m in target_models:
        sel = selected_map[m]
        print(f"  Running isolated exact evaluation for {m}...")
        e_res = run_worker_cmd([
            "eval",
            m,
            "primary",
            str(sel["seed"]),
            sel["checkpoint_sha256"]
        ])
        eval_results.append(e_res)
        print(f"    -> Probs: {e_res['prob_construction_wall_seconds']:.4f}s | Metrics: {e_res['metrics_computation_wall_seconds']:.4f}s | Total: {e_res['total_eval_wall_seconds']:.4f}s | Peak RSS: {e_res['peak_rss_mb']:.1f} MB")

    # 4. 10k Sampling in Isolated Workers
    print("\n[Phase 4] Measuring 10k Sampling in isolated processes...")
    samp_results = []
    for m in target_models:
        sel = selected_map[m]
        print(f"  Running isolated 10k sampling for {m}...")
        s_res = run_worker_cmd([
            "sampling",
            m,
            "primary",
            str(sel["seed"]),
            sel["checkpoint_sha256"]
        ])
        samp_results.append(s_res)
        print(f"    -> Probs: {s_res['prob_construction_wall_seconds']:.4f}s | Draw: {s_res['categorical_draw_wall_seconds']:.4f}s | Total: {s_res['total_sampling_wall_seconds']:.4f}s | Peak RSS: {s_res['peak_rss_mb']:.1f} MB")

    # Combine into comprehensive replay accounting record
    replay_summary = {
        "metadata": {
            "workload": "representative_fixed_workload_replay",
            "scope": "primary_encoding_seed_11_selected_configs",
            "process_isolation": "dedicated_subprocess_per_phase_worker",
            "platform": "macOS-27.0.1-arm64-Apple-Silicon",
            "python_version": sys.version,
            "notes": (
                "Pure training isolates optimization loop from probability generation; "
                "Step 0 includes in-process JIT compilation for IQP, Bitflip, and AR; "
                "Table is deterministic closed-form add-alpha fitting without updates or randomness; "
                "Sampling records both probability vector construction and categorical draw separately; "
                "Peak RSS is the worker-lifetime high-water mark, including imports and corpus preparation. "
                "It is not incremental model memory. Operation timers exclude preparation and process startup. "
                "Compilation is included in cold training time, not separately measured."
            )
        },
        "isolated_data_prep_baseline": data_prep_res,
        "pure_training_isolated": train_results,
        "exact_evaluation_isolated": eval_results,
        "sampling_10k_isolated": samp_results
    }

    # Save replay results
    out_json = os.path.join(out_dir, "resource_replay.json")
    out_csv = os.path.join(out_dir, "resource_replay.csv")

    with open(out_json, "w") as f:
        json.dump(replay_summary, f, indent=2)
    print(f"\nSaved new replay JSON: {out_json}")

    # Build CSV rows
    csv_rows = []
    for m in target_models:
        t_row = [r for r in train_results if r["model"] == m][0]
        e_row = [r for r in eval_results if r["model"] == m][0]
        s_row = [r for r in samp_results if r["model"] == m][0]
        csv_rows.append({
            "model": m,
            "encoding": "primary",
            "seed": 11,
            "selected_config_index": t_row["config_index"],
            "selected_config_value": t_row["config_value"],
            "pure_train_wall_sec": t_row["pure_train_wall_seconds"],
            "train_peak_rss_mb": t_row["peak_rss_mb"],
            "eval_prob_construction_sec": e_row["prob_construction_wall_seconds"],
            "eval_metrics_sec": e_row["metrics_computation_wall_seconds"],
            "eval_total_sec": e_row["total_eval_wall_seconds"],
            "eval_peak_rss_mb": e_row["peak_rss_mb"],
            "sampling_prob_sec": s_row["prob_construction_wall_seconds"],
            "sampling_draw_sec": s_row["categorical_draw_wall_seconds"],
            "sampling_total_sec": s_row["total_sampling_wall_seconds"],
            "sampling_peak_rss_mb": s_row["peak_rss_mb"],
            "max_param_diff_vs_frozen": t_row["max_param_diff_vs_frozen"]
        })
    df_csv = pd.DataFrame(csv_rows)
    df_csv.to_csv(out_csv, index=False)
    print(f"Saved new replay CSV: {out_csv}")

    # Generate NEW corrected resource plot
    plot_path = os.path.join(out_dir, "plot_resource_costs_corrected.png")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    models = [r["model"] for r in csv_rows]
    train_times = [r["pure_train_wall_sec"] for r in csv_rows]
    eval_times = [r["eval_total_sec"] for r in csv_rows]
    samp_times = [r["sampling_total_sec"] for r in csv_rows]

    x = np.arange(len(models))
    width = 0.25

    # Subplot 1: Wall Times
    rects1 = ax1.bar(x - width, train_times, width, label='Fit (500 steps; table closed form)', color='#1f77b4', alpha=0.85)
    rects2 = ax1.bar(x, eval_times, width, label='Exact Evaluation (FWHT/Metrics)', color='#ff7f0e', alpha=0.85)
    rects3 = ax1.bar(x + width, samp_times, width, label='10k Sampling (Total)', color='#2ca02c', alpha=0.85)

    ax1.set_ylabel('Wall Time (seconds, log scale)')
    ax1.set_title('Isolated Phase Execution Wall Time per Model\n(Primary Encoding, Seed 11 Selected Config)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontweight='bold')
    ax1.set_yscale('log')
    ax1.legend(loc='upper right')
    ax1.grid(True, which="both", ls="--", alpha=0.3)

    # Subplot 2: Isolated Peak RSS
    train_rss = [r["train_peak_rss_mb"] for r in csv_rows]
    eval_rss = [r["eval_peak_rss_mb"] for r in csv_rows]
    samp_rss = [r["sampling_peak_rss_mb"] for r in csv_rows]

    ax2.bar(x - width, train_rss, width, label='Training Peak RSS', color='#1f77b4', alpha=0.85)
    ax2.bar(x, eval_rss, width, label='Evaluation Peak RSS', color='#ff7f0e', alpha=0.85)
    ax2.bar(x + width, samp_rss, width, label='Sampling Peak RSS', color='#2ca02c', alpha=0.85)

    ax2.set_ylabel('Isolated Process Peak RSS (MB)')
    ax2.set_title('Worker Peak Memory (High-Water RSS)\n(Includes Imports and Corpus Preparation)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, ls="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved new corrected resource plot: {plot_path}")

    # Retain the generated worker for inspection; the task forbids file deletion.
    print(f"Retained worker script: {WORKER_SCRIPT}")

    print("\n" + "=" * 70)
    print("RESOURCE REPLAY PROFILING COMPLETE.")
    print("=" * 70)


if __name__ == "__main__":
    main()
