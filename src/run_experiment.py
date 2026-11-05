import os
import sys
import time
import math
import json
import hashlib
import resource
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data import (
    ALPHABET,
    download_gutenberg_100,
    build_encoding_maps,
    prepare_corpus_and_splits,
    extract_chunks_and_counts,
    encode_chunk
)
from src.evaluation import (
    fwht,
    get_standard_gates,
    probs_from_iqp,
    probs_from_bitflip,
    probs_from_autoregressive,
    probs_from_table,
    get_kernel_hat,
    exact_squared_mmd,
    evaluate_log_loss
)
from src.resolver import resolve_checkpoint_path
from src.models import (
    create_training_sampler,
    IQPModel,
    StochasticBitflipModel,
    AutoregressiveBernoulliModel,
    FrequencyTableModel
)

def get_peak_rss_mb():
    """Returns peak RSS in MB (macOS ru_maxrss is in bytes)."""
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # On macOS, ru_maxrss is bytes; on Linux it is kilobytes.
    if sys.platform == "darwin":
        return usage / (1024 * 1024)
    else:
        return usage / 1024

def sha256_of_array(arr):
    return hashlib.sha256(arr.tobytes()).hexdigest()

def main():
    raise RuntimeError(
        "Legacy monolithic run_experiment.py is disabled to protect frozen publication artifacts "
        "from accidental overwrites.\n\n"
        "Please use the supported two-stage reproduction recipe:\n"
        "  1. Candidate Training & Validation Selection:\n"
        "     python src/train_and_select.py --run-dir runs/reproduction\n\n"
        "  2. Held-Out Test Evaluation & Plotting:\n"
        "     python src/evaluate_test.py \n"
        "       --selection-manifest runs/reproduction/results/selection_manifest.json \n"
        "       --checkpoint-dir runs/reproduction/checkpoints \n"
        "       --out-dir runs/reproduction/evaluation\n"
    )
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(root_dir, "data")
    ckpt_dir = os.path.join(root_dir, "checkpoints")
    results_dir = os.path.join(root_dir, "results")
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    timing_records = {}
    
    # ---------------- Phase 1: Data Preparation ----------------
    print("\n=== Phase 1: Data Preparation ===")
    t_data_start = time.perf_counter()
    raw_path = os.path.join(data_dir, "gutenberg_100_raw.txt")
    if not os.path.exists(raw_path):
        print("Downloading Gutenberg 100...")
        raw_path, meta = download_gutenberg_100(data_dir)
        print(f"Downloaded Gutenberg 100, size: {meta['byte_count']} bytes, SHA256: {meta['sha256']}")
    else:
        print("Raw Gutenberg file already present, verifying...")
        with open(raw_path, "rb") as f:
            data = f.read()
        sha = hashlib.sha256(data).hexdigest()
        print(f"Verified Gutenberg raw: {len(data)} bytes, SHA256: {sha}")
        
    print("Normalizing corpus and generating splits...")
    passages, splits, encoding_maps, manifest = prepare_corpus_and_splits(raw_path, data_dir)
    print(f"Splits summary: Train={manifest['n_train_passages']}, Val={manifest['n_val_passages']}, Test={manifest['n_test_passages']}")
    
    # Pre-extract chunks for each encoding
    encodings = ["primary", "alt1", "alt2"]
    encoded_data = {}
    for enc_name in encodings:
        print(f"Extracting chunks for encoding: {enc_name}...")
        c2code = encoding_maps[enc_name]["char_to_code"]
        train_counts, train_passages_chunks = extract_chunks_and_counts(splits["train"], c2code)
        val_counts, val_passages_chunks = extract_chunks_and_counts(splits["val"], c2code)
        test_counts, test_passages_chunks = extract_chunks_and_counts(splits["test"], c2code)
        
        encoded_data[enc_name] = {
            "train_counts": train_counts,
            "val_counts": val_counts,
            "test_counts": test_counts,
            "train_passages": train_passages_chunks,
            "val_passages": val_passages_chunks,
            "test_passages": test_passages_chunks
        }
        
    t_data_end = time.perf_counter()
    timing_records["data_preparation_seconds"] = round(t_data_end - t_data_start, 4)
    print(f"Data preparation complete in {timing_records['data_preparation_seconds']}s")
    
    # ---------------- Phase 2: Candidate Fitting & Validation ----------------
    print("\n=== Phase 2: Candidate Training & Validation Selection ===")
    seeds = [11, 23, 37, 53, 71]
    candidate_lrs = [0.01, 0.03, 0.1]
    candidate_alphas = [0.01, 0.1, 1.0]
    
    models_dict = {
        "IQP": IQPModel(15),
        "Bitflip": StochasticBitflipModel(15),
        "Autoregressive": AutoregressiveBernoulliModel(15),
        "Table": FrequencyTableModel(15)
    }
    
    # Warmup JIT to account for JIT compile time separately
    print("Warming up JIT compilation...")
    t_jit_start = time.perf_counter()
    dummy_sampler = create_training_sampler(encoded_data["primary"]["train_counts"], seed=999)
    p_init_iqp = models_dict["IQP"].init_params(999)
    _, _ = models_dict["IQP"].train(p_init_iqp, dummy_sampler, lr=0.01, n_updates=2, seed=999)
    p_init_bf = models_dict["Bitflip"].init_params(999)
    _, _ = models_dict["Bitflip"].train(p_init_bf, dummy_sampler, lr=0.01, n_updates=2, seed=999)
    p_init_ar = models_dict["Autoregressive"].init_params(999)
    _, _ = models_dict["Autoregressive"].train(p_init_ar, dummy_sampler, lr=0.01, n_updates=2, seed=999)
    t_jit_end = time.perf_counter()
    timing_records["jit_compilation_overhead_seconds"] = round(t_jit_end - t_jit_start, 4)
    print(f"JIT warmup finished in {timing_records['jit_compilation_overhead_seconds']}s")
    
    candidate_records = []
    checkpoint_records = {}
    
    t_fit_start = time.perf_counter()
    rss_fit_start = get_peak_rss_mb()
    
    for model_name, model_obj in models_dict.items():
        print(f"\n--- Training candidates for Model: {model_name} ---")
        for enc_name in encodings:
            train_counts = encoded_data[enc_name]["train_counts"]
            val_counts = encoded_data[enc_name]["val_counts"]
            
            for seed in seeds:
                # Pre-initialize parameters using declared seed
                p_init = None
                if model_name != "Table":
                    p_init = model_obj.init_params(seed)
                
                sampler = create_training_sampler(train_counts, seed=seed)
                configs = candidate_lrs if model_name != "Table" else candidate_alphas
                
                for cfg_idx, cfg_val in enumerate(configs):
                    t0 = time.perf_counter()
                    if model_name in ["IQP", "Bitflip", "Autoregressive"]:
                        trained_params, loss_history = model_obj.train(
                            p_init, sampler, lr=cfg_val, n_updates=500, seed=seed + 1000 * (cfg_idx + 1)
                        )
                        probs_raw = model_obj.get_probs(trained_params)
                    else: # Table
                        probs_raw = model_obj.fit(train_counts, alpha=cfg_val)
                        trained_params = probs_raw
                        loss_history = []
                    t1 = time.perf_counter()
                    
                    # Validation evaluation (exact validation log loss with 1e-6 smoothing)
                    v_loss_chunk, v_loss_char, v_unsmoothed = evaluate_log_loss(probs_raw, val_counts, smoothing_eps=1e-6)
                    
                    ckpt_id = f"{model_name}_{enc_name}_s{seed}_cfg{cfg_idx}"
                    ckpt_file = os.path.join(ckpt_dir, f"{ckpt_id}.npy")
                    np.save(ckpt_file, trained_params)
                    ckpt_sha = sha256_of_array(trained_params)
                    
                    rec = {
                        "model": model_name,
                        "encoding": enc_name,
                        "seed": seed,
                        "config_index": cfg_idx,
                        "config_value": cfg_val,
                        "validation_log_loss_chunk": v_loss_chunk,
                        "validation_log_loss_char": v_loss_char,
                        "validation_unsmoothed": v_unsmoothed,
                        "train_wall_seconds": round(t1 - t0, 4),
                        "checkpoint_path": ckpt_file,
                        "checkpoint_sha256": ckpt_sha
                    }
                    candidate_records.append(rec)
                    checkpoint_records[ckpt_id] = {
                        "model": model_name,
                        "encoding": enc_name,
                        "seed": seed,
                        "config_value": cfg_val,
                        "checkpoint_sha256": ckpt_sha,
                        "validation_log_loss_char": v_loss_char,
                        "params_file": ckpt_file
                    }
                    
    t_fit_end = time.perf_counter()
    rss_fit_end = get_peak_rss_mb()
    timing_records["candidate_search_total_seconds"] = round(t_fit_end - t_fit_start, 4)
    timing_records["candidate_search_rss_peak_mb"] = round(rss_fit_end, 2)
    print(f"\nAll 180 candidate fits complete in {timing_records['candidate_search_total_seconds']}s")
    
    # Save candidate validation results
    with open(os.path.join(results_dir, "candidate_validation_results.json"), "w") as f:
        json.dump(candidate_records, f, indent=2)
    pd.DataFrame(candidate_records).to_csv(os.path.join(results_dir, "candidate_validation_results.csv"), index=False)
    
    # ---------------- Phase 3: Selection Manifest & Freeze ----------------
    print("\n=== Phase 3: Selection Manifest Construction & Freeze ===")
    selected_models = []
    
    # Selection rule: lowest validation log loss, ties within 1e-12 resolved by candidate order
    for model_name in models_dict.keys():
        for enc_name in encodings:
            for seed in seeds:
                matching = [
                    r for r in candidate_records 
                    if r["model"] == model_name and r["encoding"] == enc_name and r["seed"] == seed
                ]
                # Sort by config_index first to ensure deterministic tie-breaking
                matching.sort(key=lambda x: x["config_index"])
                best_rec = matching[0]
                for r in matching[1:]:
                    if r["validation_log_loss_chunk"] < best_rec["validation_log_loss_chunk"] - 1e-12:
                        best_rec = r
                        
                selected_models.append({
                    "model": model_name,
                    "encoding": enc_name,
                    "seed": seed,
                    "selected_config_index": best_rec["config_index"],
                    "selected_config_value": best_rec["config_value"],
                    "validation_log_loss_chunk": best_rec["validation_log_loss_chunk"],
                    "validation_log_loss_char": best_rec["validation_log_loss_char"],
                    "checkpoint_path": best_rec["checkpoint_path"],
                    "checkpoint_sha256": best_rec["checkpoint_sha256"]
                })
                
    selection_manifest = {
        "manifest_timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_candidates": len(candidate_records),
        "total_selected": len(selected_models),
        "frozen_protocol_commit": "d1a3143500768c944fc5845f3153183a4a2d291e",
        "frozen_protocol_sha256": "64086fe2e51c747d39aaba3fe17856583e05f93f6428ea20fb160c305b473dac",
        "selected_checkpoints": selected_models
    }
    
    manifest_path = os.path.join(results_dir, "selection_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(selection_manifest, f, indent=2)
    print(f"Selection manifest committed with {len(selected_models)} selected models to {manifest_path}")
    
    # ---------------- Phase 4: Test Evaluation ----------------
    print("\n=== Phase 4: Held-Out Test Evaluation ===")
    t_test_start = time.perf_counter()
    k_hat = get_kernel_hat(sigmas=[1.0, 2.0, 4.0], n_qubits=15)
    
    test_eval_results = []
    per_passage_records = []
    
    # Precompute empirical test distribution for each encoding
    test_q_empirical = {}
    train_absent_masks = {}
    for enc_name in encodings:
        t_counts = encoded_data[enc_name]["test_counts"]
        test_q_empirical[enc_name] = t_counts / np.sum(t_counts)
        tr_counts = encoded_data[enc_name]["train_counts"]
        train_absent_masks[enc_name] = (tr_counts == 0)
        
    for sel in selected_models:
        model_name = sel["model"]
        enc_name = sel["encoding"]
        seed = sel["seed"]
        
        # Load parameters
        params, _ = resolve_checkpoint_path(sel, ckpt_dir=ckpt_dir)
        model_obj = models_dict[model_name]
        
        if model_name in ["IQP", "Bitflip", "Autoregressive"]:
            probs_raw = model_obj.get_probs(params)
        else:
            probs_raw = params # Table
            
        # Verify probability distribution integrity
        assert np.all(probs_raw >= -1e-12), f"Negative probability in {model_name}"
        probs_raw = np.clip(probs_raw, 0.0, None)
        total_p = np.sum(probs_raw)
        assert abs(total_p - 1.0) < 1e-9, f"Mass not normalized: {total_p}"
        probs_raw /= total_p
        
        # Check invalid mass: structural invalid mass is identically 0 for 32-char alphabet
        invalid_mass = 0.0 # 32^3 = 32768, all 15-bit states are valid
        
        # Unseen valid chunks mass (chunks absent from training)
        unseen_mass = float(np.sum(probs_raw[train_absent_masks[enc_name]]))
        
        # Held-out squared MMD via FWHT
        test_mmd2 = exact_squared_mmd(probs_raw, test_q_empirical[enc_name], k_hat=k_hat, n_qubits=15)
        
        # Overall held-out log loss
        test_counts = encoded_data[enc_name]["test_counts"]
        test_loss_chunk, test_loss_char, test_unsmoothed = evaluate_log_loss(
            probs_raw, test_counts, smoothing_eps=1e-6
        )
        
        # Smoothing sensitivity
        smoothing_sensitivities = {}
        for eps_val in [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]:
            loss_c, loss_ch, _ = evaluate_log_loss(probs_raw, test_counts, smoothing_eps=eps_val)
            smoothing_sensitivities[f"eps_{eps_val}"] = loss_ch
            
        # Per-passage log loss evaluation
        q_eval = (1.0 - 1e-6) * probs_raw + (1e-6 / 32768)
        log_q_eval = np.log(q_eval)
        
        per_passage_losses = []
        for p_info in encoded_data[enc_name]["test_passages"]:
            p_counts = p_info["counts"]
            p_active = p_counts > 0
            # Passage log loss sum: sum_{x in passage} -log q_eval(x)
            p_loss_sum = -np.sum(p_counts[p_active] * log_q_eval[p_active])
            p_chunk_cnt = p_info["chunk_count"]
            per_passage_losses.append({
                "passage_id": p_info["passage_id"],
                "chunk_count": p_chunk_cnt,
                "loss_sum": p_loss_sum,
                "loss_per_chunk": p_loss_sum / p_chunk_cnt,
                "loss_per_char": (p_loss_sum / p_chunk_cnt) / 3.0
            })
            per_passage_records.append({
                "model": model_name,
                "encoding": enc_name,
                "seed": seed,
                "passage_id": p_info["passage_id"],
                "chunk_count": p_chunk_cnt,
                "loss_sum": p_loss_sum,
                "loss_per_char": (p_loss_sum / p_chunk_cnt) / 3.0
            })
            
        test_eval_results.append({
            "model": model_name,
            "encoding": enc_name,
            "seed": seed,
            "selected_config_value": sel["selected_config_value"],
            "test_log_loss_chunk": test_loss_chunk,
            "test_log_loss_char": test_loss_char,
            "test_unsmoothed_chunk": test_unsmoothed,
            "test_squared_mmd": test_mmd2,
            "invalid_mass": invalid_mass,
            "unseen_valid_mass": unseen_mass,
            "smoothing_sensitivity_char": smoothing_sensitivities,
            "passage_losses": per_passage_losses
        })
        
    t_test_end = time.perf_counter()
    timing_records["test_evaluation_seconds"] = round(t_test_end - t_test_start, 4)
    print(f"Held-out test evaluation complete in {timing_records['test_evaluation_seconds']}s")
    
    # Save per-passage results
    pd.DataFrame(per_passage_records).to_csv(
        os.path.join(results_dir, "test_per_passage_results.csv"), index=False
    )
    
    # Save test summary without the full per-passage lists to keep summary lightweight
    summary_list = []
    for r in test_eval_results:
        c = r.copy()
        del c["passage_losses"]
        summary_list.append(c)
    with open(os.path.join(results_dir, "test_results_summary.json"), "w") as f:
        json.dump(summary_list, f, indent=2)
    pd.DataFrame(summary_list).to_csv(os.path.join(results_dir, "test_results_summary.csv"), index=False)
    
    # ---------------- Phase 5: Paired Bootstrap Analysis ----------------
    print("\n=== Phase 5: Paired Bootstrap Uncertainty Estimation (2,000 resamples) ===")
    t_boot_start = time.perf_counter()
    n_boot = 2000
    boot_rng = np.random.Generator(np.random.PCG64(314159))
    
    # Number of test passages
    n_test_passages = len(splits["test"])
    passage_indices = np.arange(n_test_passages)
    
    # Pre-draw 2,000 resampled passage index sets
    boot_resamples = boot_rng.choice(passage_indices, size=(n_boot, n_test_passages), replace=True)
    
    bootstrap_results = {}
    for enc_name in encodings:
        print(f"Bootstrapping encoding: {enc_name}...")
        # Get passage lengths (chunk counts) for test passages
        test_p_chunks = np.array([p["chunk_count"] for p in encoded_data[enc_name]["test_passages"]], dtype=np.float64)
        total_tokens_boot = np.sum(test_p_chunks[boot_resamples], axis=1) # shape (2000,)
        
        # For each model, compute the average per-passage loss sum across the 5 seeds:
        model_avg_loss_sums = {}
        seed_scores = {}
        for model_name in models_dict.keys():
            matching_results = [
                r for r in test_eval_results if r["model"] == model_name and r["encoding"] == enc_name
            ]
            # Matrix of shape (5, n_test_passages) containing loss_sum for each seed
            seed_matrix = np.array([
                [p["loss_sum"] for p in r["passage_losses"]]
                for r in matching_results
            ])
            # Average across 5 seeds: shape (n_test_passages,)
            avg_loss_sum_per_passage = np.mean(seed_matrix, axis=0)
            model_avg_loss_sums[model_name] = avg_loss_sum_per_passage
            
            # Individual seed overall token-weighted losses
            total_tokens = np.sum(test_p_chunks)
            seed_losses_char = [
                (np.sum(seed_matrix[s]) / total_tokens) / 3.0 for s in range(len(matching_results))
            ]
            seed_scores[model_name] = {
                "seeds": [r["seed"] for r in matching_results],
                "losses_char": seed_losses_char,
                "mean_char": float(np.mean(seed_losses_char)),
                "std_char": float(np.std(seed_losses_char, ddof=1))
            }
            
        # Bootstrap distribution for each model
        model_boot_losses = {}
        model_intervals = {}
        for model_name, avg_sums in model_avg_loss_sums.items():
            # Resampled sums: shape (2000,)
            resampled_sums = np.sum(avg_sums[boot_resamples], axis=1)
            # Token-weighted nats per char
            boot_char = (resampled_sums / total_tokens_boot) / 3.0
            model_boot_losses[model_name] = boot_char
            p2_5 = float(np.percentile(boot_char, 2.5))
            p97_5 = float(np.percentile(boot_char, 97.5))
            mean_val = float(np.mean(boot_char))
            model_intervals[model_name] = {
                "mean": mean_val,
                "ci_lower_95": p2_5,
                "ci_upper_95": p97_5
            }
            
        # Paired differences: IQP minus Baseline
        iqp_boot = model_boot_losses["IQP"]
        paired_diffs = {}
        for base_name in ["Bitflip", "Autoregressive", "Table"]:
            diff_boot = iqp_boot - model_boot_losses[base_name]
            p2_5 = float(np.percentile(diff_boot, 2.5))
            p97_5 = float(np.percentile(diff_boot, 97.5))
            diff_mean = float(np.mean(diff_boot))
            
            # Verdict based on strict protocol definition:
            # strictly below zero: detected win for IQP
            # strictly above zero: detected loss for IQP
            # straddling zero: no reliable difference
            if p97_5 < 0.0:
                verdict = "IQP_WIN"
            elif p2_5 > 0.0:
                verdict = "IQP_LOSS"
            else:
                verdict = "NO_RELIABLE_DIFFERENCE"
                
            paired_diffs[base_name] = {
                "mean_diff": diff_mean,
                "ci_lower_95": p2_5,
                "ci_upper_95": p97_5,
                "verdict": verdict
            }
            
        bootstrap_results[enc_name] = {
            "model_intervals": model_intervals,
            "paired_differences": paired_diffs,
            "seed_variation": seed_scores
        }
        
    t_boot_end = time.perf_counter()
    timing_records["bootstrap_resampling_seconds"] = round(t_boot_end - t_boot_start, 4)
    print(f"Bootstrap uncertainty estimation complete in {timing_records['bootstrap_resampling_seconds']}s")
    
    with open(os.path.join(results_dir, "bootstrap_results.json"), "w") as f:
        json.dump(bootstrap_results, f, indent=2)
        
    # ---------------- Phase 6: Sampling & Resource Accounting ----------------
    print("\n=== Phase 6: 10,000 Chunk Sampling & Resource Accounting ===")
    t_sample_start = time.perf_counter()
    samples_summary = []
    
    for sel in selected_models:
        model_name = sel["model"]
        enc_name = sel["encoding"]
        seed = sel["seed"]
        sample_seed = seed + 100000
        
        # Load parameters
        params, _ = resolve_checkpoint_path(sel, ckpt_dir=ckpt_dir)
        model_obj = models_dict[model_name]
        if model_name in ["IQP", "Bitflip", "Autoregressive"]:
            probs_raw = model_obj.get_probs(params)
        else:
            probs_raw = params
            
        probs_raw = np.clip(probs_raw, 0.0, None)
        probs_raw /= np.sum(probs_raw)
        
        # Sample 10,000 independent chunks
        rng_sample = np.random.Generator(np.random.PCG64(sample_seed))
        sample_indices = rng_sample.choice(32768, size=10000, replace=True, p=probs_raw)
        
        # Check fraction of absent chunks in generated samples
        absent_mask = encoded_data[enc_name]["train_counts"] == 0
        gen_absent_fraction = float(np.mean(absent_mask[sample_indices]))
        
        samples_summary.append({
            "model": model_name,
            "encoding": enc_name,
            "seed": seed,
            "sample_seed": sample_seed,
            "n_samples": 10000,
            "unique_states_sampled": int(len(np.unique(sample_indices))),
            "gen_unseen_training_chunk_fraction": gen_absent_fraction
        })
        
    t_sample_end = time.perf_counter()
    timing_records["sampling_10k_seconds"] = round(t_sample_end - t_sample_start, 4)
    timing_records["total_cumulative_rss_peak_mb"] = round(get_peak_rss_mb(), 2)
    print(f"Sampling complete in {timing_records['sampling_10k_seconds']}s")
    
    with open(os.path.join(results_dir, "samples_10k_summary.json"), "w") as f:
        json.dump(samples_summary, f, indent=2)
        
    with open(os.path.join(results_dir, "resource_accounting.json"), "w") as f:
        json.dump(timing_records, f, indent=2)
        
    # ---------------- Phase 7: Standalone Plots Generation ----------------
    print("\n=== Phase 7: Generating Standalone PNG Plots ===")
    
    # Plot 1: Test Log Loss with 95% Confidence Intervals
    fig, ax = plt.subplots(figsize=(10, 6))
    enc_colors = {"primary": "#1f77b4", "alt1": "#ff7f0e", "alt2": "#2ca02c"}
    x_positions = np.arange(len(models_dict))
    width = 0.25
    
    for i, enc_name in enumerate(encodings):
        means = [bootstrap_results[enc_name]["model_intervals"][m]["mean"] for m in models_dict.keys()]
        err_low = [means[j] - bootstrap_results[enc_name]["model_intervals"][m]["ci_lower_95"] for j, m in enumerate(models_dict.keys())]
        err_high = [bootstrap_results[enc_name]["model_intervals"][m]["ci_upper_95"] - means[j] for j, m in enumerate(models_dict.keys())]
        
        ax.bar(
            x_positions + (i - 1) * width, means, width=width,
            yerr=[err_low, err_high], capsize=4, label=f"Encoding: {enc_name}",
            color=enc_colors[enc_name], alpha=0.85, edgecolor="black"
        )
        
    ax.set_ylabel("Held-Out Test Log Loss (nats / char)", fontsize=12)
    ax.set_title("Test Log Loss by Model and Encoding (95% Passage Bootstrap CI)", fontsize=14, fontweight="bold")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(["IQP (Quantum)", "Stochastic Bitflip", "Autoregressive Bernoulli", "Frequency Table"], fontsize=11)
    ax.legend(frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plot1_path = os.path.join(results_dir, "plot_test_log_loss.png")
    plt.savefig(plot1_path, dpi=300)
    plt.close()
    print(f"Saved Plot 1: {plot1_path}")
    
    # Plot 2: Per-Encoding Paired IQP Differences (IQP - Baseline)
    fig, ax = plt.subplots(figsize=(10, 6))
    base_models = ["Bitflip", "Autoregressive", "Table"]
    x_positions = np.arange(len(base_models))
    
    for i, enc_name in enumerate(encodings):
        diff_means = [bootstrap_results[enc_name]["paired_differences"][b]["mean_diff"] for b in base_models]
        err_low = [diff_means[j] - bootstrap_results[enc_name]["paired_differences"][b]["ci_lower_95"] for j, b in enumerate(base_models)]
        err_high = [bootstrap_results[enc_name]["paired_differences"][b]["ci_upper_95"] - diff_means[j] for j, b in enumerate(base_models)]
        
        ax.bar(
            x_positions + (i - 1) * width, diff_means, width=width,
            yerr=[err_low, err_high], capsize=4, label=f"Encoding: {enc_name}",
            color=enc_colors[enc_name], alpha=0.85, edgecolor="black"
        )
        
    ax.axhline(0, color="red", linestyle="--", linewidth=1.5, label="Zero Difference (Parity)")
    ax.set_ylabel("Paired Difference: IQP − Baseline (nats / char)", fontsize=12)
    ax.set_title("Paired Difference: IQP vs. Baselines (95% CI, < 0 is IQP Win)", fontsize=14, fontweight="bold")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(["vs. Stochastic Bitflip", "vs. Autoregressive Bernoulli", "vs. Frequency Table"], fontsize=11)
    ax.legend(frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plot2_path = os.path.join(results_dir, "plot_paired_differences.png")
    plt.savefig(plot2_path, dpi=300)
    plt.close()
    print(f"Saved Plot 2: {plot2_path}")
    
    # Plot 3: Resource Accounting (Wall Time by Model)
    fig, ax = plt.subplots(figsize=(8, 5))
    df_cand = pd.DataFrame(candidate_records)
    mean_times = df_cand.groupby("model")["train_wall_seconds"].mean()
    models_order = ["IQP", "Bitflip", "Autoregressive", "Table"]
    times = [mean_times.get(m, 0.0) for m in models_order]
    
    bars = ax.bar(["IQP", "Stochastic Bitflip", "Autoregressive Bernoulli", "Frequency Table"], times, color="#4c72b0", edgecolor="black")
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.005, f"{h:.3f}s", ha="center", va="bottom", fontsize=10)
        
    ax.set_ylabel("Mean Training Time per Fit (seconds)", fontsize=12)
    ax.set_title("Computational Cost per Candidate Fit (500 updates)", fontsize=14, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plot3_path = os.path.join(results_dir, "plot_resource_costs.png")
    plt.savefig(plot3_path, dpi=300)
    plt.close()
    print(f"Saved Plot 3: {plot3_path}")
    
    print("\nExperiment pipeline executed successfully!")

if __name__ == "__main__":
    main()
