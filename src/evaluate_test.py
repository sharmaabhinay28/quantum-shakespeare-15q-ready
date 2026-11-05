import os
import sys
import time
import json
import argparse
import resource
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.guard import guard_output_dir
from src.resolver import resolve_checkpoint_path
from src.data import (
    ALPHABET,
    build_encoding_maps,
    extract_chunks_and_counts,
    prepare_corpus_and_splits
)
from src.evaluation import (
    probs_from_iqp,
    probs_from_bitflip,
    probs_from_autoregressive,
    probs_from_table,
    get_kernel_hat,
    exact_squared_mmd,
    evaluate_log_loss
)
from src.models import (
    IQPModel,
    StochasticBitflipModel,
    AutoregressiveBernoulliModel,
    FrequencyTableModel
)

def get_peak_rss_mb():
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return usage / (1024 * 1024)
    else:
        return usage / 1024

def main():
    parser = argparse.ArgumentParser(description="Evaluate held-out test metrics and compute bootstrap statistics")
    parser.add_argument("--selection-manifest", required=True, help="Path to selection_manifest.json to evaluate")
    parser.add_argument("--checkpoint-dir", required=True, help="Authoritative directory containing the checkpoints")
    parser.add_argument("--out-dir", required=True, help="Directory to write output files (must be a new, clean directory)")
    args = parser.parse_args()

    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(root_dir, "data")
    
    out_dir = guard_output_dir(args.out_dir, repo_root=root_dir, allow_existing=False)
    os.makedirs(out_dir, exist_ok=True)
    
    manifest_path = os.path.abspath(args.selection_manifest)
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"Selection manifest does not exist: {manifest_path}")
    results_dir = os.path.dirname(manifest_path)
        
    ckpt_dir = os.path.abspath(args.checkpoint_dir)
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"Checkpoint directory does not exist: {ckpt_dir}")
        
    with open(manifest_path, "r") as f:
        selection_manifest = json.load(f)
        
    selected_models = selection_manifest["selected_checkpoints"]
    print(f"Loaded selection manifest with {len(selected_models)} selected checkpoints.")
    
    # Load corpus and splits
    raw_path = os.path.join(data_dir, "gutenberg_100_raw.txt")
    passages, splits, encoding_maps, manifest = prepare_corpus_and_splits(raw_path, output_dir=None, write_artifacts=False)
    
    encodings = ["primary", "alt1", "alt2"]
    encoded_data = {}
    for enc_name in encodings:
        c2code = encoding_maps[enc_name]["char_to_code"]
        train_counts, train_passages = extract_chunks_and_counts(splits["train"], c2code)
        val_counts, val_passages = extract_chunks_and_counts(splits["val"], c2code)
        test_counts, test_passages = extract_chunks_and_counts(splits["test"], c2code)
        encoded_data[enc_name] = {
            "train_counts": train_counts,
            "test_counts": test_counts,
            "test_passages": test_passages
        }
        
    timing_records = {}
    if os.path.exists(os.path.join(results_dir, "partial_resource_accounting.json")):
        with open(os.path.join(results_dir, "partial_resource_accounting.json"), "r") as f:
            timing_records.update(json.load(f))
            
    # ---------------- Phase 4: Test Evaluation ----------------
    print("\n=== Phase 4: Held-Out Test Evaluation ===")
    t_test_start = time.perf_counter()
    k_hat = get_kernel_hat(sigmas=[1.0, 2.0, 4.0], n_qubits=15)
    
    models_dict = {
        "IQP": IQPModel(15),
        "Bitflip": StochasticBitflipModel(15),
        "Autoregressive": AutoregressiveBernoulliModel(15),
        "Table": FrequencyTableModel(15)
    }
    
    test_eval_results = []
    per_passage_records = []
    
    test_q_empirical = {}
    train_absent_masks = {}
    for enc_name in encodings:
        t_counts = encoded_data[enc_name]["test_counts"]
        test_q_empirical[enc_name] = t_counts / np.sum(t_counts)
        tr_counts = encoded_data[enc_name]["train_counts"]
        train_absent_masks[enc_name] = (tr_counts == 0)
        
    for idx, sel in enumerate(selected_models):
        model_name = sel["model"]
        enc_name = sel["encoding"]
        seed = sel["seed"]
        
        params, _ = resolve_checkpoint_path(sel, ckpt_dir=ckpt_dir)
        model_obj = models_dict[model_name]
        
        if model_name in ["IQP", "Bitflip", "Autoregressive"]:
            probs_raw = model_obj.get_probs(params)
        else:
            probs_raw = params
            
        assert np.all(probs_raw >= -1e-12), f"Negative probability in {model_name}"
        probs_raw = np.clip(probs_raw, 0.0, None)
        total_p = np.sum(probs_raw)
        assert abs(total_p - 1.0) < 1e-9, f"Mass not normalized: {total_p}"
        probs_raw /= total_p
        
        # Invalid mass is identically 0
        invalid_mass = 0.0
        # Unseen valid mass
        unseen_mass = float(np.sum(probs_raw[train_absent_masks[enc_name]]))
        # Held-out squared MMD
        test_mmd2 = exact_squared_mmd(probs_raw, test_q_empirical[enc_name], k_hat=k_hat, n_qubits=15)
        
        # Test log loss
        test_counts = encoded_data[enc_name]["test_counts"]
        test_loss_chunk, test_loss_char, test_unsmoothed = evaluate_log_loss(
            probs_raw, test_counts, smoothing_eps=1e-6
        )
        
        # Smoothing sensitivity
        smoothing_sensitivities = {}
        for eps_val in [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]:
            _, loss_ch, _ = evaluate_log_loss(probs_raw, test_counts, smoothing_eps=eps_val)
            smoothing_sensitivities[f"eps_{eps_val}"] = loss_ch
            
        # Per-passage log loss
        q_eval = (1.0 - 1e-6) * probs_raw + (1e-6 / 32768)
        log_q_eval = np.log(q_eval)
        
        per_passage_losses = []
        for p_info in encoded_data[enc_name]["test_passages"]:
            p_counts = p_info["counts"]
            p_active = p_counts > 0
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
        print(f"[{idx+1}/{len(selected_models)}] Evaluated {model_name} {enc_name} seed {seed}: test_loss_char={test_loss_char:.4f}, mmd2={test_mmd2:.6f}")
        
    t_test_end = time.perf_counter()
    timing_records["test_evaluation_seconds"] = round(t_test_end - t_test_start, 4)
    print(f"Held-out test evaluation complete in {timing_records['test_evaluation_seconds']}s")
    
    # Save per-passage results
    pd.DataFrame(per_passage_records).to_csv(
        os.path.join(out_dir, "test_per_passage_results.csv"), index=False
    )
    
    summary_list = []
    for r in test_eval_results:
        c = r.copy()
        del c["passage_losses"]
        summary_list.append(c)
    with open(os.path.join(out_dir, "test_results_summary.json"), "w") as f:
        json.dump(summary_list, f, indent=2)
    pd.DataFrame(summary_list).to_csv(os.path.join(out_dir, "test_results_summary.csv"), index=False)
    
    # ---------------- Phase 5: Paired Bootstrap Analysis ----------------
    print("\n=== Phase 5: Paired Bootstrap Uncertainty Estimation (2,000 resamples) ===")
    t_boot_start = time.perf_counter()
    n_boot = 2000
    boot_rng = np.random.Generator(np.random.PCG64(314159))
    
    n_test_passages = len(splits["test"])
    passage_indices = np.arange(n_test_passages)
    boot_resamples = boot_rng.choice(passage_indices, size=(n_boot, n_test_passages), replace=True)
    
    bootstrap_results = {}
    for enc_name in encodings:
        print(f"Bootstrapping encoding: {enc_name}...")
        test_p_chunks = np.array([p["chunk_count"] for p in encoded_data[enc_name]["test_passages"]], dtype=np.float64)
        total_tokens_boot = np.sum(test_p_chunks[boot_resamples], axis=1)
        
        model_avg_loss_sums = {}
        seed_scores = {}
        for model_name in models_dict.keys():
            matching_results = [
                r for r in test_eval_results if r["model"] == model_name and r["encoding"] == enc_name
            ]
            seed_matrix = np.array([
                [p["loss_sum"] for p in r["passage_losses"]]
                for r in matching_results
            ])
            avg_loss_sum_per_passage = np.mean(seed_matrix, axis=0)
            model_avg_loss_sums[model_name] = avg_loss_sum_per_passage
            
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
            
        model_boot_losses = {}
        model_intervals = {}
        for model_name, avg_sums in model_avg_loss_sums.items():
            resampled_sums = np.sum(avg_sums[boot_resamples], axis=1)
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
            
        iqp_boot = model_boot_losses["IQP"]
        paired_diffs = {}
        for base_name in ["Bitflip", "Autoregressive", "Table"]:
            diff_boot = iqp_boot - model_boot_losses[base_name]
            p2_5 = float(np.percentile(diff_boot, 2.5))
            p97_5 = float(np.percentile(diff_boot, 97.5))
            diff_mean = float(np.mean(diff_boot))
            
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
    
    with open(os.path.join(out_dir, "bootstrap_results.json"), "w") as f:
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
        
        params, _ = resolve_checkpoint_path(sel, ckpt_dir=ckpt_dir)
        model_obj = models_dict[model_name]
        if model_name in ["IQP", "Bitflip", "Autoregressive"]:
            probs_raw = model_obj.get_probs(params)
        else:
            probs_raw = params
            
        probs_raw = np.clip(probs_raw, 0.0, None)
        probs_raw /= np.sum(probs_raw)
        
        rng_sample = np.random.Generator(np.random.PCG64(sample_seed))
        sample_indices = rng_sample.choice(32768, size=10000, replace=True, p=probs_raw)
        
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
    
    with open(os.path.join(out_dir, "samples_10k_summary.json"), "w") as f:
        json.dump(samples_summary, f, indent=2)
        
    with open(os.path.join(out_dir, "resource_accounting.json"), "w") as f:
        json.dump(timing_records, f, indent=2)
        
    # ---------------- Phase 7: Standalone Plots Generation ----------------
    print("\n=== Phase 7: Generating Standalone PNG Plots ===")
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
    plot1_path = os.path.join(out_dir, "plot_test_log_loss.png")
    plt.savefig(plot1_path, dpi=300)
    plt.close()
    print(f"Saved Plot 1: {plot1_path}")
    
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
    plot2_path = os.path.join(out_dir, "plot_paired_differences.png")
    plt.savefig(plot2_path, dpi=300)
    plt.close()
    print(f"Saved Plot 2: {plot2_path}")
    
    # Plot 3: Resource Accounting
    cand_csv = os.path.join(os.path.dirname(manifest_path), "candidate_validation_results.csv")
    if os.path.exists(cand_csv):
        df_cand = pd.read_csv(cand_csv)
        fig, ax = plt.subplots(figsize=(8, 5))
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
        plot3_path = os.path.join(out_dir, "plot_resource_costs.png")
        plt.savefig(plot3_path, dpi=300)
        plt.close()
        print(f"Saved Plot 3: {plot3_path}")
        
    print("\nTest evaluation and analysis complete!")

if __name__ == "__main__":
    main()
