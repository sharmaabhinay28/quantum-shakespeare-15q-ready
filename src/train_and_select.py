import os
import sys
import time
import json
import hashlib
import resource
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data import (
    download_gutenberg_100,
    prepare_corpus_and_splits,
    extract_chunks_and_counts
)
from src.evaluation import (
    probs_from_iqp,
    probs_from_bitflip,
    probs_from_autoregressive,
    probs_from_table,
    evaluate_log_loss
)
from src.guard import guard_output_dir
from src.models import (
    create_training_sampler,
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

def sha256_of_array(arr):
    return hashlib.sha256(arr.tobytes()).hexdigest()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train 180 candidate models and construct selection manifest.")
    parser.add_argument("--run-dir", required=True, help="Path to a NEW, non-existent directory where checkpoints and results will be created (e.g. runs/reproduction)")
    args = parser.parse_args()

    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(root_dir, "data")
    
    run_dir = guard_output_dir(args.run_dir, repo_root=root_dir, allow_existing=False)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    results_dir = os.path.join(run_dir, "results")
    
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    timing_records = {}
    
    # ---------------- Phase 1: Data Preparation ----------------
    print("\n=== Phase 1: Data Preparation ===")
    t_data_start = time.perf_counter()
    raw_path = os.path.join(data_dir, "gutenberg_100_raw.txt")
    if not os.path.exists(raw_path):
        raise FileNotFoundError("Frozen Gutenberg source is missing. Restore the archived source before retraining.")
    else:
        print("Raw Gutenberg file already present, verifying...")
        with open(raw_path, "rb") as f:
            data = f.read()
        sha = hashlib.sha256(data).hexdigest()
        print(f"Verified Gutenberg raw: {len(data)} bytes, SHA256: {sha}")
        
    print("Normalizing corpus and generating splits...")
    passages, splits, encoding_maps, manifest = prepare_corpus_and_splits(raw_path, output_dir=None, write_artifacts=False)
    print(f"Splits summary: Train={manifest['n_train_passages']}, Val={manifest['n_val_passages']}, Test={manifest['n_test_passages']}")
    
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
    
    # Warmup JIT
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
    
    t_fit_start = time.perf_counter()
    rss_fit_start = get_peak_rss_mb()
    
    total_candidates = len(models_dict) * len(encodings) * len(seeds) * 3
    cand_count = 0
    
    for model_name, model_obj in models_dict.items():
        print(f"\n--- Training candidates for Model: {model_name} ---")
        for enc_name in encodings:
            train_counts = encoded_data[enc_name]["train_counts"]
            val_counts = encoded_data[enc_name]["val_counts"]
            
            for seed in seeds:
                p_init = None
                if model_name != "Table":
                    p_init = model_obj.init_params(seed)
                
                sampler = create_training_sampler(train_counts, seed=seed)
                configs = candidate_lrs if model_name != "Table" else candidate_alphas
                
                for cfg_idx, cfg_val in enumerate(configs):
                    cand_count += 1
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
                        "checkpoint_path": os.path.relpath(ckpt_file, run_dir),
                        "checkpoint_sha256": ckpt_sha
                    }
                    candidate_records.append(rec)
                    print(f"[{cand_count}/{total_candidates}] {ckpt_id}: val_loss_char={v_loss_char:.4f} (took {t1-t0:.3f}s)")
                    
    t_fit_end = time.perf_counter()
    rss_fit_end = get_peak_rss_mb()
    timing_records["candidate_search_total_seconds"] = round(t_fit_end - t_fit_start, 4)
    timing_records["candidate_search_rss_peak_mb"] = round(rss_fit_end, 2)
    print(f"\nAll 180 candidate fits complete in {timing_records['candidate_search_total_seconds']}s")
    
    with open(os.path.join(results_dir, "candidate_validation_results.json"), "w") as f:
        json.dump(candidate_records, f, indent=2)
    pd.DataFrame(candidate_records).to_csv(os.path.join(results_dir, "candidate_validation_results.csv"), index=False)
    
    # ---------------- Phase 3: Selection Manifest & Freeze ----------------
    print("\n=== Phase 3: Selection Manifest Construction & Freeze ===")
    selected_models = []
    
    for model_name in models_dict.keys():
        for enc_name in encodings:
            for seed in seeds:
                matching = [
                    r for r in candidate_records 
                    if r["model"] == model_name and r["encoding"] == enc_name and r["seed"] == seed
                ]
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
    
    with open(os.path.join(results_dir, "partial_resource_accounting.json"), "w") as f:
        json.dump(timing_records, f, indent=2)
    print("\nTraining and validation selection phase completed successfully!")

if __name__ == "__main__":
    main()
