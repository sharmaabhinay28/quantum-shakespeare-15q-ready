import os
import re
import math
import json
import time
import hashlib
import unicodedata
import urllib.request
import numpy as np

ALPHABET = "abcdefghijklmnopqrstuvwxyz .,;?!"

def download_gutenberg_100(target_dir):
    """Download Project Gutenberg ebook 100 with content-length and SHA-256 integrity guards."""
    os.makedirs(target_dir, exist_ok=True)
    raw_path = os.path.join(target_dir, "gutenberg_100_raw.txt")
    meta_path = os.path.join(target_dir, "gutenberg_source_metadata.json")
    
    url = "https://www.gutenberg.org/ebooks/100.txt.utf-8"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"})
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=60) as resp:
                resolved_url = resp.geturl()
                headers = dict(resp.getheaders())
                data = resp.read()
            t1 = time.time()
            
            # Check content length if provided
            content_length = headers.get("content-length") or headers.get("Content-Length")
            if content_length is not None:
                expected_len = int(content_length)
                if len(data) != expected_len:
                    raise IOError(f"Incomplete transfer: received {len(data)} bytes, expected {expected_len}")
            
            sha256 = hashlib.sha256(data).hexdigest()
            
            with open(raw_path, "wb") as f:
                f.write(data)
                
            metadata = {
                "source_url": url,
                "resolved_url": resolved_url,
                "retrieval_timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "retrieval_duration_seconds": round(t1 - t0, 3),
                "byte_count": len(data),
                "sha256": sha256,
                "http_headers": {k: str(v) for k, v in headers.items()}
            }
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)
                
            return raw_path, metadata
        except Exception as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"Failed to download after {max_retries} attempts: {e}")
            time.sleep(2)

def build_encoding_maps():
    """Build primary and 2 alternative encodings:
    Primary: alphabet indices 0..31.
    Alt1: PCG64 seed 1729.
    Alt2: PCG64 seed 2718.
    """
    assert len(ALPHABET) == 32
    # Primary
    char_to_code_primary = {ch: i for i, ch in enumerate(ALPHABET)}
    code_to_char_primary = {i: ch for i, ch in enumerate(ALPHABET)}
    
    # Alt 1: PCG64 seed 1729
    rng1 = np.random.Generator(np.random.PCG64(1729))
    perm1 = rng1.permutation(32).tolist()
    char_to_code_alt1 = {ch: int(perm1[i]) for i, ch in enumerate(ALPHABET)}
    code_to_char_alt1 = {int(perm1[i]): ch for i, ch in enumerate(ALPHABET)}
    
    # Alt 2: PCG64 seed 2718
    rng2 = np.random.Generator(np.random.PCG64(2718))
    perm2 = rng2.permutation(32).tolist()
    char_to_code_alt2 = {ch: int(perm2[i]) for i, ch in enumerate(ALPHABET)}
    code_to_char_alt2 = {int(perm2[i]): ch for i, ch in enumerate(ALPHABET)}
    
    maps = {
        "primary": {"char_to_code": char_to_code_primary, "code_to_char": code_to_char_primary},
        "alt1": {"char_to_code": char_to_code_alt1, "code_to_char": code_to_char_alt1, "seed": 1729},
        "alt2": {"char_to_code": char_to_code_alt2, "code_to_char": code_to_char_alt2, "seed": 2718}
    }
    return maps

def encode_chunk(c0, c1, c2, char_to_code):
    """Encodes a 3-char chunk into a 15-bit integer state:
    c0 is bits 14..10, c1 is bits 9..5, c2 is bits 4..0.
    """
    code0 = char_to_code[c0]
    code1 = char_to_code[c1]
    code2 = char_to_code[c2]
    return (code0 << 10) | (code1 << 5) | code2

def prepare_corpus_and_splits(raw_text_path, output_dir=None, write_artifacts=False):
    """Processes Gutenberg raw text according to the frozen protocol.

    Args:
        raw_text_path: Path to raw Gutenberg text file.
        output_dir: Directory to write manifests if write_artifacts is True.
        write_artifacts: If True, writes splits_manifest.json and encoding_maps.json
            to output_dir. Default is False (in-memory processing with zero disk writes).
    """
    if write_artifacts:
        if not output_dir:
            raise ValueError("output_dir must be specified when write_artifacts=True")
        os.makedirs(output_dir, exist_ok=True)
    with open(raw_text_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
        
    start_m = re.search(r'\*\*\* START OF (THE|THIS) PROJECT GUTENBERG EBOOK[^\*]*\*\*\*', text)
    end_m = re.search(r'\*\*\* END OF (THE|THIS) PROJECT GUTENBERG EBOOK[^\*]*\*\*\*', text)
    if not start_m or not end_m:
        raise ValueError("Could not find Gutenberg START and END markers")
        
    body = text[start_m.end():end_m.start()]
    
    alphabet_set = set(ALPHABET)
    raw_passages = re.split(r'\r?\n\s*\r?\n+', body)
    
    passages = []
    total_raw_chars = sum(len(p) for p in raw_passages)
    
    for idx, raw in enumerate(raw_passages):
        norm = unicodedata.normalize("NFKD", raw)
        norm = "".join(c for c in norm if not unicodedata.combining(c))
        norm = norm.lower()
        norm = "".join(c if c in alphabet_set else " " for c in norm)
        norm = re.sub(r"\s+", " ", norm).strip()
        if len(norm) >= 64:
            p_sha = hashlib.sha256(norm.encode("utf-8")).hexdigest()
            passages.append({
                "passage_id": idx,
                "length": len(norm),
                "sha256": p_sha,
                "text": norm
            })
            
    total_norm_chars = sum(p["length"] for p in passages)
    
    # Group identical normalized passages by SHA-256
    group_map = {}
    for p in passages:
        h = p["sha256"]
        if h not in group_map:
            group_map[h] = []
        group_map[h].append(p)
        
    group_hashes = sorted(group_map.keys())
    
    # Permute with NumPy PCG64 seed 20261105
    rng = np.random.Generator(np.random.PCG64(20261105))
    permuted_hashes = rng.permutation(group_hashes).tolist()
    
    n_groups = len(permuted_hashes)
    n_train = math.floor(0.70 * n_groups)
    n_val = math.floor(0.15 * n_groups)
    
    train_groups = set(permuted_hashes[:n_train])
    val_groups = set(permuted_hashes[n_train:n_train + n_val])
    test_groups = set(permuted_hashes[n_train + n_val:])
    
    # Assign splits to passages
    splits = {"train": [], "val": [], "test": []}
    for p in passages:
        h = p["sha256"]
        if h in train_groups:
            p["split"] = "train"
            splits["train"].append(p)
        elif h in val_groups:
            p["split"] = "val"
            splits["val"].append(p)
        else:
            p["split"] = "test"
            splits["test"].append(p)
            
    manifest = {
        "seed": 20261105,
        "n_raw_passages": len(raw_passages),
        "total_raw_chars": total_raw_chars,
        "n_valid_passages": len(passages),
        "total_norm_chars": total_norm_chars,
        "retained_fraction_chars": total_norm_chars / total_raw_chars if total_raw_chars > 0 else 0.0,
        "n_unique_groups": n_groups,
        "n_train_groups": len(train_groups),
        "n_val_groups": len(val_groups),
        "n_test_groups": len(test_groups),
        "n_train_passages": len(splits["train"]),
        "n_val_passages": len(splits["val"]),
        "n_test_passages": len(splits["test"]),
        "passages": [
            {
                "passage_id": p["passage_id"],
                "group_sha256": p["sha256"],
                "length": p["length"],
                "split": p["split"]
            }
            for p in passages
        ]
    }
    
    encoding_maps = build_encoding_maps()
    if write_artifacts:
        with open(os.path.join(output_dir, "splits_manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        with open(os.path.join(output_dir, "encoding_maps.json"), "w") as f:
            json.dump(encoding_maps, f, indent=2)
            
    return passages, splits, encoding_maps, manifest

def extract_chunks_and_counts(passages_split, char_to_code):
    """Extract overlapping 3-character chunks for each passage and overall counts."""
    counts = np.zeros(32768, dtype=np.int64)
    per_passage_chunks = []
    
    for p in passages_split:
        text = p["text"]
        p_counts = np.zeros(32768, dtype=np.int32)
        # overlapping 3-char chunks
        for i in range(len(text) - 2):
            st = encode_chunk(text[i], text[i+1], text[i+2], char_to_code)
            p_counts[st] += 1
            counts[st] += 1
        per_passage_chunks.append({
            "passage_id": p["passage_id"],
            "sha256": p["sha256"],
            "chunk_count": len(text) - 2,
            "counts": p_counts
        })
        
    return counts, per_passage_chunks
