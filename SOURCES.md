# Source Catalog & Provenance

This document catalogs all primary data sources, external software repositories, scientific literature, and public domain legal citations for the 15-qubit quantum language distribution experiment.

---

## 1. Primary Text Corpus: Project Gutenberg Ebook #100

- **Work:** *The Complete Works of William Shakespeare*
- **Author:** William Shakespeare (1564–1616)
- **Publisher / Archive:** Project Gutenberg Literary Archive Foundation
- **Catalog Entry:** [Project Gutenberg Ebook #100](https://www.gutenberg.org/ebooks/100)
- **Raw UTF-8 Endpoint:** `https://www.gutenberg.org/ebooks/100.txt.utf-8`
- **Resolved Content URL:** `https://www.gutenberg.org/cache/epub/100/pg100.txt`
- **Local Archived File:** [`data/gutenberg_100_raw.txt`](data/gutenberg_100_raw.txt)
- **Local Metadata Record:** [`data/gutenberg_source_metadata.json`](data/gutenberg_source_metadata.json)

### Integrity Attributes
- **Byte Size:** `5,638,481` bytes (verified matching HTTP `Content-Length`)
- **SHA-256 Digest:** `eb61e33cab23dbe712ad57091ded6c7292ba91ac3bb2764f51a3223ec0068fe5`
- **Release Date:** January 1, 1994 (most recently updated: October 17, 2026)

### Legal & Copyright Status in the United States
- The Project Gutenberg catalogue labels ebook 100 "Public domain in the USA."
- The original download, including the complete Project Gutenberg licence header and footer, is retained unchanged in `data/gutenberg_100_raw.txt`. This provenance record does not make a worldwide legal claim about every edition or redistribution format.

---

## 2. Pinned Quantum Optimization Package: Xanadu `iqpopt`

- **Repository:** [XanaduAI/iqpopt on GitHub](https://github.com/XanaduAI/iqpopt)
- **Pinned Commit Hash:** `18ae04255ed80a274d8c49ed91cd4e6214757ccd`
- **Package Version:** `2024.7.0`
- **Software License:** Apache License 2.0
- **Maintainers / Authors:**
  - Erik Armengol, as listed on the IQPopt paper
  - Joseph Bowles

### Scientific Citation
- **Article:** *IQPopt: Fast optimization of instantaneous quantum polynomial circuits in JAX*
- **Primary arXiv Identifier:** [arXiv:2501.04776 [quant-ph]](https://arxiv.org/abs/2501.04776)
- **URL:** `https://arxiv.org/abs/2501.04776`
- **Upstream Active Development:** Development of train-classical-deploy-quantum (TCDQ) generative modeling has transitioned into the PennyLane Labs module:
  - [PennyLane Labs TCDQ Module](https://docs.pennylane.ai/en/latest/code/api/pennylane.labs.tcdq.html#module-pennylane.labs.tcdq)

---

## 3. Foundational Frameworks & Dependencies

- **PennyLane:** Quantum Machine Learning library by Xanadu.
  - Repository: [PennyLaneAI/pennylane](https://github.com/PennyLaneAI/pennylane)
  - Pinned Version: `0.45.1`
- **JAX:** Composable transformations of Python+NumPy programs (autodiff and XLA compilation).
  - Repository: [jax-ml/jax](https://github.com/jax-ml/jax)
  - Pinned Version: `0.12.0.1`
- **Optax:** First-order optimization library for JAX.
  - Repository: [google-deepmind/optax](https://github.com/google-deepmind/optax)
  - Pinned Version: `0.2.6`
- **NumPy:** Numerical computing foundation.
  - Pinned Version: `2.5.4`
- **Dependency Lock:** See [`dependency_lock.txt`](dependency_lock.txt) for the complete frozen dependency environment.

---

## 4. Theoretical Hypothesis & Background Article

- **Article:** *Stop Porting GPU Training to Quantum Computers*
- **Author:** Abhinay Sharma
- **Publication:** Medium (July 2026)
- **URL:** [https://medium.com/@sharmaabhinay28/stop-porting-gpu-training-to-quantum-computers-fd7cf5cf3bdb](https://medium.com/@sharmaabhinay28/stop-porting-gpu-training-to-quantum-computers-fd7cf5cf3bdb)
- **Central Thesis Investigated:** Parameterized quantum circuits applied to NLP sequence generation without a problem-native quantum inductive bias underperform classical baselines matched for parameter budget and sample access.
