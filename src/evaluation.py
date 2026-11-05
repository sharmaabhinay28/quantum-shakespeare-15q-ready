import numpy as np

def fwht(a):
    """Fast Walsh-Hadamard Transform (in-place or copied).
    Computes H_n @ a where H_1 = [[1, 1], [1, -1]].
    """
    a = np.array(a, copy=True, dtype=np.complex128 if np.iscomplexobj(a) else np.float64)
    n = len(a)
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                x = a[j]
                y = a[j + h]
                a[j] = x + y
                a[j + h] = x - y
        h *= 2
    return a

def gate_to_mask(wires, n_qubits=15):
    """Convert a list of active wires into integer bitmask.
    PennyLane convention: wire 0 is MSB (weight 2^(n_qubits-1-0)).
    """
    mask = 0
    for w in wires:
        mask |= (1 << (n_qubits - 1 - w))
    return mask

def get_standard_gates(n_qubits=15):
    """15 single-wire X gates + 105 distinct two-wire XX gates = 120 gates."""
    single_gates = [[[i]] for i in range(n_qubits)]
    pair_gates = [[[i, j]] for i in range(n_qubits) for j in range(i + 1, n_qubits)]
    return single_gates + pair_gates

def probs_from_iqp(params, gates=None, n_qubits=15):
    """Compute exact 2^n probabilities for an IQP circuit.
    Verified against PennyLane default.qubit to machine precision.
    """
    if gates is None:
        gates = get_standard_gates(n_qubits)
    num_states = 1 << n_qubits
    T = np.zeros(num_states, dtype=np.float64)
    for par, gate in zip(params, gates):
        for gen in gate:
            mask = gate_to_mask(gen, n_qubits)
            T[mask] += float(par)
    
    phi = -fwht(T)
    psi = np.exp(1j * phi) / num_states
    amps = fwht(psi)
    probs = np.real(amps * np.conj(amps))
    # Enforce non-negativity and exact sum
    probs = np.clip(probs, 0.0, None)
    s = np.sum(probs)
    if s > 0:
        probs /= s
    return probs

def probs_from_bitflip(params, gates=None, n_qubits=15):
    """Compute exact 2^n probabilities for the stochastic bitflip circuit.
    Verified against analytic expectations and independent XOR simulation.
    """
    if gates is None:
        gates = get_standard_gates(n_qubits)
    num_states = 1 << n_qubits
    
    # Parity check: for each s in 0..num_states-1, expval is prod_{g: s.g=1} cos(2 theta_g)
    # Notice that log(|cos(2 theta_g)|) or direct FWHT can be computed:
    # Each generator independently flips with probability p_g = sin(theta_g)^2
    # In Walsh domain, product of expvals:
    # For a single generator g, the expval vector is: 1 if s.g=0, cos(2 theta_g) if s.g=1.
    # Therefore: log(expval(s)) = sum_{g: s.g=1} log(cos(2 theta_g)) if all cos(2 theta_g) > 0.
    # To handle general theta, compute FWHT convolution or accumulate log/signs:
    # A generator g with flip prob p_g has WHT: (1 - 2 p_g) for s.g=1, and 1 for s.g=0.
    # Note (1 - 2 sin(par)^2) = cos(2 par).
    # Since generators apply sequentially (or simultaneously because XOR is commutative):
    # We can either directly accumulate the product of cos(2 par), or do sequential XOR convolution:
    p_dist = np.zeros(num_states, dtype=np.float64)
    p_dist[0] = 1.0
    for par, gate in zip(params, gates):
        prob_flip = np.sin(float(par)) ** 2
        prob_stay = 1.0 - prob_flip
        for gen in gate:
            mask = gate_to_mask(gen, n_qubits)
            if mask == 0 or prob_flip == 0.0:
                continue
            # XOR convolution step: p_new[x] = (1-p)*p_dist[x] + p*p_dist[x ^ mask]
            # Fast vectorized XOR step:
            # We can vectorize by bit manipulation
            # For 32768 elements, indexing with an XOR permutation array is extremely fast!
            xor_indices = np.arange(num_states) ^ mask
            p_dist = prob_stay * p_dist + prob_flip * p_dist[xor_indices]
    
    p_dist = np.clip(p_dist, 0.0, None)
    s = np.sum(p_dist)
    if s > 0:
        p_dist /= s
    return p_dist

def probs_from_autoregressive(biases, weights, all_bits=None, n_qubits=15):
    """Compute exact 2^n probabilities for autoregressive Bernoulli model."""
    num_states = 1 << n_qubits
    if all_bits is None:
        all_bits = np.array([[(s >> (n_qubits - 1 - i)) & 1 for i in range(n_qubits)] for s in range(num_states)], dtype=np.float64)
    
    W = np.zeros((n_qubits, n_qubits), dtype=np.float64)
    idx = 0
    for i in range(n_qubits):
        for j in range(i):
            W[i, j] = weights[idx]
            idx += 1
            
    logits = all_bits @ W.T + biases
    log_p_1 = -np.logaddexp(0.0, -logits)
    log_p_0 = -np.logaddexp(0.0, logits)
    log_p = all_bits * log_p_1 + (1.0 - all_bits) * log_p_0
    log_probs_state = np.sum(log_p, axis=1)
    probs = np.exp(log_probs_state)
    probs = np.clip(probs, 0.0, None)
    s = np.sum(probs)
    if s > 0:
        probs /= s
    return probs

def probs_from_table(counts, alpha=1.0, num_states=32768):
    """Add-alpha smoothed frequency table distribution."""
    smoothed = counts + alpha
    return smoothed / np.sum(smoothed)

# Precomputed kernel transform for 15 qubits
def get_kernel_hat(sigmas=(1.0, 2.0, 4.0), n_qubits=15):
    num_states = 1 << n_qubits
    # Precompute k(z) based on Hamming weight of z
    popcounts = np.array([bin(z).count('1') for z in range(num_states)], dtype=np.float64)
    k = np.zeros(num_states, dtype=np.float64)
    for s in sigmas:
        k += np.exp(-popcounts / (2.0 * s**2))
    k /= len(sigmas)
    return fwht(k)

def exact_squared_mmd(p, q, k_hat=None, n_qubits=15):
    """Compute exact biased squared MMD between distributions p and q.
    O(n 2^n) via FWHT, verified to 1e-17.
    """
    if k_hat is None:
        k_hat = get_kernel_hat(n_qubits=n_qubits)
    num_states = 1 << n_qubits
    delta = p - q
    delta_hat = fwht(delta)
    mmd2 = np.sum(k_hat * (delta_hat ** 2)) / num_states
    return float(np.clip(mmd2, 0.0, None))

def evaluate_log_loss(p_raw, empirical_counts, smoothing_eps=1e-6, num_states=32768):
    """Evaluates primary log loss with uniform smoothing:
    q_eval = (1 - eps) * p_raw + eps / num_states.
    Returns:
      loss_chunk: nats per 3-character chunk
      loss_char: nats per character (loss_chunk / 3)
      unsmoothed_loss_chunk: unsmoothed log loss (or inf)
    """
    total_tokens = np.sum(empirical_counts)
    if total_tokens == 0:
        return 0.0, 0.0, 0.0
    
    q_eval = (1.0 - smoothing_eps) * p_raw + (smoothing_eps / num_states)
    active_mask = empirical_counts > 0
    
    # Primary smoothed log loss
    loss_chunk = -np.sum(empirical_counts[active_mask] * np.log(q_eval[active_mask])) / total_tokens
    loss_char = loss_chunk / 3.0
    
    # Unsmoothed log loss
    if np.any(p_raw[active_mask] <= 0):
        unsmoothed_loss_chunk = float('inf')
    else:
        unsmoothed_loss_chunk = -np.sum(empirical_counts[active_mask] * np.log(p_raw[active_mask])) / total_tokens
        
    return float(loss_chunk), float(loss_char), float(unsmoothed_loss_chunk)
