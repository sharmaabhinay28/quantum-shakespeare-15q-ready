import jax
import jax.numpy as jnp
import numpy as np
import optax
from iqpopt.iqp_optimizer import IqpSimulator
from iqpopt.gen_qml.iqp_methods import mmd_loss_iqp
from .evaluation import (
    get_standard_gates,
    probs_from_iqp,
    probs_from_bitflip,
    probs_from_autoregressive,
    probs_from_table
)

def create_training_sampler(train_counts, seed):
    """Creates a generator that samples batches of 512 15-bit vectors
    with replacement from the full training histogram.
    """
    rng = np.random.Generator(np.random.PCG64(seed))
    # Active states and their probabilities in training data
    active_indices = np.where(train_counts > 0)[0]
    active_probs = train_counts[active_indices] / np.sum(train_counts)
    
    # Precompute bit matrix for all 32768 states: shape (32768, 15) float64
    all_bits = np.array([[(s >> (14 - i)) & 1 for i in range(15)] for s in range(32768)], dtype=np.float64)
    
    def sample_batch(batch_size=512):
        sampled_states = rng.choice(active_indices, size=batch_size, replace=True, p=active_probs)
        return all_bits[sampled_states]
        
    return sample_batch

# ----------------- 1. IQP Model -----------------
class IQPModel:
    def __init__(self, n_qubits=15):
        self.n_qubits = n_qubits
        self.gates = get_standard_gates(n_qubits)
        self.n_params = len(self.gates) # 120
        self.sim = IqpSimulator(n_qubits, self.gates, device="default.qubit")
        
    def init_params(self, seed):
        rng = np.random.Generator(np.random.PCG64(seed))
        return rng.uniform(-0.1, 0.1, self.n_params).astype(np.float64)
        
    def train(self, params_init, sampler, lr, n_updates=500, seed=123):
        optimizer = optax.adam(lr)
        opt_state = optimizer.init(params_init)
        params = jnp.array(params_init, dtype=jnp.float64)
        
        sim = self.sim
        @jax.jit
        def step(p, state, b, k):
            def loss_fn(p_inner):
                return mmd_loss_iqp(
                    params=p_inner,
                    iqp_circuit=sim,
                    ground_truth=b,
                    sigma=[1.0, 2.0, 4.0],
                    n_ops=128,
                    n_samples=128,
                    key=k
                )
            val, grad = jax.value_and_grad(loss_fn)(p)
            updates, new_state = optimizer.update(grad, state, p)
            new_params = optax.apply_updates(p, updates)
            return new_params, new_state, val

        key = jax.random.PRNGKey(seed)
        losses = []
        for _ in range(n_updates):
            batch = jnp.array(sampler(512), dtype=jnp.float64)
            key, subkey = jax.random.split(key)
            params, opt_state, val = step(params, opt_state, batch, subkey)
            losses.append(float(val))
            
        final_params = np.array(params, dtype=np.float64)
        return final_params, losses
        
    def get_probs(self, params):
        return probs_from_iqp(params, self.gates, self.n_qubits)

# ------------- 2. Stochastic Bitflip Model -------------
class StochasticBitflipModel:
    def __init__(self, n_qubits=15):
        self.n_qubits = n_qubits
        self.gates = get_standard_gates(n_qubits)
        self.n_params = len(self.gates) # 120
        self.sim = IqpSimulator(n_qubits, self.gates, bitflip=True)
        
    def init_params(self, seed):
        rng = np.random.Generator(np.random.PCG64(seed))
        return rng.uniform(-0.1, 0.1, self.n_params).astype(np.float64)
        
    def train(self, params_init, sampler, lr, n_updates=500, seed=123):
        optimizer = optax.adam(lr)
        opt_state = optimizer.init(params_init)
        params = jnp.array(params_init, dtype=jnp.float64)
        
        sim = self.sim
        @jax.jit
        def step(p, state, b, k):
            def loss_fn(p_inner):
                return mmd_loss_iqp(
                    params=p_inner,
                    iqp_circuit=sim,
                    ground_truth=b,
                    sigma=[1.0, 2.0, 4.0],
                    n_ops=128,
                    n_samples=128,
                    key=k
                )
            val, grad = jax.value_and_grad(loss_fn)(p)
            updates, new_state = optimizer.update(grad, state, p)
            new_params = optax.apply_updates(p, updates)
            return new_params, new_state, val

        key = jax.random.PRNGKey(seed)
        losses = []
        for _ in range(n_updates):
            batch = jnp.array(sampler(512), dtype=jnp.float64)
            key, subkey = jax.random.split(key)
            params, opt_state, val = step(params, opt_state, batch, subkey)
            losses.append(float(val))
            
        final_params = np.array(params, dtype=np.float64)
        return final_params, losses
        
    def get_probs(self, params):
        return probs_from_bitflip(params, self.gates, self.n_qubits)

# ----------- 3. Autoregressive Bernoulli Model -----------
class AutoregressiveBernoulliModel:
    def __init__(self, n_qubits=15):
        self.n_qubits = n_qubits
        self.n_biases = n_qubits # 15
        self.n_weights = (n_qubits * (n_qubits - 1)) // 2 # 105
        self.n_params = self.n_biases + self.n_weights # 120
        self.all_bits = np.array([[(s >> (n_qubits - 1 - i)) & 1 for i in range(n_qubits)] for s in range(1 << n_qubits)], dtype=np.float64)
        
        # Build lower triangular mask for W
        tril_indices = np.tril_indices(n_qubits, k=-1)
        self.tril_row = jnp.array(tril_indices[0])
        self.tril_col = jnp.array(tril_indices[1])
        
    def init_params(self, seed):
        rng = np.random.Generator(np.random.PCG64(seed))
        biases = rng.normal(0.0, 0.1, self.n_biases).astype(np.float64)
        weights = rng.normal(0.0, 0.1, self.n_weights).astype(np.float64)
        return np.concatenate([biases, weights])
        
    def train(self, params_init, sampler, lr, n_updates=500, seed=123):
        optimizer = optax.adam(lr)
        opt_state = optimizer.init(params_init)
        params = jnp.array(params_init, dtype=jnp.float64)
        
        tril_row, tril_col = self.tril_row, self.tril_col
        n_qubits = self.n_qubits
        
        @jax.jit
        def step(p, state, b):
            def loss_fn(p_inner):
                biases = p_inner[:n_qubits]
                weights = p_inner[n_qubits:]
                
                # Construct lower triangular matrix W
                W = jnp.zeros((n_qubits, n_qubits), dtype=jnp.float64)
                W = W.at[tril_row, tril_col].set(weights)
                
                logits = b @ W.T + biases # shape (batch, 15)
                # Binary cross-entropy (negative log likelihood)
                # log(sigma(logit)) = -softplus(-logit)
                # log(1 - sigma(logit)) = -softplus(logit)
                log_p_1 = -jax.nn.softplus(-logits)
                log_p_0 = -jax.nn.softplus(logits)
                log_prob = b * log_p_1 + (1.0 - b) * log_p_0
                return -jnp.mean(jnp.sum(log_prob, axis=1))

            val, grad = jax.value_and_grad(loss_fn)(p)
            updates, new_state = optimizer.update(grad, state, p)
            new_params = optax.apply_updates(p, updates)
            return new_params, new_state, val

        losses = []
        for _ in range(n_updates):
            batch = jnp.array(sampler(512), dtype=jnp.float64)
            params, opt_state, val = step(params, opt_state, batch)
            losses.append(float(val))
            
        final_params = np.array(params, dtype=np.float64)
        return final_params, losses
        
    def get_probs(self, params):
        biases = params[:self.n_qubits]
        weights = params[self.n_qubits:]
        return probs_from_autoregressive(biases, weights, self.all_bits, self.n_qubits)

# ----------------- 4. Frequency Table Model -----------------
class FrequencyTableModel:
    def __init__(self, n_qubits=15):
        self.n_qubits = n_qubits
        self.num_states = 1 << n_qubits
        
    def fit(self, train_counts, alpha):
        return probs_from_table(train_counts, alpha=alpha, num_states=self.num_states)
