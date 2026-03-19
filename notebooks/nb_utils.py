"""Utility functions for the showcase notebook. Keeps cells concise."""
import numpy as np
import itertools
import sys
sys.path.insert(0, '..')
from src.pauli_ops import single_qudit_paulis


def load_trained_code(noise_type, distance, n_layers, seed):
    """Load trained params from results/params/. Returns (losses, theta) or (None, None)."""
    import os
    from src.kl_loss_fast import load_varqec_result
    from pennylane import numpy as pnp
    path = f"../results/params/{noise_type}_d{distance}_{n_layers}layer_seed{seed}.npz"
    if not os.path.exists(path):
        print(f"  {path} not found — run training script first")
        return None, None
    data = load_varqec_result(path)
    losses = list(data['losses'])
    theta = pnp.array(data['params'], requires_grad=False)
    print(f"  Loaded {noise_type} d={distance}: {len(losses)} steps, final loss={losses[-1]:.2e}")
    return losses, theta


def get_code_states(theta, n_qudits=5, dim_qudit=4):
    """Extract code state vectors from trained VarQEC parameters."""
    import numpy as np
    from src.encoder import create_encoder
    K = dim_qudit
    encoder, _ = create_encoder(n_qudits, dim_qudit)
    code_states = np.zeros((K, dim_qudit**n_qudits), dtype=complex)
    for k in range(K):
        code_states[k] = np.array(encoder(theta, k))
    return code_states


def kl_residuals(code_states, E_det, E_corr, distance, max_m_samples=500):
    """
    Compute KL condition residuals:
    - off_diag: max |<psi_i|E|psi_j>| over all i<j, E in E_det
    - diag_var: max Var_k(<psi_k|M|psi_k>) over sampled M = E_a†E_b
    """
    K = code_states.shape[0]
    max_off = 0.0
    for E in E_det:
        for i in range(K):
            for j in range(i + 1, K):
                val = abs(np.vdot(code_states[i], E @ code_states[j]))
                if val > max_off:
                    max_off = val

    max_var = 0.0
    if distance >= 3:
        n_corr = len(E_corr)
        pairs = list(itertools.product(range(n_corr), repeat=2))
        if len(pairs) > max_m_samples:
            rng = np.random.default_rng(0)
            pairs = [pairs[i] for i in rng.choice(len(pairs), max_m_samples, replace=False)]
        for ia, ib in pairs:
            M = E_corr[ia].conj().T @ E_corr[ib]
            vals = np.array([np.real(np.vdot(code_states[k], M @ code_states[k]))
                             for k in range(K)])
            v = np.var(vals)
            if v > max_var:
                max_var = v
    return max_off, max_var


def kl_residuals_factored(code_states, E_det_factors, E_corr_factors, distance,
                          n_qudits=5, dim_qudit=4, max_m_samples=500):
    """
    Compute KL condition residuals using factored error representation.
    Same output as kl_residuals but avoids materializing dense d^n x d^n matrices.
    """
    from src.error_sets_factored import apply_factored_error, apply_factored_error_dag
    K = code_states.shape[0]
    max_off = 0.0
    for factors in E_det_factors:
        for i in range(K):
            E_psi_j_cache = {}
            for j in range(i + 1, K):
                if j not in E_psi_j_cache:
                    E_psi_j_cache[j] = apply_factored_error(code_states[j], factors, n_qudits, dim_qudit)
                val = abs(np.vdot(code_states[i], E_psi_j_cache[j]))
                if val > max_off:
                    max_off = val

    max_var = 0.0
    if distance >= 3:
        n_corr = len(E_corr_factors)
        pairs = list(itertools.product(range(n_corr), repeat=2))
        if len(pairs) > max_m_samples:
            rng = np.random.default_rng(0)
            pairs = [pairs[i] for i in rng.choice(len(pairs), max_m_samples, replace=False)]
        for ia, ib in pairs:
            # <psi_k|E_a†E_b|psi_k> = <E_a psi_k | E_b psi_k>
            vals = np.array([
                np.real(np.vdot(
                    apply_factored_error(code_states[k], E_corr_factors[ia], n_qudits, dim_qudit),
                    apply_factored_error(code_states[k], E_corr_factors[ib], n_qudits, dim_qudit)
                ))
                for k in range(K)
            ])
            v = np.var(vals)
            if v > max_var:
                max_var = v
    return max_off, max_var


def weight_enumerators(code_states, n_qudits=5, dim_qudit=4, max_weight=1):
    """
    Compute Shor-Laflamme weight enumerators A_j, B_j for j=0..max_weight.
    Weight-1 has 75 ops (~16s/code). Weight-2 has 2250 ops (~9 min/code).
    """
    K, dim = code_states.shape
    P_c = code_states.T @ np.conj(code_states)

    single_paulis = [np.eye(dim_qudit, dtype=complex)] + list(single_qudit_paulis())
    non_id = single_paulis[1:]
    Id = np.eye(dim_qudit, dtype=complex)

    A = np.zeros(max_weight + 1)
    B = np.zeros(max_weight + 1)

    for w in range(max_weight + 1):
        if w == 0:
            A[0] = np.abs(np.trace(P_c))**2 / K**2
            B[0] = np.real(np.trace(P_c @ P_c)) / K
            continue
        for qudit_subset in itertools.combinations(range(n_qudits), w):
            for err_choice in itertools.product(non_id, repeat=w):
                op = np.array([[1.0]], dtype=complex)
                idx = 0
                for q in range(n_qudits):
                    if q in qudit_subset:
                        op = np.kron(op, err_choice[idx])
                        idx += 1
                    else:
                        op = np.kron(op, Id)
                A[w] += np.abs(np.trace(op @ P_c))**2
                B[w] += np.real(np.trace(op @ P_c @ op.conj().T @ P_c))
        A[w] /= K**2
        B[w] /= K
    return A, B


# Consistent styling
COLORS = {
    'dephasing_d2': '#2196F3', 'dephasing_d3': '#1565C0',
    'depolarizing_d2': '#F44336', 'depolarizing_d3': '#D32F2F',
    'corr_simplified_d2': '#FF9800', 'corr_simplified_d3': '#E65100',
}
MARKERS = {
    'dephasing_d2': 'o', 'dephasing_d3': 'D',
    'depolarizing_d2': 's', 'depolarizing_d3': 'P',
    'corr_simplified_d2': '^', 'corr_simplified_d3': 'v',
}
LABELS = {
    'dephasing_d2': 'Dephasing d=2', 'dephasing_d3': 'Dephasing d=3',
    'depolarizing_d2': 'Depolarizing d=2', 'depolarizing_d3': 'Depolarizing d=3',
    'corr_simplified_d2': 'Corr. simplified d=2', 'corr_simplified_d3': 'Corr. simplified d=3',
}
