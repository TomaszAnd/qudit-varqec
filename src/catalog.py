"""
Catalog of trained VarQEC codes with unified loading interface.

Includes analytically constructed benchmark codes (e.g. [[5,1,3]]_{Z_q}).
"""
import os
import numpy as np
from pennylane import numpy as pnp

_RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "results", "params")


def five_qudit_code_states(q=3):
    """
    Construct [[5,1,3]]_{Z_q} codewords analytically (Chau 1997, Rains 1997).

    The encoding isometry:
      T_{a,k,l,m,n,p} = delta(a = k+l+m+n+p mod q) * (1/q^2) * omega^(kl+lm+mn+np+pk)

    where omega = exp(2*pi*i/q).

    Args:
        q: qudit dimension (default 3 for qutrits)

    Returns:
        (K, dim) complex array of K=q orthonormal codewords in q^5 dimensions
    """
    omega = np.exp(2j * np.pi / q)
    dim = q ** 5
    K = q
    code_states = np.zeros((K, dim), dtype=complex)

    for a in range(q):
        for k in range(q):
            for l in range(q):
                for m in range(q):
                    for n in range(q):
                        p = (a - k - l - m - n) % q
                        phase = omega ** (k*l + l*m + m*n + n*p + p*k)
                        idx = k * q**4 + l * q**3 + m * q**2 + n * q + p
                        code_states[a, idx] = phase / q**2

    return code_states


# Registry: name -> (npz_path, encoder_factory_args)
_CODE_REGISTRY = {
    # Native-gate codes
    'qutrit_d2': ('qutrit_d2_1layer_seed42.npz', {'d': 3, 'n_qudit': 3}),
    'ququart_d2': ('native_d4_dist2_1layer_n3_seed42.npz', {'d': 4, 'n_qudit': 3}),
    'qutrit_d3': ('native_d3_dist3_10layer_closed_best.npz', {'d': 3, 'n_qudit': 5}),
    # Abstract-gate ququart codes
    'dephasing_d2': ('dephasing_d2_1layer_seed42.npz', {'d': 4, 'n_qudit': 5, 'abstract': True}),
    'dephasing_d3': ('dephasing_d3_2layer_seed2.npz', {'d': 4, 'n_qudit': 5, 'abstract': True}),
    'depolarizing_d2': ('depolarizing_d2_2layer_seed42.npz', {'d': 4, 'n_qudit': 5, 'abstract': True}),
    'correlated_d2': ('correlated_simplified_d2_2layer_seed42.npz', {'d': 4, 'n_qudit': 5, 'abstract': True}),
    'correlated_d3': ('correlated_simplified_d3_3layer_seed42.npz', {'d': 4, 'n_qudit': 5, 'abstract': True}),
    # New campaign codes
    'qutrit_n7_d3': ('d3_n7_dist3_8L_seed0.npz', {'d': 3, 'n_qudit': 7}),
    'qutrit_n8_d3': ('d3_n8_dist3_8L_seed0.npz', {'d': 3, 'n_qudit': 8}),
    'ququart_n5_d3': ('d4_n5_dist3_6L_seed0.npz', {'d': 4, 'n_qudit': 5}),
    'ququint_d2': ('d5_n5_dist2_2L_best3s_seed0.npz', {'d': 5, 'n_qudit': 5}),
    # Analytical benchmark
    'five_qudit_d3': (None, {'d': 3, 'n_qudit': 5, 'analytical': True}),
}


def _auto_register_campaign_codes():
    """Scan results/params/ for campaign codes and add to registry."""
    import glob
    import re
    for npz_path in glob.glob(os.path.join(_RESULTS_DIR, "d*_n*_dist*.npz")):
        fname = os.path.basename(npz_path)
        m = re.match(r'd(\d+)_n(\d+)_dist(\d+)_(\d+)L', fname)
        if m:
            d, n, dist = int(m.group(1)), int(m.group(2)), int(m.group(3))
            name = f"d{d}_n{n}_dist{dist}"
            if name not in _CODE_REGISTRY:
                _CODE_REGISTRY[name] = (fname, {'d': d, 'n_qudit': n})

try:
    _auto_register_campaign_codes()
except Exception:
    pass


def list_codes():
    """List all available code names."""
    return list(_CODE_REGISTRY.keys())


def load_code(name):
    """
    Load a trained VarQEC code by name.

    Returns dict with:
      'code_states': (K, dim) array of orthonormal codewords
      'params': trained parameters (None for analytical codes)
      'metadata': dict with d, n_qudit, distance, n_layers, loss, etc.
      'name': the code name
    """
    if name not in _CODE_REGISTRY:
        raise ValueError(f"Unknown code '{name}'. Available: {list_codes()}")

    npz_file, info = _CODE_REGISTRY[name]

    # Analytical benchmark code
    if info.get('analytical'):
        q = info['d']
        code_states = five_qudit_code_states(q)
        return {
            'code_states': code_states,
            'params': None,
            'metadata': {
                'd': q, 'n_qudit': 5, 'K': q, 'distance': 3,
                'n_layers': 0, 'loss': 0.0,
                'type': 'stabilizer', 'name': f'[[5,1,3]]_Z{q}',
            },
            'name': name,
        }

    # Load from npz
    from src.loss import load_varqec_result
    path = os.path.join(_RESULTS_DIR, npz_file)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Trained code not found: {path}")

    data = load_varqec_result(path)
    params = pnp.array(data['params'], requires_grad=False)

    d = info['d']
    n_qudit = info['n_qudit']
    K = d

    # Extract code states
    if info.get('abstract'):
        from src.legacy.ququart_pipeline import create_encoder
        encoder, _ = create_encoder(n_qudit, d)
    else:
        from src.encoder import create_native_encoder
        encoder, _, _ = create_native_encoder(n_qudit, d, force_manual=True)

    code_states = np.zeros((K, d ** n_qudit), dtype=complex)
    for k in range(K):
        code_states[k] = np.array(encoder(params, k))

    metadata = {
        'd': d, 'n_qudit': n_qudit, 'K': K,
        'distance': int(data.get('distance', 2)),
        'n_layers': int(data.get('n_layers', 1)),
        'final_loss': float(data.get('final_loss', -1)),
        'n_steps': len(data.get('losses', [])),
        'noise_type': str(data.get('noise_type', 'unknown')),
        'abstract_gates': info.get('abstract', False),
    }

    return {
        'code_states': code_states,
        'params': params,
        'metadata': metadata,
        'name': name,
    }
