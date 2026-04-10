"""
Code characterization functions for VarQEC codes.

Computes weight enumerators, KL residuals, distance verification,
and entanglement entropy for any code (VarQEC or analytical).
"""
import numpy as np
import itertools


def compute_kl_residuals(code_states, E_corr):
    """
    Compute KL condition residuals for ALL M = E_a†E_b pairs.

    Args:
        code_states: (K, dim) array
        E_corr: list of correction error operators

    Returns:
        (max_offdiag, max_diag_var): worst-case off-diagonal overlap
                                     and worst-case diagonal variance
    """
    K = code_states.shape[0]
    max_offdiag = 0.0
    max_diag_var = 0.0

    for Ea in E_corr:
        Ea_dag = Ea.conj().T
        for Eb in E_corr:
            M = Ea_dag @ Eb
            # Off-diagonal
            for i in range(K):
                for j in range(i + 1, K):
                    val = abs(np.vdot(code_states[i], M @ code_states[j]))
                    max_offdiag = max(max_offdiag, val)
            # Diagonal variance
            diags = np.array([float(np.real(np.vdot(code_states[k], M @ code_states[k])))
                              for k in range(K)])
            max_diag_var = max(max_diag_var, float(np.var(diags)))

    return max_offdiag, max_diag_var


def compute_distance(code_states, n_qudit, d, single_error_basis, max_weight=3):
    """
    Verify code distance by checking weight-w errors.

    Returns the minimum weight at which detection fails, or max_weight+1 if
    all errors up to max_weight are detected.

    Detection fails if max|⟨ψ_i|E|ψ_j⟩| > threshold for some i≠j.
    """
    K = code_states.shape[0]
    Id = np.eye(d, dtype=complex)
    threshold = 0.01

    for w in range(1, max_weight + 1):
        worst = 0.0
        for qudit_subset in itertools.combinations(range(n_qudit), w):
            for err_choice in itertools.product(single_error_basis, repeat=w):
                op = 1
                idx = 0
                for q in range(n_qudit):
                    if q in qudit_subset:
                        op = np.kron(op, err_choice[idx])
                        idx += 1
                    else:
                        op = np.kron(op, Id)
                for i in range(K):
                    for j in range(i + 1, K):
                        val = abs(np.vdot(code_states[i], op @ code_states[j]))
                        worst = max(worst, val)
        if worst > threshold:
            return w
    return max_weight + 1


def compute_weight_enumerators(code_states, n_qudit, d, single_error_basis, max_weight=2):
    """
    Shor-Laflamme weight enumerators A_j, B_j.

    A_j = (1/K²) Σ_{E weight j} |Tr(E·P_code)|²
    B_j = (1/K)  Σ_{E weight j} Tr(E·P_code·E†·P_code)

    Args:
        code_states: (K, dim) array
        n_qudit: number of qudits
        d: qudit dimension
        single_error_basis: list of d×d non-identity single-qudit errors
        max_weight: compute up to this weight (0, 1, 2, ...)

    Returns:
        (A, B): arrays of length max_weight+1
    """
    K, dim = code_states.shape
    P_c = code_states.T @ np.conj(code_states)  # (dim, dim) projector
    Id = np.eye(d, dtype=complex)

    A = np.zeros(max_weight + 1)
    B = np.zeros(max_weight + 1)

    # Weight 0
    A[0] = np.abs(np.trace(P_c)) ** 2 / K ** 2
    B[0] = np.real(np.trace(P_c @ P_c)) / K

    for w in range(1, max_weight + 1):
        for qudit_subset in itertools.combinations(range(n_qudit), w):
            for err_choice in itertools.product(single_error_basis, repeat=w):
                op = np.array([[1.0]], dtype=complex)
                idx = 0
                for q in range(n_qudit):
                    if q in qudit_subset:
                        op = np.kron(op, err_choice[idx])
                        idx += 1
                    else:
                        op = np.kron(op, Id)
                A[w] += np.abs(np.trace(op @ P_c)) ** 2
                B[w] += np.real(np.trace(op @ P_c @ op.conj().T @ P_c))
        A[w] /= K ** 2
        B[w] /= K

    return A, B


def compute_entanglement_entropy(code_states, bipartition_A, n_qudit, d):
    """
    Von Neumann entanglement entropy S(rho_A) for each codeword across a bipartition.

    Args:
        code_states: (K, dim) array
        bipartition_A: list of qudit indices in subsystem A
        n_qudit: total number of qudits
        d: qudit dimension

    Returns:
        list of K entropy values
    """
    K = code_states.shape[0]
    n_A = len(bipartition_A)
    n_B = n_qudit - n_A
    d_A = d ** n_A
    d_B = d ** n_B

    bipartition_B = [q for q in range(n_qudit) if q not in bipartition_A]
    perm = list(bipartition_A) + bipartition_B

    entropies = []
    for k in range(K):
        psi = code_states[k].reshape([d] * n_qudit)
        psi = np.transpose(psi, perm)
        psi = psi.reshape(d_A, d_B)
        rho_A = psi @ psi.conj().T
        eigenvalues = np.linalg.eigvalsh(rho_A)
        eigenvalues = eigenvalues[eigenvalues > 1e-15]
        S = -np.sum(eigenvalues * np.log2(eigenvalues))
        entropies.append(float(S))

    return entropies
