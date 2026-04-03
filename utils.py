from typing import Dict, Tuple, Optional

import numpy as np


def denoise(
        noisy_data: np.ndarray,
        eigenvectors: np.ndarray,
        num_components: int,
        n: int,
) -> np.ndarray:
    """
    Denoise quantum states by projecting onto leading eigenvectors.

    Parameters
    ----------
    noisy_data : np.ndarray
        Array of noisy quantum states of shape (num_samples, 2**n).
    eigenvectors : np.ndarray
        Matrix of eigenvectors of shape (2**n, 2**n), assumed ordered
        by increasing eigenvalue (largest at the end).
    num_components : int
        Number of leading eigenvectors to retain for denoising.
    n : int
        Number of qubits (defines Hilbert space dimension 2**n).

    Returns
    -------
    np.ndarray
        Array of denoised quantum states with the same shape as `noisy_data`.

    Notes
    -----
    Each noisy state is projected onto the subspace spanned by the
    `num_components` largest eigenvectors.
    """
    dim = 2 ** n

    if noisy_data.shape[1] != dim:
        raise ValueError("Mismatch between 'n' and noisy_data dimension.")

    if eigenvectors.shape != (dim, dim):
        raise ValueError("eigenvectors must have shape (2**n, 2**n).")

    if num_components <= 0 or num_components > dim:
        raise ValueError("num_components must be in [1, 2**n].")

    quantum_states_denoised = np.zeros(np.shape(noisy_data), dtype=np.complex128)

    for i, quantum_state_n in enumerate(noisy_data):
        g_denoised = np.zeros(2 ** n, dtype=complex)
        for j in np.arange(1, num_components + 1):
            g_denoised += np.conj(eigenvectors[:, -j]).T @ quantum_state_n * eigenvectors[:, -j]
        quantum_states_denoised[i] = g_denoised

    return quantum_states_denoised


def gaussian_window_weights(
        size: int,
        sigma: Optional[float] = None,
) -> Dict[Tuple[int, int], float]:
    """
    Generate a normalized 2D Gaussian kernel on a discrete grid.

    Yes, the grid must be *odd*. Symmetry demands a center.

    Parameters
    ----------
    size : int
        Size of the kernel (must be odd).
    sigma : float, optional
        Standard deviation of the Gaussian. If None, defaults to size / 4.

    Returns
    -------
    dict[tuple[int, int], float]
        Dictionary mapping (x, z) coordinates to normalized weights.

    Notes
    -----
    The kernel is centered at (0, 0) and spans:
        x, z ∈ [-r, ..., r], where r = size // 2

    The weights are normalized such that:
        sum(weights.values()) == 1
    """
    if size % 2 != 1:
        raise ValueError("size must be odd to ensure a centered kernel.")

    r = size // 2
    if sigma is None:
        sigma = r / 2  # default choice

    weights = {}
    total = 0.0

    for x in range(-r, r + 1):
        for z in range(-r, r + 1):
            w = np.exp(-(x ** 2 + z ** 2) / (2 * sigma ** 2))
            weights[(x, z)] = w
            total += w

    # normalize
    for k in weights:
        weights[k] /= total

    return weights
