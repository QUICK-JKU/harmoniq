from typing import Tuple

import numpy as np

from gates import apply_WH
from utils import denoise, gaussian_window_weights


def run_algorithm(
        F: np.ndarray,
        F_n: np.ndarray,
        size: int,
        n: int,
        N,
        num_components: int,
) -> Tuple[float, float, float]:
    """
        Run denoising pipeline with and without Weyl–Heisenberg augmentation.

        Parameters
        ----------
        F : np.ndarray
            Clean dataset of shape (N, d).
        F_n : np.ndarray
            Noisy dataset of shape (N, d).
        size : int
            Window size for phase-space augmentation (must be odd).
        n : int
            Number of qubits (dimension d = 2**n).
        num_components : int
            Number of principal components for projection.

        Returns
        -------
        tuple of float
            Mean squared distances:
            (noisy, denoised, augmented_denoised)

        Notes
        -----
        - Data is centered and normalized row-wise.
        - Density matrices are constructed via weighted outer products.
        - Augmentation uses Weyl–Heisenberg shifts with Gaussian weights.
        """
    d = 2 ** n
    # Clean data
    F -= np.mean(F, axis=0)  # mean across each column
    norms = np.linalg.norm(F, axis=1)  # shape (N,)
    Q = F / norms[:, None]

    # Noisy data:
    F_n -= np.mean(F_n, axis=0)
    norms_n = np.linalg.norm(F_n, axis=1)  # shape (N,)
    norms_squared_n = norms_n ** 2
    norms_total_n = np.sum(norms_squared_n)
    probabilities_n = norms_squared_n / norms_total_n  # sum(p) = 1
    Q_n = F_n / norms_n[:, None]

    # Noisy data without augmentation
    rho_n = (probabilities_n * Q_n.T) @ np.conj(Q_n)
    eigvals_n, eigenvectors_n = np.linalg.eigh(rho_n)

    # Noisy data with augmentation
    rho_aug = np.zeros((d, d), dtype=np.complex128)

    weights = gaussian_window_weights(size)
    for (x, z), w in weights.items():
        Q_aug = apply_WH(d, x % d, z % d, Q_n)
        rho_aug += (probabilities_n * w * Q_aug.T) @ np.conj(Q_aug)
    eigvals_aug_n, eigenvectors_aug_n = np.linalg.eigh(rho_aug)

    # Project onto K components
    Q_denoised = denoise(Q_n, eigenvectors_n, num_components, n)
    Q_aug_denoised = denoise(Q_n, eigenvectors_aug_n, num_components, n)

    distance_noisy = np.sum(np.linalg.norm(Q - Q_n, axis=1) ** 2)
    distance_denoised = np.sum(np.linalg.norm(Q - Q_denoised, axis=1) ** 2)
    distance_aug_denoised = np.sum(np.linalg.norm(Q - Q_aug_denoised, axis=1) ** 2)

    return distance_noisy / N, distance_denoised / N, distance_aug_denoised / N
