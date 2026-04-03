import numpy as np

def apply_WH(
    d: int,
    x: int,
    z: int,
    psi: np.ndarray,
) -> np.ndarray:
    """
    Apply a Weyl–Heisenberg (shift + phase) operator to a quantum state.

    In discrete phase space, this is the operator:
        W(x, z) = ω^(xz) Z^z X^x

    where:
        ω = exp(2πi / d)

    Parameters
    ----------
    d : int
        Dimension of the Hilbert space.
    x : int
        Shift in computational basis (X operator exponent).
    z : int
        Phase exponent (Z operator exponent).
    psi : np.ndarray
        Quantum state(s), shape (..., d). The operation is applied along the last axis.

    Returns
    -------
    np.ndarray
        Transformed quantum state(s), same shape as `psi`.

    Notes
    -----
    - Uses periodic boundary conditions via np.roll.
    - Works on batches of states if psi has more than one dimension.
    """
    if psi.shape[-1] != d:
        raise ValueError("Last dimension of psi must match d.")

    # Root of unity
    omega = np.exp(2j * np.pi / d)

    # Computational basis indices
    k = np.arange(d)

    omega_k = omega ** k  # store ω^j

    shifted = np.roll(psi, x, axis=1)
    return np.e ** (2 * np.pi * 1j * x * z / d) * (omega_k ** z) * shifted