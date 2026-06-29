"""Tight-binding Hamiltonian construction for multilayer graphene systems."""

import numpy as np
from numpy import conj

from constants import (
    ABA_interaction,
    ABC_interaction,
    ACA_interaction,
    ACB_interaction,
    a,
    delta,
    g0,
    g1,
    g3,
    g4,
)
from validation import complex_array, real_array


__all__ = [
    "Hamiltonian",
    "energies",
    "single_hamiltonian_energies",
]


def translate_abc_to_012arr(layers):
    """Validate layer notation and convert strings to integer arrays.

    Args:
        layers (str or iterable): Layer formation (e.g., "ABCACB" or [0, 1, 2])

    Returns:
        list: Integer array where A=0, B=1, C=2

    Raises:
        ValueError: If layers are invalid or contain AA stacking
    """
    if isinstance(layers, str):
        if not all(char in "abc" for char in layers.lower()):
            raise ValueError("Layer formation should contain only A, B, C.")
        layers = [int(char) for char in layers.lower().translate(str.maketrans({"a": "0", "b": "1", "c": "2"}))]
    else:
        try:
            layers = list(layers)
        except TypeError as exc:
            raise ValueError("Layer formation should contain only 0, 1, 2.") from exc
    if not layers or not all(
        not isinstance(layer, (bool, np.bool_))
        and isinstance(layer, (int, np.integer))
        and layer in (0, 1, 2)
        for layer in layers
    ):
        raise ValueError("Layer formation should contain only 0, 1, 2.")
    layers = [int(layer) for layer in layers]
    if any(layers[i] == layers[i + 1] for i in range(len(layers) - 1)):
        raise ValueError("AA stacking detected.")
    return layers


def _structure_factor(k):
    return _structure_factors(np.asarray(k, dtype=float)[None, :])[0]


def _structure_factors(k_points):
    with np.errstate(over="ignore", invalid="ignore"):
        values = np.exp(-1j * k_points[:, 0] * a) + 2.0 * np.exp(
            1j * k_points[:, 0] * a / 2.0
        ) * np.cos(k_points[:, 1] * np.sqrt(3) * a / 2)
    if not np.all(np.isfinite(values)):
        raise FloatingPointError(
            "The momentum is finite but outside the numerically representable "
            "range of the structure factor."
        )
    return values


def f(k: np.ndarray) -> complex:
    """Return the graphene structure factor at one validated momentum."""
    return _structure_factor(real_array("k", k, shape=(2,)))


def _finite_structure_factor(f_k):
    return complex_array("f_k", f_k, shape=()).item()


def _self_interaction(f_k):
    m = np.zeros((2, 2), dtype="complex128")
    m[0, 1] = g0 * f_k
    return m


def self_int(f_k):
    """Return the intralayer interaction block for a finite structure factor."""
    return _self_interaction(_finite_structure_factor(f_k))


def _ab_interaction(f_k):
    m = np.zeros((2, 2), dtype="complex128")
    m[0, 0] = g4 * f_k
    m[0, 1] = g3 * conj(f_k)
    m[1, 0] = g1
    m[1, 1] = g4 * f_k
    return m


def AB_int(f_k):
    """Return the A-to-B interlayer interaction block."""
    return _ab_interaction(_finite_structure_factor(f_k))


def _ac_interaction(f_k):
    return conj(_ab_interaction(f_k).T)


def AC_int(f_k):
    """Return the A-to-C interlayer interaction block."""
    return _ac_interaction(_finite_structure_factor(f_k))


def _hamiltonian(layers, k):
    return _hamiltonian_batch(layers, np.asarray(k, dtype=float)[None, :])[0]


def _hamiltonian_batch(layers, k_points, output=None):
    """Build Hamiltonians directly into one reusable contiguous batch."""
    n_layers = len(layers)
    n_sites = 2 * n_layers
    k_points = np.asarray(k_points, dtype=float)
    f_k = _structure_factors(k_points)
    f_k_conjugate = np.conj(f_k)

    if output is None:
        hamiltonians = np.zeros(
            (len(k_points), n_sites, n_sites),
            dtype=np.complex128,
        )
    else:
        if output.shape[0] < len(k_points) or output.shape[1:] != (n_sites, n_sites):
            raise ValueError("output is too small for the Hamiltonian batch.")
        hamiltonians = output[:len(k_points)]
        hamiltonians.fill(0.0)

    eclipsed_sites = set()
    for layer, current_layer in enumerate(layers):
        row = 2 * layer
        hamiltonians[:, row, row + 1] = g0 * f_k
        hamiltonians[:, row + 1, row] = g0 * f_k_conjugate

        if layer >= n_layers - 1:
            continue

        column = row + 2
        direction = (layers[layer + 1] - current_layer) % 3
        if direction == 1:
            hamiltonians[:, row, column] = g4 * f_k
            hamiltonians[:, row, column + 1] = g3 * f_k_conjugate
            hamiltonians[:, row + 1, column] = g1
            hamiltonians[:, row + 1, column + 1] = g4 * f_k
            eclipsed_sites.update((row + 1, column))
        else:
            hamiltonians[:, row, column] = g4 * f_k_conjugate
            hamiltonians[:, row, column + 1] = g1
            hamiltonians[:, row + 1, column] = g3 * f_k
            hamiltonians[:, row + 1, column + 1] = g4 * f_k_conjugate
            eclipsed_sites.update((row, column + 1))

        upper_block = hamiltonians[:, row:row + 2, column:column + 2]
        hamiltonians[:, column:column + 2, row:row + 2] = np.conj(
            upper_block.swapaxes(1, 2)
        )

        if layer < n_layers - 2:
            pattern = [
                direction,
                (layers[layer + 2] - current_layer) % 3,
            ]
            match pattern:
                case [1, 2]:
                    interaction = ABC_interaction
                case [2, 1]:
                    interaction = ACB_interaction
                case [1, 0]:
                    interaction = ABA_interaction
                case [2, 0]:
                    interaction = ACA_interaction

            next_column = column + 2
            hamiltonians[:, row:row + 2, next_column:next_column + 2] = interaction
            hamiltonians[:, next_column:next_column + 2, row:row + 2] = np.conj(
                interaction.T
            )

    for site in eclipsed_sites:
        hamiltonians[:, site, site] = delta

    if not np.all(np.isfinite(hamiltonians)):
        raise FloatingPointError("Hamiltonian construction produced non-finite values.")
    return hamiltonians


def Hamiltonian(layers, k):
    """Construct the spatial tight-binding Hamiltonian for a layer stack.

    Args:
        layers (str or list): Layer structure (e.g., "abcacb" or [0,1,2,0,2,1])
        k (np.array): 2D k-vector [kx, ky]

    Returns:
        np.array: NxN spatial Tight-binding Hamiltonian matrix
    """
    return _hamiltonian(
        translate_abc_to_012arr(layers),
        real_array("k", k, shape=(2,)),
    )


def single_hamiltonian_energies(layers, k):
    """Calculate eigenvalues for a single Hamiltonian at given k-point.

    Args:
        layers (str or list): Layer structure
        k (np.array): 2D k-vector [kx, ky]

    Returns:
        np.array: Eigenvalues (energies)
    """
    return np.linalg.eigvalsh(Hamiltonian(layers, k))


def energies(layers, k_space):
    """Calculate eigenvalues for multiple k-points.

    Args:
        layers (str or list): Layer structure
        k_space (np.array): Array of 2D k-vectors

    Returns:
        np.array: Array of eigenvalues for each k-point
    """
    layers = translate_abc_to_012arr(layers)
    k_space = real_array("k_space", k_space, shape=(None, 2))
    return np.linalg.eigvalsh(_hamiltonian_batch(layers, k_space))
