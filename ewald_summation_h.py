"""Ewald summation implementation for long-range electrostatic interactions."""

from functools import lru_cache

import numpy as np
from scipy.special import erf, erfc, erfcx

from constants import (
    A,
    DIELECTRIC_PRESETS,
    LAYER_SHIFTS,
    a,
    a1,
    a2,
    b1,
    b2,
    c,
    e,
    epsilon_0,
    hubbard_u_for_layer_count,
)
from tight_binding_h import translate_abc_to_012arr
from validation import real, real_array


__all__ = [
    "clear_ewald_cache",
    "ewald_matrix_calc",
    "get_ewald_cache_stats",
    "get_unit_cell_positions",
    "resolve_dielectric",
]


def resolve_dielectric(dielectric="hbn"):
    """Resolve a homogeneous dielectric preset to (epsilon_parallel, epsilon_perp)."""
    if not isinstance(dielectric, str):
        raise ValueError("dielectric must be the name of a supported preset.")
    key = dielectric.lower().replace("-", "_")
    if key not in DIELECTRIC_PRESETS:
        valid = ", ".join(sorted(DIELECTRIC_PRESETS))
        raise ValueError(f"Unknown dielectric preset '{dielectric}'. Expected one of: {valid}.")

    epsilon_parallel, epsilon_perp = (float(value) for value in DIELECTRIC_PRESETS[key])
    if not np.isfinite(epsilon_parallel) or not np.isfinite(epsilon_perp):
        raise ValueError("Dielectric constants must be finite.")
    if epsilon_parallel <= 0.0 or epsilon_perp <= 0.0:
        raise ValueError("Dielectric constants must be positive.")
    return epsilon_parallel, epsilon_perp


OHNO_CUTOFF_POLICY = (
    (0.05, 35.0),
    (0.02, 50.0),
    (0.01, 75.0),
    (0.005, 95.0),
)

_EWALD_CACHE_SIZE = 16


def lattice_vectors_within_cutoff(v1, v2, cutoff):
    """Return all integer combinations n1*v1 + n2*v2 inside a circular cutoff."""
    cutoff = real("cutoff", cutoff, minimum=0.0)
    v1 = real_array("v1", v1, shape=(2,))
    v2 = real_array("v2", v2, shape=(2,))
    return _lattice_vectors_within_cutoff(v1, v2, cutoff)


def _lattice_vectors_within_cutoff(v1, v2, cutoff):
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    lattice_matrix = np.column_stack((v1, v2))
    min_singular = float(np.min(np.linalg.svd(lattice_matrix, compute_uv=False)))
    if min_singular <= 0.0:
        raise ValueError("lattice vectors must be linearly independent.")

    n_max = int(np.ceil(cutoff / min_singular)) + 1
    vectors = []
    for n1 in range(-n_max, n_max + 1):
        for n2 in range(-n_max, n_max + 1):
            vector = n1 * v1 + n2 * v2
            if np.linalg.norm(vector) <= cutoff:
                vectors.append(vector)
    return vectors


def ohno_cutoff_from_target(ohno_target_meV=0.05):
    """Return the validated Ohno real-space cutoff for a supported target error."""
    target = real("ohno_target_meV", ohno_target_meV, minimum=0.0, strict_minimum=True)

    for supported_target, cutoff in OHNO_CUTOFF_POLICY:
        if np.isclose(target, supported_target, rtol=0.0, atol=1e-12):
            return cutoff

    valid_targets = ", ".join(f"{value:g}" for value, _ in OHNO_CUTOFF_POLICY)
    raise ValueError(f"Unsupported ohno_target_meV. Choose one of: {valid_targets}.")


def get_unit_cell_positions(layers):
    """Get atomic positions in the unit cell for given layer stacking.

    Args:
        layers (list): List of integers representing layer types (0=A, 1=B, 2=C)

    Returns:
        np.array: Array of atomic positions [x, y, z] for each atom
    """
    layers = translate_abc_to_012arr(layers)
    q_pos = []
    for i, layer in enumerate(layers):
        shift = LAYER_SHIFTS[layer]
        z = i * c
        q_pos += [
            np.array([shift[0], shift[1], z], dtype=np.longdouble),
            np.array([shift[0] + a, shift[1], z], dtype=np.longdouble),
        ]
    return np.array(q_pos)


def _real_sum(eta, rho_bar, z_bar, real_vectors, cutoff):
    """Calculate the real space sum in Ewald summation.

    Args:
        eta (float): Ewald parameter
        rho_bar (np.array): In-plane distance vector
        z_bar (float): Out-of-plane distance
        real_vectors (list): List of precomputed real-space lattice vectors [x, y]
        cutoff (float): Three-dimensional real-space cutoff

    Returns:
        float: Real space sum contribution
    """
    displacements = rho_bar - real_vectors
    distance_squared = (
        np.einsum("ij,ij->i", displacements, displacements) + z_bar**2
    )
    distances = np.sqrt(distance_squared[distance_squared <= cutoff**2])
    nonzero = distances != 0.0
    return float(
        np.sum(erfc(distances[nonzero] / (2 * eta)) / distances[nonzero])
        - np.count_nonzero(~nonzero) / (np.sqrt(np.pi) * eta)
    )


def _reciprocal_sum(eta, rho_bar, z_bar, recip_vectors, recip_norms):
    """Calculate the reciprocal space sum in Ewald summation.

    Args:
        eta (float): Ewald parameter
        rho_bar (np.array): In-plane distance vector
        z_bar (float): Out-of-plane distance
        recip_vectors (np.array): Precomputed reciprocal lattice vectors
        recip_norms (np.array): Norm of each reciprocal lattice vector

    Returns:
        float: Reciprocal space sum contribution
    """
    z_abs = abs(z_bar)
    z_scaled = z_abs / (2 * eta)
    k_eta = recip_norms * eta
    phases = np.cos(recip_vectors @ rho_bar)

    # Stable form of exp(k*z) * erfc(k*eta + z/(2*eta)).
    # The direct product can overflow to inf*0 for thick stacks even when
    # the mathematical value is finite.
    growing_terms = (
        np.exp(-(k_eta**2) - z_scaled**2) * erfcx(k_eta + z_scaled)
    )
    decaying_terms = np.exp(-recip_norms * z_abs) * erfc(k_eta - z_scaled)
    return float(
        np.sum(phases * (growing_terms + decaying_terms) / recip_norms)
    )


def _last_term(eta, z_bar):
    """Calculate the last term in Ewald summation.

    Args:
        eta (float): Ewald parameter
        z_bar (float): Out-of-plane distance

    Returns:
        float: Last term contribution
    """
    return z_bar * erf(z_bar / (2 * eta)) + 2 * eta * np.exp(
        -(z_bar**2) / (4 * eta**2)
    ) / np.sqrt(np.pi)


def _ewald_potential(
    eta,
    rho_bar,
    z_bar,
    real_vectors,
    recip_vectors,
    recip_norms,
    real_cutoff,
):
    """Calculate the Ewald potential V_bar.

    Args:
        eta (float): Ewald parameter
        rho_bar (np.array): In-plane distance vector
        z_bar (float): Out-of-plane distance
        real_vectors (list): Precomputed real space vectors
        recip_vectors (np.array): Precomputed reciprocal space vectors
        recip_norms (np.array): Norm of each reciprocal space vector
        real_cutoff (float): Three-dimensional real-space cutoff

    Returns:
        float: Ewald potential
    """
    return (
        _real_sum(eta, rho_bar, z_bar, real_vectors, real_cutoff)
        + np.pi
        * _reciprocal_sum(
            eta,
            rho_bar,
            z_bar,
            recip_vectors,
            recip_norms,
        )
        / A
        - 2 * np.pi * _last_term(eta, z_bar) / A
    )


def _ohno_correction(
    rho_bar,
    z_bar,
    real_vectors,
    cutoff,
    coulomb_prefactor,
    ohno_length,
):
    """Return the finite-range Ohno correction and its continuum tail."""
    displacements = rho_bar - real_vectors
    rho_squared = np.einsum("ij,ij->i", displacements, displacements)
    distances = np.sqrt(rho_squared[rho_squared <= cutoff**2] + z_bar**2)
    distances = distances[distances > 1e-8]

    correction = np.sum(
        coulomb_prefactor / np.sqrt(ohno_length**2 + distances**2)
        - coulomb_prefactor / distances
    )
    tail = (2 * np.pi * coulomb_prefactor / A) * (
        np.sqrt(cutoff**2 + z_bar**2)
        - np.sqrt(cutoff**2 + z_bar**2 + ohno_length**2)
    )
    return float(correction + tail)


def _validate_ewald_inputs(
    layers,
    eta,
    tolerance,
    dielectric,
    ohno_target_meV,
):
    layers = translate_abc_to_012arr(layers)
    eta = real(
        "eta",
        0.4 * np.sqrt(A) if eta is None else eta,
        minimum=0.0,
        strict_minimum=True,
    )
    tolerance = real(
        "tolerance",
        tolerance,
        minimum=0.0,
        maximum=1.0,
        strict_minimum=True,
        strict_maximum=True,
    )
    epsilon_parallel, epsilon_perp = resolve_dielectric(dielectric)
    ohno_cutoff = ohno_cutoff_from_target(ohno_target_meV)
    hubbard_u = hubbard_u_for_layer_count(len(layers))
    return (
        layers,
        eta,
        tolerance,
        epsilon_parallel,
        epsilon_perp,
        ohno_cutoff,
        hubbard_u,
    )


def ewald_matrix_calc(
    layers,
    eta=None,
    tolerance=1e-8,
    dielectric="hbn",
    ohno_target_meV=0.05,
):
    """Calculate the Ewald summation matrix for electrostatic interactions.

    Cached matrices are keyed by the physical and numerical parameters that
    affect the result.

    Args:
        layers (str or iterable): Layer stacking in A/B/C or 0/1/2 notation.
        eta (float, optional): Positive Ewald parameter. Defaults to 0.4 * sqrt(A).
        tolerance (float, optional): Adaptive-cutoff target strictly between 0 and 1.
        dielectric (str): Background dielectric preset: 'hbn', 'vacuum', or 'graphite'.
        ohno_target_meV (float): Supported Ohno matrix-error target in meV.
            Choices are 0.05, 0.02, 0.01, and 0.005 meV.
    Returns:
        np.array: Ewald matrix representing electrostatic potential in Volts.
                  (When multiplied by a dimensionless electron count in the
                  SCF loop, this directly yields Potential Energy in eV).
    """
    (
        layers,
        eta,
        tolerance,
        epsilon_parallel,
        epsilon_perp,
        R_cut_ohno,
        hubbard_u,
    ) = _validate_ewald_inputs(
        layers,
        eta,
        tolerance,
        dielectric,
        ohno_target_meV,
    )
    return _cached_ewald_matrix(
        tuple(layers),
        eta,
        tolerance,
        epsilon_parallel,
        epsilon_perp,
        R_cut_ohno,
        hubbard_u,
    ).copy()


@lru_cache(maxsize=_EWALD_CACHE_SIZE)
def _cached_ewald_matrix(
    layers,
    eta,
    tolerance,
    epsilon_parallel,
    epsilon_perp,
    R_cut_ohno,
    hubbard_u,
):
    """Build and privately cache one validated Ewald matrix."""
    z_scale = np.sqrt(epsilon_parallel / epsilon_perp)
    epsilon_effective = np.sqrt(epsilon_parallel * epsilon_perp)

    # --- Adaptive Cutoffs ---
    # Calculate cutoff radii based on target tolerance
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        cutoff_scale = np.sqrt(-np.log(tolerance))
        R_cut = 2 * eta * cutoff_scale
        K_cut = cutoff_scale / eta
    if not np.isfinite(R_cut) or not np.isfinite(K_cut):
        raise ValueError(
            "eta and tolerance produce non-finite Ewald cutoffs; "
            "use a less extreme eta."
        )

    q_pos = get_unit_cell_positions(layers)
    rho_pos = q_pos[:, :2]
    z_pos = q_pos[:, 2]
    pair_displacements = rho_pos[:, None, :] - rho_pos[None, :, :]
    max_pair_displacement = float(
        np.max(np.linalg.norm(pair_displacements, axis=2))
    )

    # Enumerate a safe superset once; pair loops apply the physical cutoff.
    real_vectors = np.asarray(
        _lattice_vectors_within_cutoff(
            a1[:2],
            a2[:2],
            R_cut + max_pair_displacement,
        ),
        dtype=float,
    )

    # Precompute reciprocal space lattice vectors within K_cut
    recip_vectors = np.asarray(
        _lattice_vectors_within_cutoff(b1[:2], b2[:2], K_cut),
        dtype=float,
    ).reshape(-1, 2)
    recip_norms = np.linalg.norm(recip_vectors, axis=1)
    nonzero = recip_norms != 0.0
    recip_vectors = recip_vectors[nonzero]
    recip_norms = recip_norms[nonzero]

    # Assign zero matrix for the V_bar values
    V_bar_lower = np.zeros((len(q_pos), len(q_pos)))

    # Loop over all unit cell charge positions
    for i in range(len(rho_pos)):
        for j in range(i + 1):
            # Calculate the appropriate V_bar for each interaction with other unit cell charges
            V_bar_lower[i, j] = _ewald_potential(
                eta,
                rho_pos[i] - rho_pos[j],
                z_scale * (z_pos[i] - z_pos[j]),
                real_vectors,
                recip_vectors,
                recip_norms,
                R_cut,
            )

    # Use the symmetry of the matrix to get the full V_bar_matrix
    V_bar_matrix = V_bar_lower + V_bar_lower.T - np.diag(V_bar_lower.diagonal())

    # Homogeneous uniaxial dielectric background:
    # epsilon = diag(epsilon_parallel, epsilon_parallel, epsilon_perp).
    # The z coordinates above are scaled by sqrt(epsilon_parallel / epsilon_perp),
    # and the Coulomb prefactor is screened by sqrt(epsilon_parallel * epsilon_perp).
    V_bar_matrix = (V_bar_matrix / (4 * np.pi * epsilon_0 * epsilon_effective)) * e * 1e10

    # The Ohno correction decays algebraically (~1/r^3), so its cutoff is
    # independent of the exponentially convergent Ewald real-space sum.
    ohno_vectors = np.asarray(
        _lattice_vectors_within_cutoff(
            a1[:2],
            a2[:2],
            R_cut_ohno + max_pair_displacement,
        ),
        dtype=float,
    )
    k_e = (1 / (4 * np.pi * epsilon_0 * epsilon_effective)) * e * 1e10
    a_ohno = k_e / hubbard_u

    ohno_lower = np.zeros_like(V_bar_matrix)
    for i in range(len(q_pos)):
        for j in range(i + 1):
            z_ij = z_scale * (q_pos[i][2] - q_pos[j][2])
            ohno_lower[i, j] = _ohno_correction(
                rho_pos[i] - rho_pos[j],
                z_ij,
                ohno_vectors,
                R_cut_ohno,
                k_e,
                a_ohno,
            )

    V_bar_matrix += (
        ohno_lower
        + ohno_lower.T
        - np.diag(ohno_lower.diagonal())
    )

    # Hubbard U is not part of this spatial interaction matrix; the SCF loop
    # applies the local paramagnetic Hubbard correction.
    if not np.all(np.isfinite(V_bar_matrix)):
        raise FloatingPointError("Ewald calculation produced non-finite matrix values.")

    V_bar_matrix.setflags(write=False)
    return V_bar_matrix


def get_ewald_cache_stats():
    """Return cache hits, misses, hit rate, and stored-matrix count."""
    info = _cached_ewald_matrix.cache_info()
    total = info.hits + info.misses
    return {
        "hits": info.hits,
        "misses": info.misses,
        "total_requests": total,
        "hit_rate": info.hits / total if total else 0.0,
        "cache_size": info.currsize,
    }


def clear_ewald_cache():
    """Clear the Ewald matrix cache and reset statistics."""
    _cached_ewald_matrix.cache_clear()
