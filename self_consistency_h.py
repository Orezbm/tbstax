"""Self-consistent field calculations for spin-degenerate tight binding."""

from typing import NamedTuple

import numpy as np
from scipy.special import expit

from constants import A, e, hubbard_u_for_layer_count, kb
from ewald_summation_h import (
    ewald_matrix_calc,
    get_unit_cell_positions,
    ohno_cutoff_from_target,
    resolve_dielectric,
)
from k_points import monkhorst_pack_mesh
from tight_binding_h import _hamiltonian_batch, translate_abc_to_012arr
from validation import (
    boolean,
    complex_array,
    integer,
    normalized_weights,
    real,
    real_array,
)


SPIN_DEGENERACY = 2.0
# Hamiltonian, eigenvectors, density workspace, and eigensolver headroom.
_SCF_WORKING_SET_BYTES = 512 * 1024**2
_EIGENSOLVE_BYTES_PER_MATRIX_ELEMENT = 64


__all__ = [
    "DensityObservables",
    "GridConvergence",
    "SCFResult",
    "converged_on_site_potentials_from_layers",
    "fermi_dirac",
    "on_site_potentials_from_layers",
    "total_electron_count_from_eigenvalues",
]


class DensityObservables(NamedTuple):
    """Density-derived observables for one graphene unit cell."""

    out_of_plane_polarization_C_per_m: float
    electrostatic_energy_eV: float


class GridConvergence(NamedTuple):
    """Maximum changes across the accepted k-grid confirmation window."""

    potential_eV: float
    chemical_potential_eV: float | None = None
    density_electrons_per_site: float | None = None
    electrostatic_energy_eV: float | None = None


class SCFResult(NamedTuple):
    """Result and convergence diagnostics for one SCF calculation."""

    potentials: np.ndarray
    iterations: int
    converged: bool
    residual_eV: float
    electron_error: float
    chemical_potential_eV: float | None
    charge_density: np.ndarray | None
    density_observables: DensityObservables | None
    residual_history_eV: tuple[float, ...] | None


def _fermi_dirac(energy, mu, T):
    if T == 0.0:
        return np.where(energy < mu, 1.0, np.where(energy > mu, 0.0, 0.5))
    with np.errstate(over="ignore", invalid="ignore"):
        scaled_energy = -(energy - mu) / (kb * T)
    return expit(scaled_energy)


def fermi_dirac(energy, mu, T):
    """Fermi-Dirac distribution function.

    Args:
        energy (float or np.array): Energy values in eV
        mu (float): Chemical potential in eV
        T (float): Temperature in Kelvin

    Returns:
        float or np.array: Fermi-Dirac occupation numbers
    """
    energy = real_array("energy", energy, nonempty=False)
    mu = real("mu", mu)
    T = real("Temperature", T, minimum=0.0)
    occupations = _fermi_dirac(energy, mu, T)
    if not np.all(np.isfinite(occupations)):
        raise FloatingPointError("Fermi-Dirac calculation produced non-finite values.")
    return occupations


def _validated_hamiltonians(ham_list):
    hamiltonians = complex_array("Hamiltonians", ham_list, ndim=3)
    if hamiltonians.shape[1] != hamiltonians.shape[2]:
        raise ValueError("Hamiltonians must be square matrices of equal size.")
    if hamiltonians.shape[1] == 0:
        raise ValueError("Hamiltonians must contain at least one site.")
    return hamiltonians


def _total_electron_count(eigenvalues_list, mu, T, weights):
    occupations = _fermi_dirac(eigenvalues_list, mu, T)
    return float(SPIN_DEGENERACY * np.sum(weights[:, None] * occupations))


def total_electron_count_from_eigenvalues(eigenvalues_list, mu, T, weights):
    eigenvalues_list = real_array("eigenvalues_list", eigenvalues_list, ndim=2)
    weights = normalized_weights(weights, len(eigenvalues_list))
    mu = real("mu", mu)
    T = real("Temperature", T, minimum=0.0)
    return _total_electron_count(eigenvalues_list, mu, T, weights)


def _solve_mu_from_eigenvalues(
    eigenvalues_list,
    T,
    target_electrons,
    weights,
    mu_min=None,
    mu_max=None,
    electron_tolerance=1e-6,
    max_iter=100,
):
    """Find the finite-temperature chemical potential at fixed electron count."""
    T = real("Temperature", T, minimum=0.0)
    if T == 0.0:
        raise ValueError("Use exact zero-temperature filling when T is zero.")
    eigenvalues_list = real_array("eigenvalues_list", eigenvalues_list, ndim=2)
    weights = normalized_weights(weights, len(eigenvalues_list))
    target_electrons = real("target_electrons", target_electrons)
    electron_tolerance = real(
        "electron_tolerance",
        electron_tolerance,
        minimum=0.0,
        strict_minimum=True,
    )
    max_iter = integer("max_iter", max_iter, minimum=1)

    total_capacity = SPIN_DEGENERACY * len(eigenvalues_list[0])
    if not 0.0 < target_electrons < total_capacity:
        raise ValueError(
            f"target_electrons must be strictly between 0 and {total_capacity}."
        )

    low = (
        float(np.min(eigenvalues_list) - 10.0)
        if mu_min is None
        else real("mu_min", mu_min)
    )
    high = (
        float(np.max(eigenvalues_list) + 10.0)
        if mu_max is None
        else real("mu_max", mu_max)
    )
    if not np.isfinite(low) or not np.isfinite(high):
        raise FloatingPointError(
            "The eigenvalue range cannot be bracketed with finite chemical potentials."
        )
    if low >= high:
        raise ValueError("mu_min must be smaller than mu_max.")

    return _solve_mu_from_validated_eigenvalues(
        eigenvalues_list,
        T,
        target_electrons,
        weights,
        low,
        high,
        electron_tolerance,
        max_iter,
    )


def _solve_mu_from_validated_eigenvalues(
    eigenvalues_list,
    T,
    target_electrons,
    weights,
    low,
    high,
    electron_tolerance,
    max_iter,
):
    def f(mu):
        return _total_electron_count(eigenvalues_list, mu, T, weights) - target_electrons

    f_low = f(low)
    f_high = f(high)
    for _ in range(20):
        if f_low == 0.0:
            return low
        if f_high == 0.0:
            return high
        if f_low * f_high < 0.0:
            break
        span = (high - low) * 2.0
        low -= span / 2.0
        high += span / 2.0
        if not np.isfinite(low) or not np.isfinite(high):
            raise FloatingPointError(
                "Chemical-potential bracketing exceeded floating-point range."
            )
        f_low = f(low)
        f_high = f(high)

    if f_low * f_high > 0.0:
        raise ValueError("Failed to bracket chemical potential for target electron count.")

    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        f_mid = f(mid)
        if abs(f_mid) <= electron_tolerance:
            return mid
        if mid == low or mid == high:
            raise RuntimeError(
                "Finite-temperature filling is below floating-point resolution; "
                "use T=0 or a larger temperature."
            )
        if f_low * f_mid <= 0.0:
            high = mid
            f_high = f_mid
        else:
            low = mid
            f_low = f_mid

    mid = 0.5 * (low + high)
    error = abs(f(mid))
    raise RuntimeError(
        "Chemical-potential solve did not reach the requested electron count: "
        f"error={error:.3e}, tolerance={electron_tolerance:.3e}."
    )


def _zero_temperature_occupations(eigenvalues_list, target_electrons, weights):
    """Fill exactly, sharing a fractional frontier uniformly across degeneracies."""
    eigenvalues = np.asarray(eigenvalues_list, dtype=float)
    flat_energies = eigenvalues.ravel()
    state_capacities = np.repeat(
        SPIN_DEGENERACY * weights,
        eigenvalues.shape[1],
    )
    total_capacity = float(np.sum(state_capacities))
    if target_electrons <= 0.0 or target_electrons >= total_capacity:
        raise ValueError("target_electrons is outside the available state capacity.")

    order = np.argsort(flat_energies, kind="stable")
    sorted_energies = flat_energies[order]
    energy_scale = max(1.0, float(np.max(np.abs(sorted_energies))))
    degeneracy_tolerance = 64.0 * np.finfo(float).eps * energy_scale
    count_tolerance = 16.0 * np.finfo(float).eps * max(1.0, total_capacity)

    occupations = np.zeros_like(flat_energies)
    remaining = float(target_electrons)
    fractional_frontier = None

    group_start = 0
    while group_start < len(order) and remaining > count_tolerance:
        group_end = group_start + 1
        while (
            group_end < len(order)
            and abs(sorted_energies[group_end] - sorted_energies[group_start])
            <= degeneracy_tolerance
        ):
            group_end += 1

        group_indices = order[group_start:group_end]
        group_capacity = float(np.sum(state_capacities[group_indices]))
        fraction = 1.0 if remaining >= group_capacity - count_tolerance else remaining / group_capacity
        occupations[group_indices] = fraction

        remaining -= fraction * group_capacity
        if fraction < 1.0:
            fractional_frontier = float(sorted_energies[group_start])
            break
        group_start = group_end

    if abs(remaining) > count_tolerance:
        raise RuntimeError("Zero-temperature filling did not reach the target electron count.")

    if fractional_frontier is not None:
        mu = fractional_frontier
    else:
        highest_occupied = np.max(flat_energies[occupations > 0.0])
        lowest_unoccupied = np.min(flat_energies[occupations < 1.0])
        mu = 0.5 * (highest_occupied + lowest_unoccupied)

    return mu, occupations.reshape(eigenvalues.shape)


def _solve_mu_and_charge_density(
    ham_list,
    T,
    target_electrons,
    weights=None,
    electron_tolerance=1e-6,
    max_iter=100,
):
    """Diagonalize once and return the fixed-filling chemical potential and density."""
    hamiltonians = _validated_hamiltonians(ham_list)
    T = real("Temperature", T, minimum=0.0)
    target_electrons = real("target_electrons", target_electrons)
    weights = normalized_weights(weights, len(hamiltonians))
    electron_tolerance = real(
        "electron_tolerance",
        electron_tolerance,
        minimum=0.0,
        strict_minimum=True,
    )
    max_iter = integer("max_iter", max_iter, minimum=1)

    total_capacity = SPIN_DEGENERACY * hamiltonians.shape[1]
    if not 0.0 < target_electrons < total_capacity:
        raise ValueError(
            f"target_electrons must be strictly between 0 and {total_capacity}."
        )
    return _solve_mu_and_charge_density_validated(
        hamiltonians,
        T,
        target_electrons,
        weights,
        electron_tolerance,
        max_iter,
    )


def _solve_mu_and_charge_density_validated(
    hamiltonians,
    T,
    target_electrons,
    weights,
    electron_tolerance,
    max_iter,
):
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonians)

    mu, occupations_list = _occupations_from_eigenvalues_validated(
        eigenvalues,
        T,
        target_electrons,
        weights,
        electron_tolerance,
        max_iter,
    )

    charge_density = SPIN_DEGENERACY * np.einsum(
        "k,kib,kb->i",
        weights,
        np.abs(eigenvectors) ** 2,
        occupations_list,
        optimize=True,
    )
    _validate_charge_density(charge_density, target_electrons, electron_tolerance)
    return mu, charge_density


def _occupations_from_eigenvalues_validated(
    eigenvalues,
    T,
    target_electrons,
    weights,
    electron_tolerance,
    max_iter,
):
    """Return fixed-filling chemical potential and occupations."""

    if T == 0.0:
        mu, occupations_list = _zero_temperature_occupations(
            eigenvalues,
            target_electrons,
            weights,
        )
    else:
        low = float(np.min(eigenvalues) - 10.0)
        high = float(np.max(eigenvalues) + 10.0)
        if not np.isfinite(low) or not np.isfinite(high):
            raise FloatingPointError(
                "The eigenvalue range cannot be bracketed with finite chemical potentials."
            )
        mu = _solve_mu_from_validated_eigenvalues(
            eigenvalues,
            T,
            target_electrons,
            weights,
            low,
            high,
            electron_tolerance,
            max_iter,
        )
        occupations_list = _fermi_dirac(eigenvalues, mu, T)
    return mu, occupations_list


def _validate_charge_density(charge_density, target_electrons, electron_tolerance):
    """Validate finiteness and fixed filling of an accumulated density."""
    if not np.all(np.isfinite(charge_density)):
        raise FloatingPointError("Charge-density calculation produced non-finite values.")
    electron_error = abs(float(np.sum(charge_density)) - target_electrons)
    if electron_error > electron_tolerance:
        raise RuntimeError(
            "Charge density does not match the requested electron count: "
            f"error={electron_error:.3e}, tolerance={electron_tolerance:.3e}."
        )


def _k_point_batch_size(k_count, n_sites):
    """Return a batch size bounded by the SCF eigensolve memory policy."""
    bytes_per_k = _EIGENSOLVE_BYTES_PER_MATRIX_ELEMENT * n_sites**2
    return min(k_count, max(1, _SCF_WORKING_SET_BYTES // bytes_per_k))


def _build_hamiltonian_batch(
    layers,
    k_points,
    on_site_potentials=None,
    output=None,
):
    """Build one contiguous batch, optionally adding its diagonal potential."""
    n_sites = 2 * len(layers)
    hamiltonians = _hamiltonian_batch(
        layers,
        k_points,
        output=output,
    )

    if on_site_potentials is not None:
        diagonal = np.arange(n_sites)
        hamiltonians[:, diagonal, diagonal] += on_site_potentials
    return hamiltonians


def _solve_k_point_density_chunked(
    layers,
    k_points,
    on_site_potentials,
    T,
    target_electrons,
    weights,
    electron_tolerance,
    max_iter,
    batch_size,
):
    """Solve fixed filling with bounded-memory two-pass diagonalization."""
    n_k = len(k_points)
    n_sites = 2 * len(layers)
    eigenvalues = np.empty((n_k, n_sites))
    hamiltonian_buffer = np.empty(
        (batch_size, n_sites, n_sites),
        dtype=np.complex128,
    )

    for start in range(0, n_k, batch_size):
        stop = min(start + batch_size, n_k)
        hamiltonians = _build_hamiltonian_batch(
            layers,
            k_points[start:stop],
            on_site_potentials,
            hamiltonian_buffer,
        )
        eigenvalues[start:stop] = np.linalg.eigvalsh(hamiltonians)

    mu, occupations = _occupations_from_eigenvalues_validated(
        eigenvalues,
        T,
        target_electrons,
        weights,
        electron_tolerance,
        max_iter,
    )

    charge_density = np.zeros(n_sites)
    for start in range(0, n_k, batch_size):
        stop = min(start + batch_size, n_k)
        hamiltonians = _build_hamiltonian_batch(
            layers,
            k_points[start:stop],
            on_site_potentials,
            hamiltonian_buffer,
        )
        _, eigenvectors = np.linalg.eigh(hamiltonians)
        charge_density += SPIN_DEGENERACY * np.einsum(
            "k,kib,kb->i",
            weights[start:stop],
            np.abs(eigenvectors) ** 2,
            occupations[start:stop],
            optimize=True,
        )
        del eigenvectors

    _validate_charge_density(charge_density, target_electrons, electron_tolerance)
    return mu, charge_density


def _add_on_site_potentials(hamiltonians, on_site_potentials):
    result = hamiltonians.copy()
    diagonal = np.arange(result.shape[1])
    result[:, diagonal, diagonal] += on_site_potentials
    return result


def _clip_potential_step(current, proposed, max_step_eV):
    """Limit the largest per-site potential change in one SCF update."""
    if max_step_eV is None:
        return proposed

    step = proposed - current
    max_abs_step = float(np.max(np.abs(step)))
    if max_abs_step <= max_step_eV or max_abs_step == 0.0:
        return proposed

    return current + step * (max_step_eV / max_abs_step)


def _mix_potentials_guarded(
    current,
    target,
    x_history,
    residual_history,
    mixing_parameter,
    max_step_eV=0.5,
    anderson_rcond=1e-10,
    max_alpha_l1=10.0,
):
    """Return the next SCF potential using guarded Anderson acceleration."""
    if not np.all(np.isfinite(current)) or not np.all(np.isfinite(target)):
        raise FloatingPointError("SCF mixing received non-finite potentials.")

    residual = target - current
    simple_candidate = current + mixing_parameter * residual
    chosen_candidate = simple_candidate

    if len(x_history) >= 3 and len(residual_history) >= 3:
        dX = np.array([x_history[k] - x_history[k - 1] for k in range(1, len(x_history))])
        dF = np.array(
            [residual_history[k] - residual_history[k - 1] for k in range(1, len(residual_history))]
        )

        try:
            alpha, *_ = np.linalg.lstsq(dF.T, -residual_history[-1], rcond=anderson_rcond)
            alpha_l1 = float(np.sum(np.abs(alpha)))
            if np.all(np.isfinite(alpha)) and alpha_l1 <= max_alpha_l1:
                anderson_candidate = simple_candidate + np.sum(
                    alpha[:, None] * (dX + mixing_parameter * dF),
                    axis=0,
                )
                if np.all(np.isfinite(anderson_candidate)):
                    if max_step_eV is None:
                        chosen_candidate = anderson_candidate
                    else:
                        anderson_step = float(np.max(np.abs(anderson_candidate - current)))
                        if anderson_step <= float(max_step_eV):
                            chosen_candidate = anderson_candidate
        except np.linalg.LinAlgError:
            pass

    return _clip_potential_step(current, chosen_candidate, max_step_eV)


def _on_site_potentials_from_charge_density(
    charge_density,
    ewald_spatial,
    reference_density,
    hubbard_u,
):
    """Calculate paramagnetic Hartree and local Hubbard on-site potentials.

    The local Hubbard correction is applied in its nonmagnetic mean-field form:
    V_U = U/2 * (rho - rho_neutral).
    """
    charge_density = real_array("charge_density", charge_density, ndim=1)
    n_sites = len(charge_density)
    ewald_spatial = real_array(
        "ewald_spatial",
        ewald_spatial,
        shape=(n_sites, n_sites),
    )
    reference_density = real_array(
        "reference_density",
        reference_density,
        shape=(n_sites,),
    )
    hubbard_u = real("hubbard_u", hubbard_u, minimum=0.0)
    excess_electrons = charge_density - reference_density
    V_H = np.dot(ewald_spatial, excess_electrons)
    V_U = 0.5 * hubbard_u * excess_electrons
    interaction_potential = V_H + V_U
    interaction_potential -= np.mean(interaction_potential)
    if not np.all(np.isfinite(interaction_potential)):
        raise FloatingPointError(
            "Electrostatic potential calculation produced non-finite values."
        )
    return interaction_potential


def _density_observables(
    layers,
    charge_density,
    reference_density,
    ewald_spatial,
    hubbard_u,
):
    """Return induced 2D polarization and electrostatic energy."""
    charge_density = real_array("charge_density", charge_density, ndim=1)
    n_sites = len(charge_density)
    reference_density = real_array(
        "reference_density",
        reference_density,
        shape=(n_sites,),
    )
    ewald_spatial = real_array(
        "ewald_spatial",
        ewald_spatial,
        shape=(n_sites, n_sites),
    )
    hubbard_u = real("hubbard_u", hubbard_u, minimum=0.0)

    positions = get_unit_cell_positions(layers)
    if len(positions) != n_sites:
        raise ValueError("charge density does not match the layer stack.")

    excess_electrons = charge_density - reference_density
    z = np.asarray(positions[:, 2], dtype=float)
    z -= 0.5 * (float(np.min(z)) + float(np.max(z)))

    # Finite stacks have a two-dimensional polarization: dipole per area.
    polarization = float(
        -e * 1e10 * float(np.dot(z, excess_electrons)) / A
    )
    hartree_energy = 0.5 * float(
        excess_electrons @ ewald_spatial @ excess_electrons
    )
    hubbard_energy = 0.25 * hubbard_u * float(
        np.dot(excess_electrons, excess_electrons)
    )
    electrostatic_energy = hartree_energy + hubbard_energy

    if not np.isfinite(polarization) or not np.isfinite(electrostatic_energy):
        raise FloatingPointError("Density-observable calculation produced non-finite values.")
    return DensityObservables(polarization, electrostatic_energy)


def on_site_potentials_from_layers(
    layers,
    T,
    k_grid_size=96,
    iteration_threshold=1000,
    tolerance=1e-3,
    m_hist=5,
    mixing_parameter=0.3,
    max_potential_step=0.5,
    dielectric="hbn",
    ohno_target_meV=0.05,
    delta_electrons=0.0,
    electron_tolerance=1e-8,
    max_mu_iterations=1000,
    return_mu=False,
    return_charge_density=False,
    return_density_observables=False,
    return_residual_history=False,
    raise_on_nonconvergence=True,
    layer_bias=0.0,
    initial_potentials=None,
    verbose=True,
):
    """Calculate spin-degenerate self-consistent on-site potentials.

    Args:
        layers (str): Layer structure (e.g., "abc")
        T (float): Electronic occupation temperature in Kelvin. Exact zero
            uses discontinuous finite-grid filling and may not converge.
        k_grid_size (int): Linear size of the Gamma-centered k-point grid.
        iteration_threshold (int): Maximum number of SCF iterations
        tolerance (float): Maximum per-site fixed-point residual in eV
        m_hist (int): Number of history steps for Anderson mixing
        mixing_parameter (float): Simple mixing parameter (fallback)
        max_potential_step (float or None): Maximum per-site potential update in eV
        dielectric (str): Background dielectric preset for Ewald interactions:
            'hbn', 'vacuum', or 'graphite'.
        ohno_target_meV (float): Supported Ohno matrix-error target in meV.
            Choices are 0.05, 0.02, 0.01, and 0.005 meV.
        delta_electrons (float): Fixed excess carrier filling per unit cell.
        electron_tolerance (float): Maximum electron-count error per unit cell.
        layer_bias (float): Imposed bare top-to-bottom layer onsite energy
            difference in eV.
        return_density_observables (bool): Return induced out-of-plane
            polarization in C/m and electrostatic interaction energy in eV per
            unit cell, both relative to the neutral unbiased reference.
        return_residual_history (bool): Include the per-iteration fixed-point
            residual history in the result.
        raise_on_nonconvergence (bool): Raise RuntimeError when the iteration
            limit is reached before satisfying the SCF tolerance.
        initial_potentials (array-like or None): Optional SCF starting point.
        verbose (bool): Print SCF progress if True.

    Returns:
        SCFResult: Potentials, requested outputs, and convergence diagnostics.
    """
    layers_int = translate_abc_to_012arr(layers)
    n_sites = 2 * len(layers_int)
    T = real("Temperature", T, minimum=0.0)
    k_grid_size = integer("k_grid_size", k_grid_size, minimum=1)
    if k_grid_size % 3:
        raise ValueError("k_grid_size must be divisible by 3 to include K and K'.")
    iteration_threshold = integer(
        "iteration_threshold",
        iteration_threshold,
        minimum=1,
    )
    tolerance = real(
        "tolerance",
        tolerance,
        minimum=0.0,
        strict_minimum=True,
    )
    m_hist = integer("m_hist", m_hist, minimum=1)
    max_mu_iterations = integer(
        "max_mu_iterations",
        max_mu_iterations,
        minimum=1,
    )
    mixing_parameter = real(
        "mixing_parameter",
        mixing_parameter,
        minimum=0.0,
        maximum=1.0,
        strict_minimum=True,
    )
    if max_potential_step is not None:
        max_potential_step = real(
            "max_potential_step",
            max_potential_step,
            minimum=0.0,
            strict_minimum=True,
        )
    delta_electrons = real("delta_electrons", delta_electrons)
    electron_tolerance = real(
        "electron_tolerance",
        electron_tolerance,
        minimum=0.0,
        strict_minimum=True,
    )
    layer_bias = real("layer_bias", layer_bias)
    return_mu = boolean("return_mu", return_mu)
    return_charge_density = boolean(
        "return_charge_density",
        return_charge_density,
    )
    return_density_observables = boolean(
        "return_density_observables",
        return_density_observables,
    )
    return_residual_history = boolean(
        "return_residual_history",
        return_residual_history,
    )
    raise_on_nonconvergence = boolean(
        "raise_on_nonconvergence",
        raise_on_nonconvergence,
    )
    verbose = boolean("verbose", verbose)
    resolve_dielectric(dielectric)
    ohno_cutoff_from_target(ohno_target_meV)
    if len(layers_int) == 1 and layer_bias != 0.0:
        raise ValueError("layer_bias requires at least two layers.")
    if initial_potentials is not None:
        initial_potentials = real_array(
            "initial_potentials",
            initial_potentials,
            shape=(n_sites,),
            copy=True,
        )

    layers_label = "".join("ABC"[layer] for layer in layers_int)
    hubbard_u = hubbard_u_for_layer_count(len(layers_int))

    # The spatial Hamiltonian has one band per site. With spin degeneracy two,
    # neutral graphene has one electron per site in total.
    target_electrons = n_sites + delta_electrons
    max_electrons = SPIN_DEGENERACY * n_sites
    if not 0.0 < target_electrons < max_electrons:
        raise ValueError(
            "delta_electrons gives an impossible filling: "
            f"target_electrons={target_electrons}, "
            f"allowed range is (0, {max_electrons})."
        )

    ewald_spatial = ewald_matrix_calc(
        layers_int,
        dielectric=dielectric,
        ohno_target_meV=ohno_target_meV,
    )
    k_points, weights = monkhorst_pack_mesh(k_grid_size)
    batch_size = _k_point_batch_size(len(k_points), n_sites)
    base_hamiltonians = None
    if batch_size == len(k_points):
        base_hamiltonians = _build_hamiltonian_batch(layers_int, k_points)

    n_layers = len(layers_int)
    bias_array = np.zeros(n_sites)
    if n_layers > 1 and layer_bias != 0.0:
        for i in range(n_layers):
            v_bare = -layer_bias / 2.0 + i * (layer_bias / (n_layers - 1))
            bias_array[2 * i] = v_bare
            bias_array[2 * i + 1] = v_bare

    # All fillings share the neutral, unbiased empirical state as their reference.
    if base_hamiltonians is None:
        mu_neutral, reference_density = _solve_k_point_density_chunked(
            layers_int,
            k_points,
            None,
            T,
            n_sites,
            weights,
            electron_tolerance,
            max_mu_iterations,
            batch_size,
        )
    else:
        mu_neutral, reference_density = _solve_mu_and_charge_density_validated(
            base_hamiltonians,
            T,
            n_sites,
            weights,
            electron_tolerance,
            max_mu_iterations,
        )

    if initial_potentials is None:
        on_site_potentials_current = bias_array.copy()
    else:
        on_site_potentials_current = initial_potentials
    mu_current = mu_neutral

    num_of_iterations = 0
    change_in_potentials = np.inf
    charge_distribution = None
    residual_norm_history = [] if return_residual_history else None

    X = []
    F = []

    if verbose:
        print(f"Starting spin-degenerate SCF calculation for {layers_label}...")
        print(f"Temperature: {T:.1f} K")
        print(f"Target electrons per unit cell: {target_electrons:.6f}")
        print(f"K-point grid: {k_grid_size} x {k_grid_size}")
        print("-" * 60)

    while num_of_iterations < iteration_threshold:
        X.append(on_site_potentials_current.copy())

        if base_hamiltonians is None:
            mu_current, charge_distribution = _solve_k_point_density_chunked(
                layers_int,
                k_points,
                on_site_potentials_current,
                T,
                target_electrons,
                weights,
                electron_tolerance,
                max_mu_iterations,
                batch_size,
            )
        else:
            hamiltonians = _add_on_site_potentials(
                base_hamiltonians,
                on_site_potentials_current,
            )
            mu_current, charge_distribution = _solve_mu_and_charge_density_validated(
                hamiltonians,
                T,
                target_electrons,
                weights,
                electron_tolerance,
                max_mu_iterations,
            )
            del hamiltonians

        new_on_site_potentials = _on_site_potentials_from_charge_density(
            charge_distribution,
            ewald_spatial,
            reference_density,
            hubbard_u,
        )

        if verbose:
            print("Charge Distribution:  ", np.round(charge_distribution, 6))

        new_total_potentials = bias_array + new_on_site_potentials
        residual = new_total_potentials - on_site_potentials_current
        F.append(residual.copy())
        change_in_potentials = float(np.max(np.abs(residual)))
        num_of_iterations += 1
        if residual_norm_history is not None:
            residual_norm_history.append(change_in_potentials)

        if len(X) > m_hist:
            X.pop(0)
            F.pop(0)

        if verbose:
            print(
                f"Iteration {num_of_iterations:3d}: "
                f"Change = {change_in_potentials:.6f} eV, "
                f"Max potential = {np.max(np.abs(on_site_potentials_current)):.6f} eV, "
                f"Mu = {mu_current:.6f} eV"
            )

        if change_in_potentials < tolerance or num_of_iterations == iteration_threshold:
            break

        on_site_potentials_current = _mix_potentials_guarded(
            on_site_potentials_current,
            new_total_potentials,
            X,
            F,
            mixing_parameter,
            max_step_eV=max_potential_step,
        )

    converged = change_in_potentials < tolerance
    electron_error = abs(
        float(np.sum(charge_distribution)) - target_electrons
    )

    density_observables = None
    if return_density_observables:
        density_observables = _density_observables(
            layers_int,
            charge_distribution,
            reference_density,
            ewald_spatial,
            hubbard_u,
        )

    if verbose:
        print("-" * 60)
        if converged:
            print(f"SCF converged after {num_of_iterations} iterations!")
        else:
            print(f"SCF did not converge after {num_of_iterations} iterations.")
            print(f"Final change: {change_in_potentials:.6f} eV (tolerance: {tolerance:.6f} eV)")

        print(f"Final on-site potentials: {np.round(on_site_potentials_current, 6)}")
        print(f"Final chemical potential: {mu_current:.6f} eV")

    result = SCFResult(
        on_site_potentials_current.copy(),
        num_of_iterations,
        converged,
        change_in_potentials,
        electron_error,
        mu_current if return_mu else None,
        charge_distribution.copy() if return_charge_density else None,
        density_observables,
        tuple(residual_norm_history) if residual_norm_history is not None else None,
    )
    if not converged and raise_on_nonconvergence:
        raise RuntimeError(
            f"SCF did not converge for {layers_label} at T={T:g} K on the "
            f"{k_grid_size} x {k_grid_size} grid after {num_of_iterations} iterations: "
            f"residual={change_in_potentials:.3e} eV "
            f"(tolerance={tolerance:.3e} eV); "
            f"electron-count error={electron_error:.3e}; "
            f"mu={mu_current:.6g} eV."
        )
    return result


def converged_on_site_potentials_from_layers(
    layers,
    T,
    initial_grid_size=24,
    grid_step=24,
    max_grid_size=288,
    k_tolerance=1e-3,
    mu_tolerance=1e-3,
    density_tolerance=1e-5,
    energy_tolerance=1e-4,
    confirmation_steps=2,
    **scf_options,
):
    """Refine valley-compatible grids until requested outputs all converge.

    Recommended observable tolerances activate only when their corresponding
    outputs are requested; passing None disables that convergence check.
    Enabled outputs must agree with every grid in the same confirmation window
    used for the on-site potentials.
    """
    initial_grid_size = integer(
        "initial_grid_size",
        initial_grid_size,
        minimum=1,
    )
    grid_step = integer("grid_step", grid_step, minimum=1)
    max_grid_size = integer("max_grid_size", max_grid_size, minimum=1)
    confirmation_steps = integer(
        "confirmation_steps",
        confirmation_steps,
        minimum=1,
    )
    if initial_grid_size % 3 or grid_step % 3:
        raise ValueError("initial_grid_size and grid_step must be divisible by 3.")
    if max_grid_size < initial_grid_size + confirmation_steps * grid_step:
        raise ValueError(
            "max_grid_size must allow at least confirmation_steps refinements."
        )
    k_tolerance = real(
        "k_tolerance",
        k_tolerance,
        minimum=0.0,
        strict_minimum=True,
    )
    tolerances = {"potential": k_tolerance}
    for name, value in {
        "mu": mu_tolerance,
        "density": density_tolerance,
        "energy": energy_tolerance,
    }.items():
        if value is not None:
            value = real(
                f"{name}_tolerance",
                value,
                minimum=0.0,
                strict_minimum=True,
            )
        tolerances[name] = value

    return_mu = boolean("return_mu", scf_options.get("return_mu", False))
    return_density = boolean(
        "return_charge_density",
        scf_options.get("return_charge_density", False),
    )
    return_observables = boolean(
        "return_density_observables",
        scf_options.get("return_density_observables", False),
    )
    active_tolerances = {
        "potential": k_tolerance,
        "mu": tolerances["mu"] if return_mu else None,
        "density": (
            tolerances["density"]
            if return_density or return_observables
            else None
        ),
        "energy": tolerances["energy"] if return_observables else None,
    }
    active_tolerances = {
        name: value for name, value in active_tolerances.items() if value is not None
    }
    scf_tolerance = min(
        value
        for name, value in active_tolerances.items()
        if name != "density"
    ) / 10.0
    if scf_tolerance == 0.0:
        raise ValueError("An energy tolerance is below floating-point resolution.")
    if (
        "k_grid_size" in scf_options
        or "tolerance" in scf_options
        or "initial_potentials" in scf_options
    ):
        raise ValueError(
            "Grid size, SCF tolerance, and warm start are controlled by this function."
        )

    internal_density = return_density or "density" in active_tolerances
    run_options = dict(scf_options)
    run_options["return_charge_density"] = internal_density
    output_specs = (
        ("mu", return_mu, lambda result: result.chemical_potential_eV),
        ("density", internal_density, lambda result: result.charge_density.copy()),
        (
            "energy",
            return_observables,
            lambda result: result.density_observables.electrostatic_energy_eV,
        ),
    )
    history = []
    errors = None
    for grid_size in range(initial_grid_size, max_grid_size + 1, grid_step):
        result = on_site_potentials_from_layers(
            layers,
            T,
            k_grid_size=grid_size,
            tolerance=scf_tolerance,
            initial_potentials=history[-1]["potential"] if history else None,
            **run_options,
        )
        potentials = result.potentials
        if not result.converged:
            raise RuntimeError(
                f"SCF did not converge on the {grid_size} x {grid_size} grid: "
                f"residual={result.residual_eV:.3e} eV."
            )

        values = {"potential": potentials.copy()}
        for name, present, transform in output_specs:
            if present:
                values[name] = transform(result)
        history.append(values)
        if len(history) > confirmation_steps + 1:
            history.pop(0)
        if len(history) == confirmation_steps + 1:
            current = history[-1]
            metric_errors = {
                name: max(
                    float(np.max(np.abs(current[name] - previous[name])))
                    for previous in history[:-1]
                )
                for name in active_tolerances
            }
            errors = GridConvergence(
                metric_errors["potential"],
                metric_errors.get("mu"),
                metric_errors.get("density"),
                metric_errors.get("energy"),
            )
            if all(
                metric_errors[name] < tolerance
                for name, tolerance in active_tolerances.items()
            ):
                if internal_density and not return_density:
                    result = result._replace(charge_density=None)
                return result, grid_size, errors

    labels = {
        "potential": ("potential", "eV"),
        "mu": ("chemical-potential", "eV"),
        "density": ("density", "electrons/site"),
        "energy": ("energy", "eV"),
    }
    error_values = {
        "potential": errors.potential_eV,
        "mu": errors.chemical_potential_eV,
        "density": errors.density_electrons_per_site,
        "energy": errors.electrostatic_energy_eV,
    }
    details = [
        f"{labels[name][0]} error={error_values[name]:.3e} {labels[name][1]} "
        f"(tolerance={tolerance:.3e})"
        for name, tolerance in active_tolerances.items()
    ]
    raise RuntimeError(
        "K-point sampling did not converge: " + "; ".join(details) + "."
    )
