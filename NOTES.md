# Implementation Notes

This file only documents choices that are not obvious from reading the code.

## Ohno Cutoff Policy

The Ohno correction has a slow algebraic tail, unlike the exponentially
convergent Ewald sum, so it uses a separate validated cutoff target:

| `ohno_target_meV` | cutoff |
| --- | --- |
| `0.05` | `35 Angstrom` |
| `0.02` | `50 Angstrom` |
| `0.01` | `75 Angstrom` |
| `0.005` | `95 Angstrom` |

These cutoffs came from scans against stricter reference calculations over many
stackings and layer counts. Unsupported targets raise an error instead of
extrapolating beyond the validated values.

## Hubbard And Ohno Coupling

The interaction strength is fixed by layer count: monolayer graphene uses the
graphene cRPA value `U = 9.3 eV`, while multilayers use the bulk-graphite value
`U = 8.0 eV` as a model approximation. No unsupported interpolation is made for
few-layer stacks.

The selected value sets both the Ohno zero-distance scale and the local
paramagnetic term `V_U = 0.5 * U * (rho - rho_neutral)`, keeping both parts of
the interaction on one calibrated scale rather than exposing an arbitrary
runtime parameter.

## Spin Treatment

The current code is intentionally spin-degenerate, not magnetic. Spin enters
only through the occupation factor `SPIN_DEGENERACY = 2.0`; there are no
separate spin-up/spin-down Hamiltonians, densities, or potentials.

This avoids implying a magnetic symmetry-broken solution. Choosing
ferromagnetic or antiferromagnetic seeds is a separate physical modeling
problem, not something this code can infer reliably from stacking alone.

## Chemical Potential And Doping

The chemical potential is always solved self-consistently for a fixed total
electron count. The doping input is `delta_electrons`, meaning excess electrons
per unit cell:

```python
target_electrons = n_sites + delta_electrons
```

This replaced fixed-chemical-potential runs, which can let the total charge
drift during SCF and make the result physically ambiguous for this workflow.

`delta_electrons` and `layer_bias` independently set carrier filling and bare
layer asymmetry. All fillings use the neutral, unbiased state as the
electrostatic reference, so doping-induced charge redistribution is included.
For a net-charged slab, the common interaction-potential offset is removed;
only internal potential differences are defined without a device-specific gate
model.

## Lattice-Vector Enumeration

`lattice_vectors_within_cutoff` bounds the integer search using the smallest
singular value of the lattice basis, then filters by the actual circular cutoff.
This avoids missing valid vectors for oblique bases, which can otherwise create
cutoff-dependent electrostatic errors.

## K-Point Sampling

SCF uses equal-weight, Gamma-centered Monkhorst-Pack meshes whose linear size is
divisible by three, so both graphene valleys are sampled exactly. K-point
convergence refines the grid in divisible-by-three steps and requires two
refinements whose final potential agrees with every grid in the confirmation
window. Each refinement starts from the previous potential, and its inner SCF
tolerance is one tenth of the requested k-point tolerance.

Exact-zero-temperature filling is supported, but its finite-grid SCF map can
be discontinuous and need not converge. For zero-temperature predictions,
converge at successively lower positive electronic temperatures using the
previous potentials as the next initial state, and verify the limiting result.
SCF nonconvergence raises by default rather than returning a usable state.

SCF diagonalization remains single-pass while its estimated working set is at
most 512 MiB. Larger calculations automatically use bounded-memory k-point
batches, retaining only the global eigenvalues between density passes.

Requesting chemical potential, site density, or density observables activates
recommended convergence targets on the same refinement sequence:
`mu_tolerance = 1e-3 eV`, `density_tolerance = 1e-5` electrons per site, and
`energy_tolerance = 1e-4 eV` per unit cell. Each remains configurable, and
passing `None` explicitly disables that check. Unrequested outputs add no
convergence requirement.

## Density Observables

For excess electron density `q = rho - rho_neutral`, the induced out-of-plane
polarization is the electronic dipole per unit-cell area,
`Pz = -e * sum(z * q) / A`, in C/m, with `z = 0` at the stack midplane. It is
origin-independent for neutral redistribution; at nonzero net doping the
midplane convention is device-relative rather than an absolute polarization.

The returned electrostatic energy is
`0.5 * q.T @ V @ q + 0.25 * U * q.T @ q` in eV per unit cell. It excludes band
energy, exchange, and work done by an explicit gate. Without a device boundary
model, charged-state energies are comparable only at the same total filling.

## SWMcC Next-Nearest-Layer Couplings

`g2` and `g5` retain the conventional bulk SWMcC values. Their direct
finite-stack hopping matrix elements are `g2 / 2` and `g5 / 2`.

`delta` is the dimer-site on-site energy relative to non-dimer sites: it is
Koshino and McCann's `delta`, and corresponds to the bulk dimer/non-dimer
splitting `2 * Delta'` in the convention of Garcia-Ruiz et al. For the
Dresselhaus bulk set, `Delta' = 0.025 eV`, so `delta = 0.050 eV`.

## References

- Koshino and McCann, [*Gate-induced interlayer asymmetry in ABA-stacked
  trilayer graphene*](https://arxiv.org/abs/0809.0983): multilayer SWMcC
  Hamiltonian, dimer-site `delta`, finite-stack `g2 / 2` and `g5 / 2`, and
  fixed-density Hartree screening under an external layer asymmetry.
- Garcia-Ruiz et al., [*Full Slonczewski-Weiss-McClure parametrization of
  few-layer twistronic graphene*](https://arxiv.org/abs/2105.00086):
  Dresselhaus SWMcC parameter set and the `Delta'` dimer/non-dimer convention.
- Jung and MacDonald, [*Accurate tight-binding and continuum models for the
  pi bands of bilayer graphene*](https://arxiv.org/abs/1309.5429): hopping-sign
  conventions and the mappings `t2 = g2 / 2`, `t5 = g5 / 2`.
- McCann, [*Asymmetry gap in the electronic band structure of bilayer
  graphene*](https://arxiv.org/abs/cond-mat/0608221): self-consistent Hartree
  screening in gated bilayer graphene.
- Wehling et al., [*Strength of effective Coulomb interactions in graphene and
  graphite*](https://arxiv.org/abs/1101.4007): constrained-RPA interaction
  scales used for the graphene and graphite Hubbard parameters.
- Laturia, Van de Put, and Vandenberghe, [*Dielectric properties of hexagonal
  boron nitride and transition metal dichalcogenides: from monolayer to
  bulk*](https://doi.org/10.1038/s41699-018-0050-x): static anisotropic h-BN
  dielectric constants.
- Parry, [*The electrostatic potential in the surface region of an ionic
  crystal*](https://doi.org/10.1016/0039-6028(75)90362-3): Ewald summation for
  systems periodic in two dimensions.
- Ohno, [*Some remarks on the Pariser-Parr-Pople method*](https://doi.org/10.1007/BF00528281):
  short-range regularization of the Coulomb interaction.
- Monkhorst and Pack, [*Special points for Brillouin-zone
  integrations*](https://doi.org/10.1103/PhysRevB.13.5188): uniform reciprocal
  space meshes.
- Anderson, [*Iterative procedures for nonlinear integral
  equations*](https://doi.org/10.1145/321296.321305): accelerated fixed-point
  mixing used by the SCF iteration.
