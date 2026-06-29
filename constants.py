"""Physical constants and parameters for tight-binding calculations."""

import numpy as np

# PHYSICAL CONSTANTS

a = 1.42  # Distance between A-B atoms in Angstrom
a0 = np.sqrt(3) * a  # Lattice constant in Angstrom
c = 3.4  # Distance between layers in Angstrom
epsilon_0 = 8.8541878128e-12  # Permittivity of vacuum in Farads/Meter
kb = 8.61733326e-5  # Boltzmann constant in eV/Kelvin
hbar = 6.58211957e-16  # Reduced Planck constant in eV·sec
delta = 0.050  # Bulk Dresselhaus dimer/non-dimer splitting, 2 * Delta'
e = 1.60217662e-19  # Elementary charge in Coulombs

# Coulomb interaction parameters
u00_graphite = 8.0  # Effective on-site (Hubbard) interaction for partially-screened bulk graphite(cRPA) in eV
u00_graphene = 9.3  # Effective on-site (Hubbard) interaction for partially-screened graphene (cRPA) in eV
epsilon_r_graphite = 2.4  # In-plane dielectric for bulk graphite / encapsulated stack
epsilon_r_graphene = 1.36  # In-plane dielectric for graphene 
epsilon_parallel_hbn = 6.9  # In-plane static dielectric constant of h-BN
epsilon_r_hbn = 3.76  # Out-of-plane static dielectric constant (ϵ⊥0) of Hexagonal Boron Nitride (h-BN)

# Effective background dielectrics
epsilon_parallel = epsilon_parallel_hbn
epsilon_perp = epsilon_r_hbn
DIELECTRIC_PRESETS = {
    "hbn": (epsilon_parallel_hbn, epsilon_r_hbn),
    "vacuum": (1.0, 1.0),
    "graphite": (epsilon_r_graphite, epsilon_r_graphite),
}

def hubbard_u_for_layer_count(layer_count):
    """Return the fixed cRPA Hubbard parameter for a finite graphene stack."""
    if (
        isinstance(layer_count, (bool, np.bool_))
        or not isinstance(layer_count, (int, np.integer))
        or layer_count < 1
    ):
        raise ValueError("layer_count must be a positive integer.")
    return u00_graphene if layer_count == 1 else u00_graphite

# Tight-binding interlayer hopping parameters (γ) in eV
g0 = -3.16
g1 = 0.39
g2 = -0.02
g3 = -0.315
g4 = 0.044
g5 = 0.038

# Direct next-nearest-layer hoppings corresponding to the bulk SWMcC parameters
t2 = g2 / 2
t5 = g5 / 2

# Primitive vectors in real space
a1 = np.sqrt(3) * a * np.array([np.sqrt(3), 1, 0], dtype=np.longdouble) / 2
a2 = np.sqrt(3) * a * np.array([np.sqrt(3), -1, 0], dtype=np.longdouble) / 2

# Unit cell area in real space
A = np.linalg.norm(np.cross(a1, a2))

# Primitive vectors in reciprocal space
b1 = (2 * np.pi / (3 * a)) * np.array([1, np.sqrt(3), 0], dtype=np.longdouble)
b2 = (2 * np.pi / (3 * a)) * np.array([1, -np.sqrt(3), 0], dtype=np.longdouble)

# In-plane shifts for A, B, and C layer registries
LAYER_SHIFTS = np.array([[0, 0], [a, 0], [-a, 0]], dtype=np.longdouble)

# First Brillouin Zone (FBZ) high-symmetry points (K, M, Γ) in reciprocal space
K = (2 * b1[:2] + b2[:2]) / 3
M = (b1[:2] + b2[:2]) / 2
Gamma = np.array([0, 0])


# Second order (k-independent) interaction matrices
ABA_interaction = np.diag([t2, t5]).astype("complex128")
ACA_interaction = np.diag([t5, t2]).astype("complex128")
ABC_interaction = np.array([[0, t2], [0, 0]], dtype="complex128")
ACB_interaction = np.array([[0, 0], [t2, 0]], dtype="complex128")
