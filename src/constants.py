"""
Constants and Conversion Factors

This module defines the universal physical constants, unit conversion factors,
and standard data column indices utilized throughout the planetary interior
modeling framework.
"""

import numpy as np

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

# Universal gravitational constant (Units: m^3 kg^-1 s^-2)
G_CONST: float = 6.67430e-11

# Mass of the Earth (Units: kg)
M_EARTH: float = 5.972e24

# Volumetric mean radius of the Earth (Units: m)
R_EARTH: float = 6357000.0

# Mass of Jupiter (Units: kg)
M_JUPITER: float = 1.898e27

# Equatorial radius of Jupiter (Units: m)
R_JUPITER: float = 71492000.0


# =============================================================================
# UNIT CONVERSIONS
# =============================================================================

# Conversion factor from Bar to Pascals (1 Bar = 10^5 Pa)
BAR_TO_PA: float = 1e5

# Conversion factor from Megajoules to Joules (1 MJ = 10^6 J)
MJ_TO_J: float = 1e6


# =============================================================================
# DATA COLUMNS (INTERNAL CONVENTION)
# =============================================================================
# Standardized column indices for parsing and interpolating 
# Equation of State (EOS) data arrays.

# Index for Temperature (T)
T_COL: int = 0

# Index for Pressure (P)
P_COL: int = 1

# Index for Density (Rho)
RHO_COL: int = 2

# Index for Entropy (S)
S_COL: int = 3