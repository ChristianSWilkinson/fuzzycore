# Planetary Interior Framework: Fuzzy Cores & Sub-Neptunes

This repository contains the Python-based numerical framework developed to solve the equations of hydrostatic equilibrium for multi-phase planetary interiors. It is designed to rigorously map the structural degeneracies between solid cores, volatile water mantles, and deeply suspended dilute "fuzzy" gradients.

This framework is built for robustness, explicitly handling severe thermodynamic discontinuities (such as liquid-vapor phase transitions) and adaptive atmospheric downsampling, making it capable of modelling both dense water-worlds (e.g., GJ 1214 b) and highly inflated super-puffs (e.g., Kepler-11e).

**Primary Reference:**
If you use this code in your research, please cite our corresponding *Astronomy & Astrophysics* paper:
> Wilkinson, C., Mazevet, S., Lagrange, A.-M., Charnay, B. (2026). *A robust numerical framework for giant planet interior modeling: Constraining the core-envelope equivalence in the presence of dilute gradients.* A&A.

---

## 📂 Project Structure

```text
planetary_interior/
├── data/               # Equation of State (EOS) tables (ANEOS, AQUA, Chabrier)
├── figures/            # Output directory for generated plots
├── notebooks/          # Jupyter notebooks for exploratory analysis
├── scripts/            # Executable scripts for parameter sweeps and modeling
└── src/                # Core physics, solver, and EOS interpolation modules