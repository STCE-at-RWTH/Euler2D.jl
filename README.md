# Euler2D.jl

An FVM solver for the Euler equations using the Harten-Lax-van Leer numerical scheme.

## Solving the Euler Equations

The package provides two types of simulations:

- `EulerSim{N, NAXES, T}` and `simulate_euler_equations` to simulate the solution to the Euler equations on a regular cartesian grid in `N` dimensions.
- `CellBasedEulerSim{T, Q1, Q2, Q3}` and `simulate_euler_equations_cells` to simulate the solution to the euler equations on a regular grid in two dimensions, with the option to include solid boundaries inside the computational domain.

## Scripts

The `scripts` directory contains scripts to generate data files for further analysis.
