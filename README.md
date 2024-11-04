# Euler2D.jl

This package provides an implementation of an HLL solver for the Euler equations. The purpose of the project is to support STCE's teaching efforts and research into differentiable programming.

## Usage

This package exports `simulate_euler_equations`, which simulates the Euler equations in 1, 2, and 3 dimensions. See the documentation for more details. This package also exports `load_euler_sim`, for loading the data files written during simulation.

This package exports `simulate_euler_equations_cells`, which simulations solutions to the Euler equations using a cell list in two dimensions. Simple obstacles are implemented for this system.

## Dependencies

This package depends on [`ShockwaveProperties.jl`](https://github.com/STCE-at-RWTH/ShockwaveProperties.jl). This can be accessed through STCE's Julia Registry, which can be added at the REPL via:

```julia
]registry add https://github.com/STCE-at-RWTH/STCEJuliaRegistry
```

## Scripts

The `scripts` directory contains scripts to generate data files for further analysis. The script files are most easily run by `cd`-ing to the project directory (where the repository is cloned, default `Euler2D.jl`) and using the following command:

```bash
julia --project scripts/script_name.jl
```

The cell-based simulation routines in this package support multithreading! To run a script with multiple threads (_this is recommended_), use

```bash
julia --project --threads=4 scripts/script_name.jl
```

If your machine many (or few...) cores available, you can adjust the `--threads` parameter up (or down...). Parallelism in this package is task-based, which means there are no restrictions on odd numbers of threads.

## Documentation

Documentation is on its way... the docstrings should be good enough (TM). As soon as I figure out how to use GH Actions for deploying docs, they will exist.
