# Euler2D.jl

This package provides an implementation of an HLL solver for the Euler equations. The purpose of the project is to support STCE's teaching efforts and research into differentiable programming.

## Usage

The "simplest" set-up for a bow-shock simulation is shown at `scripts/mwe.jl`.

The process is as follows:

1. Set ``u0(x, p)``.
2. Choose bounds, grid size, a nondimensionalization scale, and starting parameters.
3. Get a keyword dictionary for the simulation configuration.
4. Start the simulation.

It is also possible to resume a simulation from an already-generated output file.

## Dependencies

This package depends on 

- [`ShockwaveProperties.jl`](https://github.com/STCE-at-RWTH/ShockwaveProperties.jl)
- [`PlanePolygons.jl`](https://github.com/STCE-at-RWTH/PlanePolygons.jl)
- [`SimpleIntegration.jl`](https://github.com/STCE-at-RWTH/SimpleIntegration.jl)

These packages can be accessed through STCE's Julia Registry, which can be added at the REPL via:

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
