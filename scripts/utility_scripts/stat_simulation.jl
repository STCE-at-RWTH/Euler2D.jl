#! /usr/bin/env julia
# requires Julia 1.11

using Euler2D
using Accessors

function (@main)(ARGS)
    @assert length(ARGS) >= 1

    res = open(ARGS[1]) do f
        n_tsteps = read(f, Int)
        mode = read(f, Euler2D.EulerSimulationMode)
        n_seeds = mode == Euler2D.TANGENT ? read(f, Int) : 0
        n_active = read(f, Int)
        n_dims = read(f, Int)
        n_cells = (read(f, Int), read(f, Int))

        return (n_tsteps, mode, n_seeds, n_active, n_dims, n_cells)
    end

    @info "Euler2D cell simulation information:" N_t = res[1] MODE = res[2] SEEDS = res[3] ACTIVE_CELLS =
        res[4] NDIMS = res[5] (n_x, n_y) = res[6]
end
