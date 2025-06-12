#! /usr/bin/env julia
# requires Julia 1.11

using Euler2D
using Accessors

###
# accepts file output file [time step indices]
###
function (@main)(ARGS)
    @assert length(ARGS) >= 3
    big_sim = load_cell_sim(ARGS[1])
    tsteps = parse.(Int, ARGS[3:end])
    @info "Creating new file with different time step data:" tsteps ARGS[2]
    small_sim = CellBasedEulerSim(
        big_sim.ncells,
        length(tsteps),
        big_sim.bounds,
        big_sim.tsteps[tsteps],
        big_sim.cell_ids,
        big_sim.cells[tsteps],
    )
    Euler2D.write_cell_sim(ARGS[2], small_sim)
end
