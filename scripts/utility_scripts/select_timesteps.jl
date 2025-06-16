#! /usr/bin/env julia
# requires Julia 1.11

using Euler2D
using Accessors

###
# accepts file output file [time step indices]
###
function (@main)(ARGS)
    @assert length(ARGS) >= 3
    tsteps = parse.(Int, ARGS[3:end])
    sim = load_cell_sim(ARGS[1]; steps = tsteps)
    @info "Creating new file with different time step data:" tsteps ARGS[2]
    Euler2D.write_cell_sim(ARGS[2], sim)
end
