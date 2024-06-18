"""
    EulerSim{N, T}

Represents a completed simulation of the Euler equations on an N-dimensional grid.

## Fields
- `ncells::(Int, Int, ...)`: Number of cells in each of the `N` dimensions of the simulation.
- `nsteps::Int`: Number of time steps taken in the simulation
- `dims::{(T, T), (T, T), ...}`: Bounds of the simulation in each of the dimensions.
- `tsteps::Vector{T}`: Time steps.
- `u::Array{T, N+1}`: `(N+2) × ncells... x nsteps` array of data for u

## Methods

"""
struct EulerSim{N,T}
    ncells::NTuple{N,Int}
    nsteps::Int
    dims::NTuple{N,Tuple{T,T}}
    tsteps::Vector{T}
    u::Array{T,N + 1}
end

# TODO utility functions for plotting these things, maybe look at PlotRecipes? No need to go too overboard, though.
# extend this to work with out-of-memory data?

function cell_boundaries(e, dim)
    return range(e.dims[dim]...; length = e.ncells[dim] + 1)
end

function cell_centers(e, dim)
    faces = cell_boundaries(e, dim)
    return faces[1:end-1] .+ step(faces) / 2
end

function nth_step(e::EulerSim{N,T}, n) where {N,T}
    return e.tsteps[n], view(e.u, ntuple(i -> Colon(), N - 2)..., n)
end

# TODO for Very Large simulations we'll need to tape these rather than holding them entirely in RAM

function simulate_euler_equations(
    bounds,
    ncells,
    boundary_conditions,
    T_end,
    u0;
    gas::CaloricallyPerfectGas = DRY_AIR,
    cfl_limit = 0.75,
    max_tsteps = typemax(Int),
    write_result = true,
    return_complete_result = false,
    history_in_memory = false,
    output_tag = "euler_sim",
)
    N = length(ncells)
    @assert N == length(bounds) "ncells and bounds must match in length"
    @assert N == length(boundary_conditions) "must provide a boundary condition for each bound (or vice versa)"
    T_end > 0 && DomainError("T_end = $T_end invalid, T_end must be positive")
    0 < cfl_limit < 1 && @warn "CFL invalid, must be between 0 and 1 for stabilty" cfl_limit

    cell_ifaces = [range(b...; length = n + 1) for (b, n) ∈ zip(bounds, ncells)]
    cell_centers = [ax[1:end-1] .+ step(ax) / 2 for ax ∈ cell_ifaces]
    dV = step.(cell_centers)

    @assert length(u0((getindex.(cell_centers, 1))...)) == N + 2

    # set up initial data
    u = stack(p -> u0(p...), Iterators.product(cell_centers...))
    u_next = zeros(eltype(u), size(u)...)
    n_tsteps = 1
    t = zero(eltype(u))

    if write_result
        if !isdir("data")
            mkdir("data")
        end
        tape_file = joinpath("data", output_tag * ".tape")
    end

    if !write_result && !history_in_memory
        @info "Only the final value of the simulation will be returned." T_end
    end
    u_history = Vector{typeof(u)}[]
    t_history = Vector{eltype(u)}[]
    if history_in_memory
        push!(u_history, copy(u))
        push!(t_history, t)
    end
    write_result && open(tape_file, "w+") do f
        write(f, zero(Int)) #need this later
        write(f, length(ncells))
        for (n, b) ∈ zip(ncells, bounds)
            write(f, n, b...)
        end
        write(f, t)
        write(f, u)
    end

    while !(t > T_end || t ≈ T_end) && n_tsteps < max_tsteps
        Δt = try
            maximum_Δt(u, dV, boundary_conditions, CFL, gas)
        catch e
            err isa DomainError || throw(err)
            st = stacktrace(first(current_exceptions()).backtrace)
            is_speed_of_sound = any(st) do f
                f.func == :speed_of_sound
            end
            is_speed_of_sound || throw(err)
            P_over_ρ = err.val / DRY_AIR.γ
            @error "Non-physical state reached in simulation. 
                    Likely cause is a boundary interaction. Aborting simulation early." P_over_ρ e
            break
        end
        Δt = min(Δt, T_end - t)
        if length(t) % 10 == 1
            @info TSTEP = n_tsteps t_k = t Δt
        end

        step_euler_hll!(u_next, u, Δt, dV, boundary_conditions, gas)
        t += Δt
        # MONUMENT TO MY STUPIDITY
        u .= u_next
        n_tsteps += 1

        # opening the file is probably trivial compared to writing
        #   one megafloat
        write_result && open(tape_file, "a") do f
            write(f, t)
            write(f, u)
        end
        if history_in_memory
            push!(u_history, copy(u))
            push!(t_history, t)
        end
    end

    write_result && open(tape_file, "w") do f
        write(f, n_tsteps)
    end

    if history_in_memory
        @assert n_tsteps == length(t_history)
        return EulerSim(
            (ncells...,),
            n_tsteps,
            (((first(r), last(r)) for r ∈ cell_ifaces)...,),
            t_history,
            stack(u_history),
        )
    elseif return_complete_result && write_result
        return load_euler_sim(tape_file)
    end
    return EulerSim(
        (ncells...,),
        1,
        (((first(r), last(r)) for r ∈ cell_ifaces)...,),
        [t],
        u,
    )
    # TODO need to create sim object and write it out if flagged
end


load_euler_sim(path) = open(path) do f
    n_tsteps = read(f, Int)
    ndims = read(f, Int)
    bounds = 
end