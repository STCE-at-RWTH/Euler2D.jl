"""
    EulerSim{N, NAXES, T}

Represents a completed simulation of the Euler equations on an N-dimensional grid.

## Type Parameters
- `N`: Number of space dimensions.
- `NAXES`: Number of data axes. Should always be `N+2`.
- `T`: Data type for all computations.

## Fields
- `ncells::(Int, Int, ...)`: Number of cells in each of the `N` dimensions of the simulation.
- `nsteps::Int`: Number of time steps taken in the simulation
- `bounds::{(T, T), (T, T), ...}`: Bounds of the simulation in each of the dimensions.
- `tsteps::Vector{T}`: Time steps.
- `u::Array{T, N+2}`: `(N+2) × ncells... x nsteps` array of data for u

## Methods
- `n_data_dims`, `n_space_dims`, `n_tsteps``, ``grid_size``
- `cell_centers`: Get the co-ordinates of cell centers.
- `cell_boundaries`: Get the co-ordinates of cell faces.
- `nth_step`: Get the information at time step `n`.
- `eachstep`: Get a vector of `(t_k, u_k)` tuples for easy iteration.
- `density_field, momentum_density_field, total_internal_energy_density_field`: Get views of certain quantities at a time step.
- `pressure_field, velocity_field, mach_number_field`: Compute quantities at a given time step.
"""
struct EulerSim{N,NAXES,T}
    ncells::NTuple{N,Int}
    nsteps::Int
    bounds::NTuple{N,Tuple{T,T}}
    tsteps::Vector{T}
    u::Array{T,NAXES}
end

# TODO utility functions for plotting these things, maybe look at PlotRecipes? No need to go too overboard, though.
# extend this to work with out-of-memory data?

"""
    n_space_dims(e::EulerSim)

Number of spatial dimensions simulated in `e`.
"""
n_space_dims(::EulerSim{N,NAXES,T}) where {N,NAXES,T} = N

"""
    grid_size(sim)

Size of the spatial axes of this simulation.
"""
grid_size(sim) = sim.ncells

"""
    n_data_dims(e::EulerSim)

Number of axes in the data array of an `EulerSim`.
"""
n_data_dims(::EulerSim{N,NAXES,T}) where {N,NAXES,T} = NAXES

"""
    n_tsteps(sim)

Number of time steps taken in an EulerSim or CellBasedEulerSim.
"""
n_tsteps(sim) = sim.nsteps

""" 
    cell_boundaries(sim, n)

Return `StepRange` of all the cell face positions for the `n`th space dimension in `sim`.
"""
function cell_boundaries(sim, dim)
    return range(sim.bounds[dim]...; length = grid_size(sim)[dim] + 1)
end

"""
    cell_boundaries(esim::EulerSim{N, NAXES, T})

Return a tuple of `N` `StepRanges` of all the cell face positions in `esim`.
"""
function cell_boundaries(e::EulerSim{N,NAXES,T}) where {N,NAXES,T}
    return ntuple(i -> cell_boundaries(e, i), N)
end

"""
    cell_centers(sim, n)

Return `StepRange` of all the cell center positions for the `n`th space dimension in `sim`.
"""
function cell_centers(sim, dim)
    ifaces = cell_boundaries(sim, dim)
    return ifaces[1:end-1] .+ step(ifaces) / 2
end

"""
    cell_boundaries(esim::EulerSim{N, NAXES, T})

Return a tuple of `N` `StepRanges` of all the cell center positions in `esim`.
"""
function cell_centers(e::EulerSim{N,NAXES,T}) where {N,NAXES,T}
    return ntuple(i -> cell_centers(e, i), N)
end

"""
    nth_step(sim, n)

Return `(t, u)` for the `nth` time step in `sim`. `u` will be a view.
"""
function nth_step(esim::EulerSim{N,NAXES,T}, n) where {N,NAXES,T}
    return (
        esim.tsteps[n],
        view(esim.u, Colon(), ntuple(i -> Colon(), n_space_dims(esim))..., n),
    )
end

"""
    eachstep(sim)

Return a vector of `[(t1, u1), (t2, u2),...]`. 
Wraps `nth_step` for for convieniently iterating through a simulation.
"""
eachstep(esim::EulerSim) = [nth_step(esim, n) for n ∈ 1:n_tsteps(esim)]

"""
    pressure_field(esim::EulerSim, n, gas::CaloricallyPerfectGas)

Compute the pressure field for an array-based Euler simulation `esim` at time step `n`.
"""
function pressure_field(
    esim::EulerSim{N,NAXES,T},
    n,
    gas::CaloricallyPerfectGas,
) where {N,NAXES,T}
    _, u = nth_step(esim, n)
    P = reshape(
        mapslices(u; dims = 1) do uij
            ustrip(
                ShockwaveProperties._units_P,
                pressure(uij[1], @view(uij[2:end-1]), uij[end], gas),
            )
        end,
        grid_size(esim),
    )
    return P
end

"""
    density_field(esim::EulerSim, n)

Get a view of the density field for an array-based euler simulation at time step `n`
"""
function density_field(esim::EulerSim, n)
    _, u = nth_step(esim, n)
    idxs = ntuple(Returns(Colon()), n_space_dims(esim))
    return view(u, 1, idxs...)
end

"""
    momentum_density_field(esim::EulerSim, n)

Get a view of the momentum density field for an array-based euler simulation at time step `n`.
"""
function momentum_density_field(esim::EulerSim, n)
    _, u = nth_step(esim, n)
    free_idxs = ntuple(Returns(Colon()), n_space_dims(esim))
    idxs = 2:(n_space_dims(esim)+1)
    return view(u, idxs, free_idxs...)
end

"""
    total_internal_energy_density_field(esim::EulerSim, n)

Get a view of the total internal energy density field for an array-based euler simulation at time step `n`.
"""
function total_internal_energy_density_field(esim::EulerSim, n)
    _, u = nth_step(esim, n)
    free_idxs = ntuple(Returns(Colon()), n_space_dims(esim))
    return view(u, 4, free_idxs...)
end

"""
    velocity_field(esim::EulerSim, n)

Compute the velocity field for an array-based euler simulation at time step `n`.
"""
function velocity_field(esim::EulerSim, n)
    ρ = density_field(esim, n)
    ρv = momentum_density_field(esim, n)
    v = similar(ρv)
    for i ∈ CartesianIndices(grid_size(esim))
        v[:, i] .= ρv[:, i] ./ ρ[i]
    end
    return v
end

"""
    mach_number_field(esim::EulerSim, n, gas::CaloricallyPerfectGas)

Compute the mach number field for an array-based euler simulation at time step `n` in gas `gas`.
"""
function mach_number_field(esim::EulerSim, n, gas::CaloricallyPerfectGas)
    _, u = nth_step(esim, n)
    v = velocity_field(esim, n)
    M = similar(v)
    for i ∈ CartesianIndices(grid_size(esim))
        a_loc = ustrip(
            ShockwaveProperties._units_v,
            speed_of_sound(u[1, i], u[2:end-1, i], u[end, i], gas),
        )
        M[:, i] .= v[:, i] / a_loc
    end
    return M
end

"""
    simulate_euler_equations(u0, T_end, boundary_conditions, bounds, ncells; gas, CFL, max_tsteps, write_output, output_tag)

Simulate the solution to the Euler equations from `t=0` to `t=T`, with `u(0, x) = u0(x)`. 
Time step size is computed from the CFL condition.

The simulation will fail if any nonphysical conditions are reached (speed of sound cannot be computed). 
There is an attempt to gracefully handle this.

The simulation can be written to disk.

Arguments
---
- `u0`: ``u(0, x...):ℝ^n↦ℝ^{n+2}``: conditions at time `t=0`.
- `T_end`: Must be greater than zero.
- `boundary_conditions`: a tuple of boundary conditions for each space dimension
- `bounds`: a tuple of extents for each space dimension (tuple of tuples)
- `ncells`: a tuple of cell counts for each dimension

Keyword Arguments
---
- `gas=DRY_AIR`: The fluid to be simulated.
- `CFL=0.75`: The CFL condition to apply to `Δt`. Between zero and one, default `0.75`.
- `max_tsteps`: Maximum number of time steps to take. Defaults to "very large".
- `write_result=true`: Should output be written to disk?
- `history_in_memory`: Should we keep whole history in memory?
- `return_complete_result`: Should a complete record of the simulation be returned by this function?
- `output_tag`: File name for the tape and output summary.
- `thread_pool`: Which thread pool should be used? 
    (currently unused, but this does multithread via Threads.@threads)
"""
function simulate_euler_equations(
    u0,
    T_end,
    boundary_conditions,
    bounds,
    ncells;
    gas::CaloricallyPerfectGas = DRY_AIR,
    cfl_limit = 0.75,
    max_tsteps = typemax(Int),
    write_result = true,
    return_complete_result = false,
    history_in_memory = false,
    output_tag = "euler_sim",
    thread_pool = :default,
)
    N = length(ncells)
    @assert N == length(bounds) "ncells and bounds must match in length"
    @assert N == length(boundary_conditions) "must provide a boundary condition for each bound (or vice versa)"
    T_end > 0 && DomainError("T_end = $T_end invalid, T_end must be positive")
    0 < cfl_limit < 1 || @warn "CFL invalid, must be between 0 and 1 for stabilty" cfl_limit

    cell_ifaces = [range(b...; length = n + 1) for (b, n) ∈ zip(bounds, ncells)]
    cell_centers = [ax[1:end-1] .+ step(ax) / 2 for ax ∈ cell_ifaces]
    dV = step.(cell_centers)

    @assert length(u0((getindex.(cell_centers, 1))...)) == N + 2

    # set up initial data
    u = stack(p -> u0(p...), Iterators.product(cell_centers...))
    u_next = zeros(eltype(u), size(u)...)
    n_tsteps = 1
    t = zero(eltype(u))
    n_bytes_written = 0

    if write_result
        if !isdir("data")
            mkdir("data")
        end
        tape_file = joinpath("data", output_tag * ".tape")
    end

    if !write_result && !history_in_memory
        @info "Only the final value of the simulation will be returned." T_end
    end
    u_history = typeof(u)[]
    t_history = eltype(u)[]
    if history_in_memory
        push!(u_history, copy(u))
        push!(t_history, t)
    end
    write_result && open(tape_file, "w+") do f
        n_bytes_written += write(f, zero(Int)) #need this later
        n_bytes_written += write(f, length(ncells), ncells...)
        for b ∈ bounds
            write(f, b...)
        end
        n_bytes_written += write(f, t)
        n_bytes_written += write(f, u)
    end

    if write_result
        tape_stream = open(tape_file, "a+")
    end

    while !(t > T_end || t ≈ T_end) && n_tsteps < max_tsteps
        Δt = try
            maximum_Δt(u, dV, boundary_conditions, cfl_limit, gas)
        catch err
            err isa DomainError || throw(err)
            st = stacktrace(first(current_exceptions()).backtrace)
            is_speed_of_sound = any(st) do f
                f.func == :speed_of_sound
            end
            if is_speed_of_sound
                P_over_ρ = err.val / DRY_AIR.γ
                @error "Non-physical state reached in simulation. 
                    Likely cause is a boundary interaction. Aborting simulation early." P_over_ρ e
            else
                @error "Unexpected DomainError!" err
            end
            break
        end
        Δt = min(Δt, T_end - t)
        if n_tsteps % 10 == 1
            @info "Time step..." n_tsteps t_k = t Δt
        end

        step_euler_hll!(u_next, u, Δt, dV, boundary_conditions, gas)
        t += Δt
        # MONUMENT TO MY STUPIDITY
        u .= u_next
        n_tsteps += 1

        # opening the file is probably trivial compared to writing
        #   one megafloat
        write_result && begin
            n_bytes_written += write(tape_stream, t)
            n_bytes_written += write(tape_stream, u)
        end
        if history_in_memory
            push!(u_history, copy(u))
            push!(t_history, t)
        end
    end

    write_result && begin
        seekstart(tape_stream)
        n_bytes_written += write(tape_stream, n_tsteps)
        seekend(tape_stream)
        close(tape_stream)
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
        reshape(u, size(u)..., 1),
    )
end

function load_euler_sim(path; T = Float64, show_info = true)
    return open(path, "r") do f
        n_tsteps = read(f, Int)
        N = read(f, Int)
        ncells = ntuple(i -> read(f, Int), N)
        bounds = ntuple(i -> (read(f, T), read(f, T)), N)
        t = Vector{T}(undef, n_tsteps)
        u = Array{T,N + 2}(undef, N + 2, ncells..., n_tsteps)
        for i = 1:n_tsteps
            t[i] = read(f, T)
            idxs = (ntuple(i -> Colon(), N + 1)..., i)
            read!(f, @view u[idxs...])
        end

        if show_info
            @info "Loaded Euler eqns. simulation" n_tsteps ncells bounds size(u)
        end
        return EulerSim(ncells, n_tsteps, bounds, t, u)
    end
end