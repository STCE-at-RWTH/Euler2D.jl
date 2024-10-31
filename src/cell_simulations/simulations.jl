
"""
    CellBasedEulerSim{T, Q1<:Density, Q2<:MomentumDensity, Q3<:EnergyDensity}

Represents a completed simulation of the Euler equations on a mesh of 2-dimensional quadrilateral cells.

## Type Parameters
- `T`: Data type for all computations.
- `Q1, Q2, Q3`: Quantity types for density, momentum density, and energy density.

## Fields
- `ncells::(Int, Int)`: Number of cells in each of of the simulation density.
- `nsteps::Int`: Number of time steps taken in the simulation
- `bounds::{(T, T), (T, T)}`: Bounds of the simulation in each of the dimensions.
- `tsteps::Vector{T}`: Time steps.
- `cell_ids::Matrix{Int}`: Matrix of active cell IDs. Inactive cells
- `u::Array{ConservedProps{T, Q1, Q2, Q3}, 2}`: Array of cell data in each time step.

## Methods
- `n_space_dims`, `n_tsteps``, ``grid_size``
- `cell_centers`: Get the co-ordinates of cell centers.
- `cell_boundaries`: Get the co-ordinates of cell faces.
- `nth_step`: Get the information at time step `n`.
- `eachstep`: Get a vector of `(t_k, u_k)` tuples for easy iteration.
- `density_field, momentum_density_field, total_internal_energy_density_field`: Compute quantities at a given time step.
- `pressure_field, velocity_field, mach_number_field`: Compute quantities at a given time step.
"""
struct CellBasedEulerSim{T,Q1,Q2,Q3}
    ncells::Tuple{Int,Int}
    nsteps::Int
    bounds::NTuple{2,Tuple{T,T}}
    tsteps::Vector{T}
    cell_ids::Array{Int,2}
    cells::Array{Dict{Int64,PrimalQuadCell{T,Q1,Q2,Q3}},1}
end

n_space_dims(::CellBasedEulerSim) = 2

function cell_boundaries(e::CellBasedEulerSim)
    return ntuple(i -> cell_boundaries(e, i), 2)
end

function cell_centers(e::CellBasedEulerSim)
    return ntuple(i -> cell_centers(e, i), 2)
end

"""
    nth_step(csim::CellBasedEulerSim, n)

Return `(t, cells)` for time step `n`. `cells` will be a view.
"""
function nth_step(csim::CellBasedEulerSim, n)
    return csim.tsteps[n], csim.cells[n]
end

eachstep(csim::CellBasedEulerSim) = [nth_step(csim, n) for n ∈ 1:n_tsteps(csim)]

"""
    density_field(csim::CellBasedEulerSim, n)

Compute the density field for a cell-based Euler simulation `csim` at time step `n`.
"""
function density_field(csim::CellBasedEulerSim{T,Q1,Q2,Q3}, n) where {T,Q1,Q2,Q3}
    _, u_cells = nth_step(csim, n)
    ρ = Array{Union{Q1,Nothing},2}(undef, grid_size(csim))
    fill!(ρ, nothing)
    for i ∈ eachindex(IndexCartesian(), csim.cell_ids)
        csim.cell_ids[i] == 0 && continue
        ρ[i] = density(u_cells[csim.cell_ids[i]].u)
    end
    return ρ
end

"""
    momentum_density_field(csim::CellBasedEulerSim, n)

Compute the momentum density field for a cell-based Euler simulation `csim` at time step `n`.
"""
function momentum_density_field(csim::CellBasedEulerSim{T,Q1,Q2,Q3}, n) where {T,Q1,Q2,Q3}
    _, u_cells = nth_step(csim, n)
    ρv = Array{Union{Q2,Nothing},3}(undef, (2, grid_size(csim)...))
    fill!(ρv, nothing)
    for i ∈ eachindex(IndexCartesian(), csim.cell_ids)
        csim.cell_ids[i] == 0 && continue
        ρv[:, i] = momentum_density(u_cells[csim.cell_ids[i]].u)
    end
    return ρv
end

"""
    velocity_field(csim::CellBasedEulerSim, n)

Compute the velocity field for a cell-based Euler simulation `csim` at time step `n`.
"""
function velocity_field(csim::CellBasedEulerSim, n)
    _, u_cells = nth_step(csim, n)
    # is this a huge runtime problem? who can know.
    T = eltype(velocity(u_cells[1].u))
    v = Array{Union{T,Nothing},3}(undef, (2, grid_size(csim)...))
    fill!(v, nothing)
    for i ∈ eachindex(IndexCartesian(), csim.cell_ids)
        csim.cell_ids[i] == 0 && continue
        v[:, i] = velocity(u_cells[csim.cell_ids[i]].u)
    end
    return v
end

"""
    total_internal_energy_density_field(csim::CellBasedEulerSim, n)

Compute the total internal energy density field for a cell-based Euler simulation at time step `n`.
"""
function total_internal_energy_density_field(
    csim::CellBasedEulerSim{T,Q1,Q2,Q3},
    n,
) where {T,Q1,Q2,Q3}
    _, u_cells = nth_step(csim, n)
    ρE = Array{Union{Q3,nothing},2}(undef, grid_size(csim))
    fill!(ρE, nothing)
    for i ∈ eachindex(IndexCartesian(), csim.cell_ids)
        csim.cell_ids[i] == 0 && continue
        ρE[i] = total_internal_energy_density(u_cells[csim.cell_ids[i]].u)
    end
    return ρE
end

"""
    pressure_field(csim::CellBasedEulerSim, n, gas)

Compute the pressure field for a cell-based Euler simulation `csim` at time step `n` in gas `gas`.
"""
function pressure_field(csim::CellBasedEulerSim, n, gas::CaloricallyPerfectGas)
    _, u_cells = nth_step(csim, n)
    # is this a huge runtime problem? who can know.
    T = typeof(pressure(u_cells[1].u, gas))
    P = Array{Union{T,Nothing},2}(undef, grid_size(csim))
    fill!(P, nothing)
    for i ∈ eachindex(IndexCartesian(), csim.cell_ids)
        csim.cell_ids[i] == 0 && continue
        P[i] = pressure(u_cells[csim.cell_ids[i]].u, gas)
    end
    return P
end

"""
    mach_number_field(csim::CellBasedEulerSim, n, gas)

Compute the Mach number field for a cell-based Euler simulation `csim` at time step `n` in gas `gas`.
"""
function mach_number_field(
    csim::CellBasedEulerSim{T,Q1,Q2,Q3},
    n,
    gas::CaloricallyPerfectGas,
) where {T,Q1,Q2,Q3}
    _, u_cells = nth_step(csim, n)
    # is this a huge runtime problem? who can know.
    M = Array{Union{T,Nothing},3}(undef, (2, grid_size(csim)...))
    fill!(M, nothing)
    for i ∈ eachindex(IndexCartesian(), csim.cell_ids)
        csim.cell_ids[i] == 0 && continue
        M[:, i] = mach_number(u_cells[csim.cell_ids[i]].u, gas)
    end
    return M
end

function write_tstep_to_stream(stream, t, global_cells)
    @info "Writing time step to file." t = t ncells=length(global_cells)
    write(stream, t)
    for (id, cell) ∈ global_cells
        @assert id == cell.id
        write(stream, Ref(cell))
    end
end

"""
    simulate_euler_equations_cells(u0, T_end, boundary_conditions, bounds, ncells)

Simulate the solution to the Euler equations from `t=0` to `t=T`, with `u(0, x) = u0(x)`.
Time step size is computed from the CFL condition.

The simulation will fail if any nonphysical conditions are reached (speed of sound cannot be computed).

The simulation can be written to disk.

Arguments
---
- `u0`: ``u(t=0, x, y):ℝ^2↦ConservedProps{2, T, ...}``: conditions at time `t=0`.
- `T_end`: Must be greater than zero.
- `boundary_conditions`: a tuple of boundary conditions for each space dimension
- `obstacles`: list of obstacles in the flow.
- `bounds`: a tuple of extents for each space dimension (tuple of tuples)
- `ncells`: a tuple of cell counts for each dimension

Keyword Arguments
---
- `gas::CaloricallyPerfectGas = DRY_AIR`: The fluid to be simulated.
- `cfl_limit = 0.75`: The CFL condition to apply to `Δt`. Between zero and one, default `0.75`.
- `max_tsteps=typemax(Int)`: Maximum number of time steps to take. Defaults to "very large".
- `write_result = true`: Should output be written to disk?
- `output_channel_size = 5`: How many time steps should be buffered during I/O?
- `write_frequency = 1`: How often should time steps be written out?
- `history_in_memory = false`: Should we keep whole history in memory?
- `output_tag = "cell_euler_sim"`: File name for the tape and output summary.
- `show_info = true` : Should diagnostic information be printed out?
- `info_frequency = 10`: How often should info be printed?
- `tasks_per_axis = Threads.nthreads()`: How many partitions should be created on each axis?
"""
function simulate_euler_equations_cells(
    u0,
    T_end,
    boundary_conditions,
    obstacles,
    bounds,
    ncells;
    gas::CaloricallyPerfectGas = DRY_AIR,
    cfl_limit = 0.75,
    max_tsteps = typemax(Int),
    write_result = true,
    output_channel_size = 5,
    write_frequency = 1,
    history_in_memory = false,
    output_tag = "cell_euler_sim",
    show_info = true,
    info_frequency = 10,
    tasks_per_axis = Threads.nthreads(),
)
    N = length(ncells)
    T = typeof(T_end)
    @assert N == 2
    @assert length(bounds) == 2
    @assert length(boundary_conditions) == 5
    T_end > 0 && DomainError("T_end = $T_end invalid, T_end must be positive")
    0 < cfl_limit < 1 || @warn "CFL invalid, must be between 0 and 1 for stabilty" cfl_limit
    cell_ifaces = [range(b...; length = n + 1) for (b, n) ∈ zip(bounds, ncells)]
    global_cells, global_cell_ids = quadcell_list_and_id_grid(u0, bounds, ncells, obstacles)
    cell_partitions = partition_cell_list(global_cells, global_cell_ids, tasks_per_axis)
    show_info &&
        @info "Simulation starting information " ncells = length(global_cells) npartitions =
            length(cell_partitions)

    n_tsteps = 1
    n_written_tsteps = 1
    t = zero(T)
    if write_result
        if !isdir("data")
            @info "Creating directory at" dir = joinpath(pwd(), "data")
            mkdir("data")
        end
        tape_file = joinpath("data", output_tag * ".celltape")
    end
    if !write_result && !history_in_memory
        @info "Only the final value of the simulation will be available." T_end
    end

    u_history = typeof(global_cells)[]
    t_history = typeof(t)[]
    if (history_in_memory)
        push!(u_history, copy(global_cells))
        push!(t_history, t)
    end

    if write_result
        tape_stream = open(tape_file, "w+")
        write(tape_stream, zero(Int), length(global_cells), length(ncells), ncells...)
        for b ∈ bounds
            write(tape_stream, b...)
        end
        write(tape_stream, global_cell_ids)
        # TODO we would like to preallocate buffers here...
        writer_taskref = Ref{Task}()
        writer_channel = Channel{Union{Symbol,Tuple{T,typeof(global_cells)}}}(
            output_channel_size;
            taskref = writer_taskref,
        ) do ch
            while true
                val = take!(ch)
                if val == :stop
                    break
                end
                write_tstep_to_stream(tape_stream, val...)
            end
        end
        put!(writer_channel, (t, global_cells))
    end

    while !(t > T_end || t ≈ T_end) && n_tsteps < max_tsteps
        Δt = step_cell_simulation!(
            cell_partitions,
            T_end - t,
            boundary_conditions,
            cfl_limit,
            gas,
        )
        if show_info && ((n_tsteps - 1) % info_frequency == 0)
            @info "Time step..." k = n_tsteps t_k = t Δt t_next = t + Δt
        end
        n_tsteps += 1
        t += Δt
        if (
            ((n_tsteps - 1) % write_frequency == 0 || n_tsteps == max_tsteps) &&
            (write_result || history_in_memory)
        )
            if write_result && history_in_memory
                u_cur = collect_cell_partitions(cell_partitions, global_cell_ids)
                put!(writer_channel, (t, u_cur))
                push!(u_history, u_cur)
                push!(t_history, t)
            elseif write_result
                put!(
                    writer_channel,
                    (t, collect_cell_partitions(cell_partitions, global_cell_ids)),
                )
            elseif history_in_memory
                push!(u_history, collect_cell_partitions(cell_partitions, global_cell_ids))
                push!(t_history, t)
            end
            n_written_tsteps += 1
            if show_info
                @info "Saving simulation state at " k = n_tsteps total_saved =
                    n_written_tsteps
            end
        end
    end

    if write_result
        put!(writer_channel, :stop)
        wait(writer_taskref[])

        seekstart(tape_stream)
        write(tape_stream, n_written_tsteps)
        seekend(tape_stream)
        close(tape_stream)
    end

    if history_in_memory
        @assert n_written_tsteps == length(t_history)
        return CellBasedEulerSim{T}(
            (ncells...,),
            n_written_tsteps,
            (((first(r), last(r)) for r ∈ cell_ifaces)...),
            t_history,
            global_cell_ids,
            u_history,
        )
    end

    return CellBasedEulerSim(
        (ncells...,),
        1,
        (((first(r), last(r)) for r ∈ cell_ifaces)...,),
        [t],
        global_cell_ids,
        [collect_cell_partitions(cell_partitions, global_cell_ids)],
    )
end

"""
    load_cell_sim(path; T=Float64, show_info=true)

Load a cell-based simulation from path, computed with data type `T`.
Other kwargs include:
- `density_unit = ShockwaveProperties._units_ρ`
- `momentum_density_unit = ShockwaveProperties._units_ρv`
- `internal_energy_density_unit = ShockwaveProperties._units_ρE`
"""
function load_cell_sim(
    path;
    T = Float64,
    density_unit = ShockwaveProperties._units_ρ,
    momentum_density_unit = ShockwaveProperties._units_ρv,
    internal_energy_density_unit = ShockwaveProperties._units_ρE,
    show_info = true,
)
    U =
        typeof.(
            one(T) .* (density_unit, momentum_density_unit, internal_energy_density_unit)
        )
    CellDType = PrimalQuadCell{T,U...}
    return open(path, "r") do f
        n_tsteps = read(f, Int)
        n_active = read(f, Int)
        n_dims = read(f, Int)
        @assert n_dims == 2
        ncells = (read(f, Int), read(f, Int))
        bounds = ntuple(i -> (read(f, T), read(f, T)), 2)
        if show_info
            @info "Loaded metadata for cell-based Euler simulation at $path." n_tsteps n_active n_dims ncells
        end
        active_cell_ids = Array{Int,2}(undef, ncells...)
        read!(f, active_cell_ids)
        time_steps = Vector{T}(undef, n_tsteps)
        cell_vals = Vector{Dict{Int,CellDType}}(undef, n_tsteps)
        temp_cell_vals = Vector{CellDType}(undef, n_active)
        for k = 1:n_tsteps
            time_steps[k] = read(f, T)
            #@info "Reading..." k t_k = time_steps[k] n_active
            read!(f, temp_cell_vals)
            cell_vals[k] = Dict{Int,CellDType}()
            sizehint!(cell_vals[k], n_active)
            for cell ∈ temp_cell_vals
                cell_vals[k][cell.id] = cell
            end
        end

        return CellBasedEulerSim(
            ncells,
            n_tsteps,
            bounds,
            time_steps,
            active_cell_ids,
            cell_vals,
        )
    end
end