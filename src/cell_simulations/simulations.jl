@enum EulerSimulationMode::UInt8 begin
    PRIMAL
    TANGENT
end

"""
    CellBasedEulerSim{T}

Represents a completed simulation of the Euler equations on a mesh of 2-dimensional quadrilateral cells.

## Type Parameters
- `T`: Data type for all computations.
- `C`: Cell data type

## Fields
- `ncells::(Int, Int)`: Number of cells in each of of the simulation axes.
- `nsteps::Int`: Number of time steps taken in the simulation.
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
struct CellBasedEulerSim{T,C<:QuadCell}
    ncells::Tuple{Int,Int}
    nsteps::Int
    bounds::NTuple{2,Tuple{T,T}}
    tsteps::Vector{T}
    cell_ids::Array{Int,2}
    cells::Array{Dict{Int64,C},1}
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
function density_field(csim::CellBasedEulerSim{T}, n::Integer) where {T}
    _, u_cells = nth_step(csim, n)
    ρ = Array{Union{T,Nothing},2}(undef, grid_size(csim))
    fill!(ρ, nothing)
    for i ∈ eachindex(IndexCartesian(), csim.cell_ids)
        csim.cell_ids[i] == 0 && continue
        u = u_cells[csim.cell_ids[i]].u
        ρ[i] = u[1]
    end
    return ρ
end

"""
    density_field(csim::CellBasedEulerSim, scale, n)

Compute the density field for a cell-based Euler simulation `csim` and redimensionalize it at time step `n`.
"""
function density_field(
    csim::CellBasedEulerSim{T},
    n::Integer,
    scale::EulerEqnsScaling,
) where {T}
    return map(density_field(csim, n)) do ρ
        isnothing(ρ) && return nothing
        return ρ * density_scale(scale)
    end
end

"""
    momentum_density_field(csim::CellBasedEulerSim, n)

Compute the dimensionless momentum density field for a cell-based Euler simulation `csim` at time step `n`.
"""
function momentum_density_field(csim::CellBasedEulerSim{T}, n::Integer) where {T}
    _, u_cells = nth_step(csim, n)
    ρv = Array{T,3}(undef, (2, grid_size(csim)...))
    fill!(ρv, nothing)
    for i ∈ eachindex(IndexCartesian(), csim.cell_ids)
        csim.cell_ids[i] == 0 && continue
        u = u_cells[csim.cell_ids[i]].u
        ρv[:, i] = select_middle(u)
    end
    return ρv
end

"""
    momentum_density_field(csim::CellBasedEulerSim, scale, n)

Compute the momentum density field for a cell-based Euler simulation `csim` and redimensionalize it at time step `n`.
"""
function momentum_density_field(
    csim::CellBasedEulerSim{T},
    n::Integer,
    scale::EulerEqnsScaling,
) where {T}
    return map(momentum_density_field(csim, n)) do ρv
        isnothing(ρv) && return nothing
        return ρv * density_scale(scale) * velocity_scale(scale)
    end
end

"""
    velocity_field(csim::CellBasedEulerSim, n)

Compute the dimensionless velocity field for a cell-based Euler simulation `csim` at time step `n`.
"""
function velocity_field(csim::CellBasedEulerSim{T}, n::Integer) where {T}
    _, u_cells = nth_step(csim, n)
    v = Array{Union{T,Nothing},3}(undef, (2, grid_size(csim)...))
    fill!(v, nothing)
    for i ∈ eachindex(IndexCartesian(), csim.cell_ids)
        csim.cell_ids[i] == 0 && continue
        u = u_cells[csim.cell_ids[i]].u
        v[:, i] = select_middle(u) / first(u)
    end
    return v
end

"""
    velocity_field(csim::CellBasedEulerSim, scale, n)

Compute the velocity field for a cell-based Euler simulation `csim` and redimensionalize it at time step `n`.
"""
function velocity_field(
    csim::CellBasedEulerSim{T},
    n::Integer,
    scale::EulerEqnsScaling,
) where {T}
    return map(velocity_field(csim, n)) do v
        isnothing(v) && return nothing
        return v * velocity_scale(scale)
    end
end

"""
    total_internal_energy_density_field(csim::CellBasedEulerSim, n)

Compute the dimensionless total internal energy density field for a cell-based Euler simulation at time step `n`.
"""
function total_internal_energy_density_field(
    csim::CellBasedEulerSim{T},
    n::Integer,
) where {T}
    _, u_cells = nth_step(csim, n)
    ρE = Array{Union{T,Nothing},2}(undef, grid_size(csim))
    fill!(ρE, nothing)
    for i ∈ eachindex(IndexCartesian(), csim.cell_ids)
        csim.cell_ids[i] == 0 && continue
        u = u_cells[csim.cell_ids[i]].u
        ρE[i] = last(u)
    end
    return ρE
end

"""
    total_internal_energy_density_field(csim::CellBasedEulerSim, scale, n)

Compute the total internal energy density field for a cell-based Euler simulation `csim` and redimensionalize it at time step `n`.
"""
function total_internal_energy_density_field(
    csim::CellBasedEulerSim{T},
    n::Integer,
    scale::EulerEqnsScaling,
) where {T}
    return map(total_internal_energy_density_field(csim, n)) do ρE
        isnothing(ρE) && return nothing
        return ρE * energy_density_scale(scale)
    end
end

"""
    pressure_field(csim::CellBasedEulerSim, n, gas)

Compute the dimensionless pressure field for a cell-based Euler simulation `csim` at time step `n` in gas `gas`.
"""
function pressure_field(
    csim::CellBasedEulerSim{T},
    n::Integer,
    gas::CaloricallyPerfectGas,
) where {T}
    _, u_cells = nth_step(csim, n)
    P = Array{Union{T,Nothing},2}(undef, grid_size(csim))
    fill!(P, nothing)
    for i ∈ eachindex(IndexCartesian(), csim.cell_ids)
        csim.cell_ids[i] == 0 && continue
        u = u_cells[csim.cell_ids[i]].u
        P[i] = _pressure(u, gas)
    end
    return P
end

"""
    pressure_field(csim::CellBasedEulerSim, scale, n)

Compute the total internal energy density field for a cell-based Euler simulation `csim` and redimensionalize it at time step `n`.
"""
function pressure_field(
    csim::CellBasedEulerSim{T},
    n::Integer,
    gas::CaloricallyPerfectGas,
    scale::EulerEqnsScaling,
) where {T}
    return map(pressure_field(csim, n, gas)) do P
        isnothing(P) && return nothing
        return P * pressure_scale(scale)
    end
end

"""
    mach_number_field(csim::CellBasedEulerSim, n, gas)

Compute the Mach number field for a cell-based Euler simulation `csim` at time step `n` in gas `gas`.
"""
function mach_number_field(
    csim::CellBasedEulerSim{T},
    n::Integer,
    gas::CaloricallyPerfectGas,
) where {T}
    _, u_cells = nth_step(csim, n)
    # is this a huge runtime problem? who can know.
    M = Array{Union{T,Nothing},3}(undef, (2, grid_size(csim)...))
    fill!(M, nothing)
    for i ∈ eachindex(IndexCartesian(), csim.cell_ids)
        csim.cell_ids[i] == 0 && continue
        u = u_cells[csim.cell_ids[i]].u
        a = dimensionless_speed_of_sound(u, gas)
        M[:, i] = select_middle(u) / (first(u) * a)
    end
    return M
end

# for completeness
mach_number_field(
    csim::CellBasedEulerSim,
    n::Integer,
    gas::CaloricallyPerfectGas,
    scale::EulerEqnsScaling,
) = mach_number_field(csim, n, gas)

function write_tstep_to_stream(stream, t, global_cells)
    @info "Writing time step to file." t = t ncells = length(global_cells)
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
- `u0`: ``u(t=0, x, p):ℝ^2×ℝ^{n_p}↦ConservedProps{2, T, ...}``: conditions at time `t=0`.
- `params`: Parameter vector for `u0`.
- `T_end`: Must be greater than zero.
- `boundary_conditions`: a tuple of boundary conditions for each space dimension
- `obstacles`: list of obstacles in the flow.
- `bounds`: a tuple of extents for each space dimension (tuple of tuples)
- `ncells`: a tuple of cell counts for each dimension

Keyword Arguments
---
- `mode::EulerSimulationMode = PRIMAL`: `PRIMAL` or `TANGENT` 
- `gas::CaloricallyPerfectGas = DRY_AIR`: The fluid to be simulated.
- `scale::EulerEqnsScaling = _SI_DEFAULT_SCALE`: A set of non-dimensionalization parameters.
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
    params,
    T_end,
    boundary_conditions,
    obstacles,
    bounds,
    ncells;
    mode::EulerSimulationMode = PRIMAL,
    gas::CaloricallyPerfectGas = DRY_AIR,
    scale::EulerEqnsScaling = _SI_DEFAULT_SCALE,
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

    global_cells, global_cell_ids = if mode == PRIMAL
        primal_quadcell_list_and_id_grid(u0, params, bounds, ncells, scale, obstacles)
    else
        tangent_quadcell_list_and_id_grid(u0, params, bounds, ncells, scale, obstacles)
    end

    cell_partitions = partition_cell_list(global_cells, global_cell_ids, tasks_per_axis)
    wall_clock_start_time = Dates.now()
    previous_tstep_wall_clock = wall_clock_start_time
    if show_info
        start_str = Dates.format(wall_clock_start_time, "HH:MM:SS.sss")
        @info "Starting simulation at $start_str" ncells = length(global_cells) npartitions =
            length(cell_partitions)
    end

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
        write(tape_stream, zero(Int), mode)
        if mode == TANGENT
            write(tape_stream, n_seeds(valtype(global_cells)))
        end
        write(tape_stream, length(global_cells), length(ncells), ncells...)
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
        current_tstep_wall_clock = Dates.now()
        if show_info && ((n_tsteps - 1) % info_frequency == 0)
            d = current_tstep_wall_clock - previous_tstep_wall_clock
            avg_duration = (current_tstep_wall_clock - wall_clock_start_time) ÷ n_tsteps
            @info "Time step $n_tsteps (duration $d, avg. $avg_duration)" t_k = t Δt t_next =
                t + Δt
        end
        previous_tstep_wall_clock = current_tstep_wall_clock
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
function load_cell_sim(path; T = Float64, show_info = true)
    return open(path, "r") do f
        n_tsteps = read(f, Int)
        mode = read(f, EulerSimulationMode)
        n_seeds = if mode == TANGENT
            read(f, Int)
        else
            0
        end
        n_active = read(f, Int)
        n_dims = read(f, Int)
        @assert n_dims == 2
        ncells = (read(f, Int), read(f, Int))
        bounds = ntuple(i -> (read(f, T), read(f, T)), 2)
        if show_info
            @info "Loaded metadata for cell-based Euler simulation at $path." mode n_seeds n_tsteps n_active n_dims ncells
        end
        active_cell_ids = Array{Int,2}(undef, ncells...)
        read!(f, active_cell_ids)
        time_steps = Vector{T}(undef, n_tsteps)
        CellDType = if mode == PRIMAL
            PrimalQuadCell{T}
        else
            TangentQuadCell{T,n_seeds,4 * n_seeds}
        end
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