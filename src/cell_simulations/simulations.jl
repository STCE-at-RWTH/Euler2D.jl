@enum EulerSimulationMode::UInt8 begin
    PRIMAL
    TANGENT
end

"""
    CellBasedEulerSim{T, C<:QuadCell}

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
struct CellBasedEulerSim{T,C<:FVMCell}
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
        P[i] = dimensionless_pressure(u, gas)
    end
    return P
end

"""
    pressure_field!(result, csim::CellBasedEulerSim, n, gas)

Compute the dimensionless pressure field for a cell-based Euler simulation `csim` at time step `n` in gas `gas`.

Stores zero if the cell in question is not active.
Stores results in `result`. Returns `result`.
"""
function pressure_field!(
    result,
    csim::CellBasedEulerSim,
    n::Integer,
    gas::CaloricallyPerfectGas,
)
    _, cells = nth_step(csim, n)
    for i ∈ eachindex(result, csim.cell_ids)
        if csim.cell_ids[i] == 0
            result[i] = zero(eltype(result))
        else
            result[i] = dimensionless_pressure(cells[csim.cell_ids[i]].u, gas)
        end
    end
    return result
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

function minimum_cell_size(sim::CellBasedEulerSim)
    _, cells = nth_step(sim, 1)
    return tmapreduce(
        (size_a, size_b) -> min.(size_a, size_b),
        sim.cell_ids;
        init = (Inf, Inf),
    ) do id
        id == 0 && return (Inf, Inf)
        return cells[id].extent
    end
end

function maximum_cell_size(sim::CellBasedEulerSim)
    _, cells = nth_step(sim, 1)
    return tmapreduce(
        (size_a, size_b) -> max.(size_a, size_b),
        sim.cell_ids;
        init = (-Inf, -Inf),
    ) do id
        id == 0 && return (-Inf, -Inf)
        return cells[id].extent
    end
end

function _find_window_on_range(range, window, buffer)
    idx1 = max(
        something(findfirst(>=(window[1]), range), firstindex(range)) - buffer,
        firstindex(range),
    )
    idx2 = min(
        something(findfirst(>=(window[2]), range), lastindex(range)) + buffer,
        lastindex(range),
    )
    return idx1:idx2
end

function _cell_ids_view_from_window(cell_center_ranges, window; buffer = 10)
    return _find_window_on_range.(cell_center_ranges, window, buffer)
end

function _in_window(cell, window_x, window_y)
    return (
        window_x[1] ≤ cell.center[1] ≤ window_x[2] &&
        window_y[1] ≤ cell.center[2] ≤ window_y[2]
    )
end

# I am satisfied that this implementation is correct
function ∇u_at(sim, n, x, y, boundary_conditions, gas; padding = nothing)
    window = if isnothing(padding)
        dx, dy = 2 .* minimum_cell_size(sim)
        (x .+ (-dx, dx), y .+ (-dy, dy))
    else
        (x .+ (-padding[1], padding[2]), y .+ (-padding[2], padding[2]))
    end
    _, cells = nth_step(sim, n)
    slices = _cell_ids_view_from_window(cell_centers(sim), window; buffer = 8)
    contains_point = Iterators.filter(@view sim.cell_ids[slices...]) do id
        return (
            id != 0 &&
            PlanePolygons.point_inside(cell_boundary_polygon(cells[id]), Point(x, y))
        )
    end
    counter = 0
    acc = zero(SMatrix{4,2,Float64,8})
    for id ∈ contains_point
        counter += 1
        nbrs = neighbor_cells(cells[id], cells, boundary_conditions, gas)
        dudx = (nbrs.east.u - nbrs.west.u) / cells[id].extent[1]
        dudy = (nbrs.north.u - nbrs.south.u) / cells[id].extent[2]
        @reset acc += hcat(dudx, dudy)
    end
    return acc / counter
end

"""
    all_cells_contained_by(poly, sim; padding=nothing)

Get all of the cell IDs from `sim` that are contained by `poly`. 
Will compute the window size automatically if `padding` is `nothing`.
"""
function all_cells_contained_by(poly, sim; padding = nothing)
    bbox_x = extrema(p -> p[1], edge_starts(poly))
    bbox_y = extrema(p -> p[2], edge_starts(poly))
    window = if isnothing(padding)
        dx, dy = minimum_cell_size(sim)
        (bbox_x .+ (-dx, dx), bbox_y .+ (-dy, dy))
    else
        (bbox_x .+ (-padding[1], padding[2]), bbox_y .+ (-padding[2], padding[2]))
    end
    _, cells = nth_step(sim, 1)
    slices = _cell_ids_view_from_window(cell_centers(sim), window; buffer = 4)
    return Iterators.filter(@view sim.cell_ids[slices...]) do id
        return (
            id != 0 &&
            _in_window(cells[id], window...) &&
            is_cell_contained_by(cells[id], poly)
        )
    end
end

"""
    all_cells_overlapping(poly, sim; padding=nothing)

Get all of the cell IDs from `sim` that are overlapping `poly` but not contained by `poly`. 
Will compute the window size automatically if `padding` is `nothing`.
"""
function all_cells_overlapping(poly, sim; padding = nothing)
    bbox_x = extrema(p -> p[1], edge_starts(poly))
    bbox_y = extrema(p -> p[2], edge_starts(poly))
    window = if isnothing(padding)
        dx, dy = minimum_cell_size(sim)
        (bbox_x .+ (-dx, dx), bbox_y .+ (-dy, dy))
    else
        (bbox_x .+ (-padding[1], padding[2]), bbox_y .+ (-padding[2], padding[2]))
    end
    _, cells = nth_step(sim, 1)
    slices = _cell_ids_view_from_window(cell_centers(sim), window; buffer = 8)
    return Iterators.filter(@view sim.cell_ids[slices...]) do id
        return (
            id != 0 &&
            _in_window(cells[id], window...) &&
            is_cell_overlapping(cells[id], poly)
        )
    end
end

"""
    total_mass_contained_by(poly, sim, tstep)

Get the TOTAL `(mass, momentum, energy)` contained inside of `poly` at timestep `tstep`. 
"""
function total_mass_contained_by(
    poly,
    sim::CellBasedEulerSim{T,TangentQuadCell{T,NSEEDS,NP}},
    tstep;
    poly_bbox_padding = nothing,
) where {T,NSEEDS,NP}
    contained = all_cells_contained_by(poly, sim; padding = poly_bbox_padding)
    overlapping = all_cells_overlapping(poly, sim; padding = poly_bbox_padding)
    _, cells = nth_step(sim, tstep)
    U_zero = zero(SVector{4,Float64})
    Udot_zero = zero(SMatrix{4,NSEEDS,T,NP})
    U_contained, Udot_contained =
        mapreduce((a, b) -> a .+ b, contained; init = (U_zero, Udot_zero)) do id
            A = cell_volume(cells[id])
            return A .* (cells[id].u, cells[id].u̇)
        end

    U_overlap, Udot_overlap =
        mapreduce((a, b) -> a .+ b, overlapping; init = (U_zero, Udot_zero)) do id
            A = overlapping_cell_area(cells[id], poly)
            return A .* (cells[id].u, cells[id].u̇)
        end

    return U_contained + U_overlap, Udot_contained + Udot_overlap
end

"""
Take the tape data stream and return a channel where simulation state can be queued then pushed.
"""
function _start_output_task(buffer_size, tstep_type, data_stream, empty_buffer_channel)
    taskref = Ref{Task}()
    ch = Channel{Union{Symbol,tstep_type}}(
        buffer_size;
        taskref = taskref,
        spawn = true,
    ) do ch
        while true
            val = take!(ch)
            if val == :stop
                break
            end
            (t, cells) = val
            write_tstep_to_stream(data_stream, t, cells)
            # return the cell buffer so that the main loop can use it
            put!(empty_buffer_channel, cells)
        end
    end
    return taskref, ch
end

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
    maximum_wall_duration = Hour(167),
    write_result = true,
    output_channel_size = 5,
    write_frequency = 1,
    output_tag = "cell_euler_sim",
    show_info = true,
    show_detailed_info = false,
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
        primal_cell_list_and_id_grid(u0, params, bounds, ncells, scale, obstacles)
    else
        tangent_cell_list_and_id_grid(u0, params, bounds, ncells, scale, obstacles)
    end

    n_global_cells = length(global_cells)
    OUTPUT_BUFFER_TYPE = typeof(global_cells)
    # do the partitioning
    cell_partitions = fast_partition_cell_list(
        global_cells,
        global_cell_ids,
        tasks_per_axis;
        show_info = show_detailed_info,
    )
    partition_neighboring = partition_neighbor_map(cell_partitions)
    # if we are writing the result we should make sure there is a file available.
    if write_result
        tape_file = joinpath(pwd(), "data", output_tag * ".celltape")
        tape_path = dirname(tape_file)
        status_file = joinpath(pwd(), "data", output_tag * ".status")
        if !isdir(tape_path)
            @info "Creating data directory/ies at $tape_path"
            mkpath(tape_path)
        end
    else
        @info "Only the final value of the simulation will be available." T_end
    end
    # initialize counters and timer
    n_tsteps = 1
    n_written_tsteps = 1
    t = zero(T)
    wall_clock_start_time = Dates.now()
    previous_tstep_wall_clock = wall_clock_start_time
    avg_duration = (previous_tstep_wall_clock - wall_clock_start_time) ÷ 2
    if show_info
        start_str = Dates.format(wall_clock_start_time, "HH:MM:SS.sss")
        @info "Starting simulation at $start_str" ncells = n_global_cells npartitions =
            length(cell_partitions)
    end
    # open output stream
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

        idle_buffer_channel = Channel{OUTPUT_BUFFER_TYPE}(output_channel_size)
        for _ = 2:output_channel_size
            d = OUTPUT_BUFFER_TYPE()
            sizehint!(d, n_global_cells)
            put!(idle_buffer_channel, d)
        end
        writer_taskref, writer_channel = _start_output_task(
            output_channel_size,
            Tuple{T,OUTPUT_BUFFER_TYPE},
            tape_stream,
            idle_buffer_channel,
        )
        # sacrifice the global_cells buffer ;)
        put!(writer_channel, (t, global_cells))
    end

    # status flag; 0 is "continue"
    time_stepping_status = 0
    # do the time stepping
    while time_stepping_status == 0
        # figure out what the time step size was (after doing it)
        Δt = step_cell_simulation_with_strang_splitting!(
            cell_partitions,
            partition_neighboring,
            T_end - t,
            boundary_conditions,
            cfl_limit,
            gas,
        )

        current_tstep_wall_clock = Dates.now()
        avg_duration = (current_tstep_wall_clock - wall_clock_start_time) ÷ n_tsteps
        if show_info && ((n_tsteps - 1) % info_frequency == 0)
            d = current_tstep_wall_clock - previous_tstep_wall_clock
            remaining = canonicalize(
                maximum_wall_duration - (current_tstep_wall_clock - wall_clock_start_time),
            )
            remaining = Dates.CompoundPeriod(
                remaining.periods[1:max(1, length(remaining.periods) - 2)],
            )
            @info "Time step $n_tsteps (duration $d, avg. $avg_duration, remaining wall clock $remaining)" cur_t =
                t del_t = Δt nex_t = t + Δt
        end
        previous_tstep_wall_clock = current_tstep_wall_clock
        n_tsteps += 1
        t += Δt

        if t > T_end || t ≈ T_end
            # exit status 1 for "finished at t=T"
            time_stepping_status = 1
        elseif n_tsteps >= max_tsteps
            # exit status 2 for "finished at Nt=Nmax"
            time_stepping_status = 2
        elseif (n_tsteps + 4) * avg_duration > maximum_wall_duration
            # exit status 3 for "out of wall clock time"
            time_stepping_status = 3
        end
        # push output to the writer task
        # if we are writing ther result AND
        # the number of time steps is 1 mod write frequency OR
        # time stepping will stop this iteration
        if (
            write_result &&
            (((n_tsteps - 1) % write_frequency == 0) || time_stepping_status != 0)
        )
            put!(
                writer_channel,
                (t, collect_cell_partitions!(take!(idle_buffer_channel), cell_partitions)),
            )
            n_written_tsteps += 1
            if show_info
                @info "Saving simulation state at " k = n_tsteps total_saved =
                    n_written_tsteps
            end
        end
    end
    # stop writing and clean up the output stream
    if write_result
        put!(writer_channel, :stop)
        wait(writer_taskref[])

        seekstart(tape_stream)
        write(tape_stream, n_written_tsteps)
        seekend(tape_stream)
        close(tape_stream)
        open(status_file, "w+") do f
            println(f, time_stepping_status)
        end
    end

    return CellBasedEulerSim(
        (ncells...,),
        1,
        (((first(r), last(r)) for r ∈ cell_ifaces)...,),
        [t],
        global_cell_ids,
        [collect_cell_partitions(cell_partitions, n_global_cells)],
    )
end

"""
  _load_tsteps_from_file

Load the time steps `k ∈ tstep_range` into the provided dict and vector of times `t`.
"""
function _load_tsteps_from_file!(
    cell_vals::Vector{Dict{Int,CellType}},
    ts::Vector{T},
    stream,
    tstep_range,
    n_active;
    show_info = true,
) where {CellType,T}
    step_size_bytes = sizeof(T) + n_active * sizeof(CellType)
    skip_size = diff(vcat(0, tstep_range)) .- 1
    temp = Vector{CellType}(undef, n_active)
    for i ∈ eachindex(tstep_range)
        if skip_size[i] > 0
            skip(stream, skip_size[i] * step_size_bytes)
        end
        ts[i] = read(stream, T)
        if show_info
            #@info "Reading time step at " t = ts[i] skipped = skip_size[i]
        end
        read!(stream, temp)
        cell_vals[i] = Dict{Int,CellType}()
        sizehint!(cell_vals[i], n_active)
        for cell ∈ temp
            cell_vals[i][cell.id] = cell
        end
    end
    return nothing
end

"""
    load_cell_sim(path; steps=:all, T=Float64, show_info=true)

Load a cell-based simulation from path, computed with data type `T`.
Other kwargs include:
- `steps = :all`, or `steps=:last`, or `steps = ` some iterable of values
- `show_info=true`: show metadata via `@info`
"""
function load_cell_sim(path; steps = :all, T = Float64, show_info = true)
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

        (time_steps_to_read, n_tsteps) = if steps == :all
            1:n_tsteps, n_tsteps
        elseif steps == :last
            [n_tsteps], 1
        else
            steps, length(steps)
        end

        CellDType = if mode == PRIMAL
            PrimalQuadCell{T}
        else
            TangentQuadCell{T,n_seeds,4 * n_seeds}
        end

        if show_info
            @info "Preparing to load from file:" DType = CellDType steps =
                time_steps_to_read
        end

        time_steps = Vector{T}(undef, n_tsteps)
        cell_vals = Vector{Dict{Int,CellDType}}(undef, n_tsteps)
        _load_tsteps_from_file!(
            cell_vals,
            time_steps,
            f,
            time_steps_to_read,
            n_active;
            show_info = show_info,
        )

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

_sim_mode(::Type{TangentQuadCell{T,N1,N2}}) where {T,N1,N2} = TANGENT
_sim_mode(::Type{PrimalQuadCell{T}}) where {T} = PRIMAL

function write_cell_sim(path, sim::CellBasedEulerSim{T,C}) where {T,C}
    return open(path, "w") do f
        # how many time steps
        # what is the mode
        write(f, sim.nsteps, _sim_mode(C))
        if _sim_mode(C) == TANGENT
            # if the mode is TANGENT, how many seeds
            write(f, n_seeds(C))
        end
        # how many active cells are in each time step
        write(f, length(first(sim.cells)))
        # ndimensions & specific size
        write(f, length(sim.ncells), sim.ncells...)
        foreach(sim.bounds) do b
            write(f, b...)
        end
        write(f, sim.cell_ids)
        foreach(zip(sim.tsteps, sim.cells)) do (t, c)
            write_tstep_to_stream(f, t, c)
        end
    end
end
