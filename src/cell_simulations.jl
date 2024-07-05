# I always think in "north south east west"... who knows why.
#   anyway
@enum CellBoundaries::Int begin
    NORTH_BOUNDARY = 1
    SOUTH_BOUNDARY = 2
    EAST_BOUNDARY = 3
    WEST_BOUNDARY = 4
    INTERNAL_STRONGWALL = 5
end

@enum CellNeighboring::Int begin
    OTHER_QUADCELL
    BOUNDARY_CONDITION
end

struct RegularQuadCell{T,Q1<:Density,Q2<:MomentumDensity,Q3<:EnergyDensity}
    id::Int
    idx::CartesianIndex{2}
    center::SVector{2,T}
    u::ConservedProps{2,T,Q1,Q2,Q3}
    # either (:boundary, :cell)
    # and then the ID of the appropriate boundary
    neighbors::NamedTuple{
        (:north, :south, :east, :west),
        NTuple{4,Tuple{CellNeighboring,Int}},
    }
end

function Base.convert(
    ::Type{RegularQuadCell{T,A1,A2,A3}},
    cell::RegularQuadCell{T,B1,B2,B3},
) where {T,A1,A2,A3,B1,B2,B3}
    return RegularQuadCell(
        cell.id,
        cell.idx,
        cell.center,
        convert(ConservedProps{2,T,A1,A2,A3}, cell.u),
        cell.neighbors,
    )
end

dtype(::RegularQuadCell{T,Q1,Q2,Q3}) where {T,Q1,Q2,Q3} = T

function inward_normals(T)
    return (
        north = SVector((zero(T), -one(T))...),
        south = SVector((zero(T), one(T))...),
        east = SVector((-one(T), zero(T))...),
        west = Svector((one(T), zero(T))...),
    )
end

function outward_normals(T)
    return (
        north = SVector((zero(T), one(T))...),
        south = SVector((zero(T), -one(T))...),
        east = SVector((one(T), zero(T))...),
        west = Svector((-one(T), zero(T))...),
    )
end

inward_normals(c::RegularQuadCell) = inward_normals(dtype(c))
outward_normals(c::RegularQuadCell) = outward_normals(dtype(c))

props_dtype(::ConservedProps{N,T,U1,U2,U3}) where {N,T,U1,U2,U3} = T
props_unitstypes(::ConservedProps{N,T,U1,U2,U3}) where {N,T,U1,U2,U3} = (U1, U2, U3)

function cprops_dtype(::RegularQuadCell{T,Q1,Q2,Q3}) where {T,Q1,Q2,Q3}
    return ConservedProps{2,T,Q1,Q2,Q3}
end

function cprops_dtype(::Type{RegularQuadCell{T,Q1,Q2,Q3}}) where {T,Q1,Q2,Q3}
    return ConservedProps{2,T,Q1,Q2,Q3}
end

function quadcell_dtype(::ConservedProps{N,T,U1,U2,U3}) where {N,T,U1,U2,U3}
    return RegularQuadCell{T,U1,U2,U3}
end

abstract type Obstacle end

struct CircularObstacle{T} <: Obstacle
    center::SVector{2,T}
    radius::T
end

function CircularObstacle(center, radius)
    CircularObstacle(SVector{2}(center...), radius)
end

function point_inside(s::CircularObstacle, pt)
    Δr = pt - s.center
    return sum(x -> x^2, Δr) <= s.radius^2
end

struct RectangularObstacle{T} <: Obstacle
    center::SVector{2,T}
    extent::SVector{2,T}
end

function point_inside(s::RectangularObstacle, pt)
    Δx = pt - s.center
    return all(abs.(Δx) .<= s.extent)
end

struct TriangularObstacle{T} <: Obstacle
    points::NTuple{3,SVector{2,T}}
end

function TriangularObstacle(pts)
    return TriangularObstacle(tuple((SVector{2}(p) for p ∈ pts)...))
end

function point_inside(s::TriangularObstacle, pt)
    return all(zip(s.points, s.points[[2, 3, 1]])) do (p1, p2)
        (SMatrix{2,2}(0, -1, 1, 0) * (p2 - p1)) ⋅ (pt - p1) > 0
    end
end

# TODO we should actually be more serious about compting these overlaps
#  and then computing volume-averaged quantities
point_inside(s::Obstacle, q::RegularQuadCell) = point_inside(s, q.center)

function active_cell_mask(cell_centers_x, cell_centers_y, obstacles)
    return map(Iterators.product(cell_centers_x, cell_centers_y)) do (x, y)
        p = SVector{2}(x, y)
        return all(obstacles) do o
            !point_inside(o, p)
        end
    end
end

function active_cell_ids_from_mask(active_mask)
    cell_ids = zeros(Int, size(active_mask))
    live_count = 0
    for i ∈ eachindex(IndexLinear(), active_mask, cell_ids)
        live_count += active_mask[i]
        if active_mask[i]
            cell_ids[i] = live_count
        end
    end
    return cell_ids
end

function cell_neighbor_status(i, cell_ids, active_mask)
    idx = CartesianIndices(size(cell_ids))[i]
    _cell_neighbor_offsets = (
        north = CartesianIndex(0, 1),
        south = CartesianIndex(0, -1),
        east = CartesianIndex(1, 0),
        west = CartesianIndex(-1, 0),
    )
    map(_cell_neighbor_offsets) do offset
        neighbor = idx + offset
        if neighbor[1] < 1
            return (BOUNDARY_CONDITION, Int(WEST_BOUNDARY))
        elseif neighbor[1] > size(cell_ids)[1]
            return (BOUNDARY_CONDITION, Int(EAST_BOUNDARY))
        elseif neighbor[2] < 1
            return (BOUNDARY_CONDITION, Int(SOUTH_BOUNDARY))
        elseif neighbor[2] > size(cell_ids)[2]
            return (BOUNDARY_CONDITION, Int(NORTH_BOUNDARY))
        elseif active_mask[neighbor]
            return (OTHER_QUADCELL, cell_ids[neighbor])
        else
            return (BOUNDARY_CONDITION, Int(INTERNAL_STRONGWALL))
        end
    end
end

function quadcell_list_and_id_grid(u0, bounds, ncells, obstacles)
    centers = map(zip(bounds, ncells)) do (b, n)
        v = range(b...; length = n + 1)
        return v[1:end-1] .+ step(v) / 2
    end

    # u0 is probably cheap
    u0_grid = map(u0, Iterators.product(centers...))
    active_mask = active_cell_mask(centers..., obstacles)
    active_ids = active_cell_ids_from_mask(active_mask)
    @assert sum(active_mask) == last(active_ids)

    cell_list = Vector{quadcell_dtype(first(u0_grid))}(undef, sum(active_mask))
    for i ∈ eachindex(IndexCartesian(), active_ids, active_mask)
        active_mask[i] || continue
        j = active_ids[i]
        (m, n) = Tuple(i)
        x_i = centers[1][m]
        y_j = centers[2][n]
        neighbors = cell_neighbor_status(i, active_ids, active_mask)
        cell_list[j] = RegularQuadCell(j, i, SVector(x_i, y_j), u0_grid[i], neighbors)
    end
    return cell_list, active_ids
end

function phantom_neighbor(id, active_cells, dir, bc, gas)
    # TODO use nneighbors as intended.
    @assert nneighbors(bc) == 1 "dirty hack alert, this function needs to be extended for bcs with more neighbors"

    dirs_reverse_bc = (north = true, south = false, east = true, west = false)
    dirs_dim = (north = 2, south = 2, east = 1, west = 1)

    reverse_phantom = dirs_reverse_bc[dir] && reverse_right_edge(bc)
    phantom = phantom_cell(bc, active_cells[id].u, dirs_dim[dir], gas)
    if reverse_phantom
        return flip_velocity(phantom, dirs_dim[dir])
    end
    return phantom
end

# FIXME this allocates
function single_cell_neighbor_data(
    cell_id,
    active_cells,
    boundary_conditions,
    gas::CaloricallyPerfectGas,
)
    neighbors = active_cells[cell_id].neighbors
    cells_view = @view active_cells[:]
    map((ntuple(i -> ((keys(neighbors)[i], neighbors[i])), 4))) do (dir, (kind, id))
        if kind == BOUNDARY_CONDITION
            return phantom_neighbor(id, cells_view, dir, boundary_conditions[id], gas)
        else
            return cells_view[id].u
        end
    end |> NamedTuple{(:north, :south, :east, :west)}
end

# accepts SVectors!
function compute_cell_update(cell_data, neighbor_data, Δx, Δy, gas)
    ifaces = (
        north = (2, cell_data, neighbor_data.north),
        south = (2, neighbor_data.south, cell_data),
        east = (1, cell_data, neighbor_data.east),
        west = (1, neighbor_data.west, cell_data),
    )

    maximum_signal_speed = mapreduce(max, ifaces) do (dim, uL, uR)
        max(abs.(interface_signal_speeds(uL, uR, dim, gas))...)
    end

    ϕ = map(ifaces) do (dim, uL, uR)
        ϕ_hll(uL, uR, dim, gas)
    end
    # we want to write this as u_next = u + Δt * diff
    Δu = inv(Δx) * (ϕ.west - ϕ.east) + inv(Δy) * (ϕ.south - ϕ.north)

    return (cell_speed = maximum_signal_speed, cell_Δu = Δu)
end

function compute_next_u(cell, Δt, Δu)
    u_next = state_to_vector(cell.u) + Δt * Δu
    RegularQuadCell(cell.id, cell.idx, cell.center, ConservedProps(u_next), cell.neighbors)
end

function step_cell_simulation!(
    cells_next,
    active_cells,
    Δt_maximum,
    boundary_conditions,
    cfl_limit,
    Δx,
    Δy,
    gas::CaloricallyPerfectGas;
    tpool = :default,
)
    T = dtype(active_cells[1])
    step_tasks = map(enumerate(active_cells)) do (idx, cell)
        #@show cell
        n_data = single_cell_neighbor_data(idx, active_cells, boundary_conditions, gas)
        c_data = cell.u
        Threads.@spawn begin
            mid = state_to_vector($c_data)
            nbrs = map(state_to_vector, $n_data)
            return compute_cell_update(mid, nbrs, $Δx, $Δy, gas)
        end
    end
    next::Vector{@NamedTuple{cell_speed::T, cell_Δu::SVector{4,T}}} = fetch.(step_tasks)
    a_max = mapreduce(el -> el.cell_speed, max, next)
    Δt = min(Δt_maximum, cfl_limit * Δx / a_max)
    for i ∈ eachindex(next, cells_next, active_cells)
        cells_next[i] = compute_next_u(active_cells[i], Δt, next[i].cell_Δu)
    end
    return Δt
end

struct CellBasedEulerSim{T,Q1,Q2,Q3}
    ncells::Tuple{Int,Int}
    nsteps::Int
    bounds::NTuple{2,Tuple{T,T}}
    tsteps::Vector{T}
    cell_ids::Array{Int,2}
    cells::Array{RegularQuadCell{T,Q1,Q2,Q3},2}
end

n_space_dims(::CellBasedEulerSim) = 2

function cell_boundaries(e::CellBasedEulerSim)
    return ntuple(i -> cell_boundaries(e, i), 2)
end

function cell_centers(e::CellBasedEulerSim)
    return ntuple(i -> cell_boundaries(e, i), 2)
end

function nth_step(csim::CellBasedEulerSim, n)
    return csim.tsteps[n], view(esim.cells, Colon(), n)
end

eachstep(csim::CellBasedEulerSim) = [nth_step(csim, n) for n ∈ 1:n_tsteps(csim)]

"""
    simulate_euler_equations_cells(u0, T_end, boundary_conditions, bounds, ncells; gas, CFL, max_tsteps, write_output, output_tag)

Simulate the solution to the Euler equations from `t=0` to `t=T`, with `u(0, x) = u0(x)`. 
Time step size is computed from the CFL condition.

The simulation will fail if any nonphysical conditions are reached (speed of sound cannot be computed). 

The simulation can be written to disk.

Arguments
---
- `u0`: ``u(0, x, y):ℝ^2↦ConservedProps{2, T, ...}``: conditions at time `t=0`.
- `T_end`: Must be greater than zero.
- `boundary_conditions`: a tuple of boundary conditions for each space dimension
- `obstacles`: list of obstacles in the flow.
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
    (currently unused, but this does multithread via Threads.@spawn)
- `info_frequency`: How often should info be printed?
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
    return_complete_result = false,
    history_in_memory = false,
    output_tag = "cell_euler_sim",
    info_frequency = 10,
    thread_pool = :default,
)
    N = length(ncells)
    T = typeof(T_end)
    @assert N == 2
    @assert length(bounds) == 2
    @assert length(boundary_conditions) == 5
    T_end > 0 && DomainError("T_end = $T_end invalid, T_end must be positive")
    0 < cfl_limit < 1 || @warn "CFL invalid, must be between 0 and 1 for stabilty" cfl_limit

    cell_ifaces = [range(b...; length = n + 1) for (b, n) ∈ zip(bounds, ncells)]
    cell_centers = [ax[1:end-1] .+ step(ax) / 2 for ax ∈ cell_ifaces]
    dV = step.(cell_centers)

    u_cells, ids = quadcell_list_and_id_grid(u0, bounds, ncells, obstacles)
    @show ids[end]
    u_cells_next = similar(u_cells)
    n_tsteps = 1
    t = zero(T)

    if write_result
        if !isdir("data")
            mkdir("data")
        end
        tape_file = joinpath("data", output_tag * ".celltape")
    end

    if !write_result && !history_in_memory
        @info "Only the final value of the simulation will be available." T_end
    end

    u_history = typeof(u_cells)[]
    t_history = typeof(t)[]
    if (history_in_memory)
        push!(u_history, copy(u))
        push!(t_history, t)
    end
    if write_result
        tape_stream = open(tape_file, "w+")
        write(tape_stream, zero(Int), length(u_cells), length(ncells), ncells...)
        for b ∈ bounds
            write(tape_stream, b...)
        end
        write(tape_stream, ids, t)
        for i ∈ eachindex(u_cells)
            write(tape_stream, state_to_vector(u_cells[i].u))
        end
    end

    while !(t > T_end || t ≈ T_end) && n_tsteps < max_tsteps
        Δt = step_cell_simulation!(
            u_cells_next,
            u_cells,
            T_end - t,
            boundary_conditions,
            cfl_limit,
            dV...,
            gas,
        )
        if (n_tsteps - 1) % info_frequency == 0
            @info "Time step..." n_tsteps t_k = t Δt
        end
        n_tsteps += 1
        t += Δt
        u_cells .= u_cells_next

        if write_result
            write(tape_stream, t)
            for i ∈ eachindex(u_cells)
                write(tape_stream, state_to_vector(u_cells[i].u))
            end
        end

        if history_in_memory
            push!(u_history, copy(u_cells))
            push!(t_history, t)
        end
    end

    if write_result
        seekstart(tape_stream)
        write(tape_stream, n_tsteps)
        seekend(tape_stream)
        close(tape_stream)
    end

    if history_in_memory
        @assert n_tsteps == length(t_history)
        return CellBasedEulerSim{T}(
            (ncells...,),
            n_tsteps,
            (((first(r), last(r)) for r ∈ cell_ifaces)...),
            t_history,
            ids,
            stack(u_history),
        )
    elseif return_complete_result && write_result
        return load_cell_sim(tape_file)
    end
    return CellBasedEulerSim(
        (ncells...,),
        1,
        (((first(r), last(r)) for r ∈ cell_ifaces)...,),
        [t],
        ids,
        reshape(u_cells, size(u_cells)..., 1),
    )
end