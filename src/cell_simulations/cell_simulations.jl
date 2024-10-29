# I always think in "north south east west"... who knows why.
#   anyway
@enum CellBoundaries::UInt8 begin
    NORTH_BOUNDARY = 1
    SOUTH_BOUNDARY = 2
    EAST_BOUNDARY = 3
    WEST_BOUNDARY = 4
    INTERNAL_STRONGWALL = 5
end

@enum CellNeighboring::UInt8 begin
    OTHER_QUADCELL
    BOUNDARY_CONDITION
    IS_PHANTOM
end

abstract type QuadCell end

struct PrimalQuadCell{T,Q1<:Density,Q2<:MomentumDensity,Q3<:EnergyDensity} <: QuadCell
    id::Int
    idx::CartesianIndex{2}
    center::SVector{2,T}
    extent::SVector{2,T}
    u::ConservedProps{2,T,Q1,Q2,Q3}
    # either (:boundary, :cell)
    # and then the ID of the appropriate boundary
    neighbors::NamedTuple{
        (:north, :south, :east, :west),
        NTuple{4,Tuple{CellNeighboring,Int}},
    }
end

struct TangentQuadCell{T,Q1<:Density,Q2<:MomentumDensity,Q3<:EnergyDensity} <: QuadCell
    id::Int
    idx::CartesianIndex{2}
    center::SVector{2,T}
    extent::SVector{2,T}
    u::ConservedProps{2,T,Q1,Q2,Q3}
    u̇::SVector{4,T}
    neighbors::NamedTuple{
        (:north, :south, :east, :west),
        NTuple{4,Tuple{CellNeighboring,Int}},
    }
end

for CELL ∈ (PrimalQuadCell, TangentQuadCell)
    @eval numeric_dtype(::$CELL{T,U,V,W}) where {T,U,V,W} = T
    @eval numeric_dtype(::Type{$CELL{T,U,V,W}}) where {T,U,V,W} = T
    @eval cprops_dtype(::$CELL{T,U,V,W}) where {T,U,V,W} = ConservedProps{2,T,U,V,W}
    @eval cprops_dtype(::Type{$CELL{T,U,V,W}}) where {T,U,V,W} = ConservedProps{2,T,U,V,W}
end

@doc """
        numeric_dtype(cell)
        numeric_dtype(::Type{CELL_TYPE})

    Get the numeric data type associated with this cell.
    """ numeric_dtype

@doc """
        cprops_dtype(cell)
        cprops_dtype(::Type{CELL_TYPE})

    Get the `ConservedProps` data type associated with this cell.
    """ cprops_dtype

function Base.convert(
    ::Type{PrimalQuadCell{T,A1,A2,A3}},
    cell::PrimalQuadCell{T,B1,B2,B3},
) where {T,A1,A2,A3,B1,B2,B3}
    return PrimalQuadCell(
        cell.id,
        cell.idx,
        cell.center,
        cell.extent,
        convert(ConservedProps{2,T,A1,A2,A3}, cell.u),
        cell.neighbors,
    )
end

function Base.convert(
    ::Type{TangentQuadCell{T,A1,A2,A3}},
    cell::TangentQuadCell{T,B1,B2,B3},
) where {T,A1,A2,A3,B1,B2,B3}
    return TangentQuadCell(
        cell.id,
        cell.idx,
        cell.center,
        cell.extent,
        convert(ConservedProps{2,T,A1,A2,A3}, cell.u),
        cell.u̇,
        cell.neighbors,
    )
end

function inward_normals(T::DataType)
    return (
        north = SVector((zero(T), -one(T))...),
        south = SVector((zero(T), one(T))...),
        east = SVector((-one(T), zero(T))...),
        west = SVector((one(T), zero(T))...),
    )
end

function outward_normals(T::DataType)
    return (
        north = SVector((zero(T), one(T))...),
        south = SVector((zero(T), -one(T))...),
        east = SVector((one(T), zero(T))...),
        west = SVector((-one(T), zero(T))...),
    )
end

inward_normals(cell) = inward_normals(numeric_dtype(cell))
outward_normals(cell) = outward_normals(numeric_dtype(cell))

cell_volume(cell) = cell.extent[1] * cell.extent[2]

# TODO we should actually be more serious about compting these overlaps
#  and then computing volume-averaged quantities
point_inside(s::Obstacle, q) = point_inside(s, q.center)

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

function cell_neighbor_status(i, cell_ids)
    idx = CartesianIndices(cell_ids)[i]
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
        elseif cell_ids[neighbor] == 0
            return (BOUNDARY_CONDITION, Int(INTERNAL_STRONGWALL))
        else
            return (OTHER_QUADCELL, cell_ids[neighbor])
        end
    end
end

"""
    quadcell_list_and_id_grid(u0, bounds, ncells, obstacles)

Computes a collection of active cells and their locations in a grid determined by `bounds` and `ncells`.
`Obstacles` can be placed into the simulation grid.
"""
function quadcell_list_and_id_grid(u0, bounds, ncells, obstacles = [])
    centers = map(zip(bounds, ncells)) do (b, n)
        v = range(b...; length = n + 1)
        return v[1:end-1] .+ step(v) / 2
    end
    extent = SVector{2}(step.(centers)...)

    # u0 is probably cheap
    u0_grid = map(u0, Iterators.product(centers...))
    active_mask = active_cell_mask(centers..., obstacles)
    active_ids = active_cell_ids_from_mask(active_mask)
    @assert sum(active_mask) == last(active_ids)
    cell_list = Dict{
        Int,
        PrimalQuadCell{numeric_dtype(eltype(u0_grid)),quantity_types(eltype(u0_grid))...},
    }()
    sizehint!(cell_list, sum(active_mask))
    for i ∈ eachindex(IndexCartesian(), active_ids, active_mask)
        active_mask[i] || continue
        j = active_ids[i]
        (m, n) = Tuple(i)
        x_i = centers[1][m]
        y_j = centers[2][n]
        neighbors = cell_neighbor_status(i, active_ids)
        cell_list[j] =
            PrimalQuadCell(j, i, SVector(x_i, y_j), extent, u0_grid[i], neighbors)
    end
    return cell_list, active_ids
end

function phantom_neighbor(cell, dir, bc, gas)
    # HACK use nneighbors as intended.
    @assert dir ∈ (:north, :south, :east, :west) "dir is not a cardinal direction..."
    @assert nneighbors(bc) == 1 "dirty hack alert, this function needs to be extended for bcs with more neighbors"
    dirs_bc_is_reversed = (north = true, south = false, east = false, west = true)
    dirs_dim = (north = 2, south = 2, east = 1, west = 1)
    phantom = @set cell.id = 0

    @inbounds begin
        reverse_phantom = dirs_bc_is_reversed[dir] && reverse_right_edge(bc)
        @reset phantom.center = cell.center + outward_normals(cell)[dir] .* cell.extent
        @reset phantom.neighbors =
            NamedTuple{(:north, :south, :east, :west)}(ntuple(Returns((IS_PHANTOM, 0)), 4))

        u = if dirs_bc_is_reversed[dir]
            flip_velocity(cell.u, dirs_dim[dir])
        else
            cell.u
        end
        phantom_u = phantom_cell(bc, u, dirs_dim[dir], gas)
        if reverse_phantom
            @reset phantom.u = flip_velocity(phantom_u, dirs_dim[dir])
        else
            @reset phantom.u = phantom_u
        end
    end
    return phantom
end

"""
    neighbor_cells(cell, active_cells, boundary_conditions, gas)

Extract the states of the neighboring cells to `cell` from `active_cells`. 
Will compute phantoms as necessary from `boundary_conditions` and `gas`.
"""
function neighbor_cells(cell, active_cells, boundary_conditions, gas)
    neighbors = cell.neighbors
    # TODO use named tuple fusion rather than... this
    map((ntuple(i -> ((keys(neighbors)[i], neighbors[i])), 4))) do (dir, (kind, id))
        res = if kind == BOUNDARY_CONDITION
            @inbounds phantom_neighbor(cell, dir, boundary_conditions[id], gas)
        else
            active_cells[id]
        end
        return res
    end |> NamedTuple{(:north, :south, :east, :west)}
end

function split_axis(len, n)
    l = len ÷ n
    rem = len - (n * l)
    tpl_ranges = [(i * l + 1, (i + 1) * l) for i = 0:(n-1)]
    if rem > 0
        tpl_ranges[end] = (l * (n - 1) + 1, len)
    end
    return tpl_ranges
end

function expand_to_neighbors(left_idx, right_idx, axis_size)
    len = right_idx - left_idx + 1
    if left_idx > 1
        new_l = left_idx - 1
        left_idx = 2
    else
        new_l = 1
        left_idx = 1
    end

    if right_idx < axis_size
        new_r = right_idx + 1
        right_idx = left_idx + len - 1
    else
        new_r = right_idx
        right_idx = left_idx + len - 1
    end
    return (new_l, new_r), (left_idx, right_idx)
end

update_size(::Type{T}) where {T<:PrimalQuadCell} = 1
update_size(::Type{T}) where {T<:TangentQuadCell} = 2

struct CellGridPartition{T,U,V}
    id::Int
    # which slice of the global grid was copied into this partition?
    global_extent::NTuple{2,NTuple{2,Int}}
    # which (global) indices is this partition responsible for updating?
    global_computation_indices::NTuple{2,NTuple{2,Int}}
    # which (local) indices is this partition responsible for updating?
    computation_indices::NTuple{2,NTuple{2,Int}}
    # what cell IDs were copied into this partition?
    cells_copied_ids::Array{Int,2}
    #TODO Switch to Dictionaries.jl? Peformance seems fine as of now.
    cells_map::Dict{Int,T}
    cells_update::Dict{Int,NTuple{U,SVector{4,V}}}

    function CellGridPartition(
        id,
        global_extent,
        global_computation_indices,
        computation_indices,
        cells_copied_ids,
        cells_map::Dict{Int,T},
        cells_update,
    ) where {T<:QuadCell}
        return new{T,update_size(T),numeric_dtype(T)}(
            id,
            global_extent,
            global_computation_indices,
            computation_indices,
            cells_copied_ids,
            cells_map,
            cells_update,
        )
    end
end

"""
    numeric_dtype(::CellGridPartition)
    numeric_dtype(::Type{CellGridPartition})

Underlying numeric data type of this partition.
"""
numeric_dtype(::CellGridPartition{T,U,V}) where {T,U,V} = numeric_dtype(T)
numeric_dtype(::Type{CellGridPartition{T,U,V}}) where {T,U,V} = numeric_dtype(T)

cell_type(::CellGridPartition{T,U,V}) where {T,U,V} = T
cell_type(::Type{CellGridPartition{T,U,V}}) where {T,U,V} = T

"""
    cells_map_type(::CellGridPartition)
    cells_map_type(::Type{CellGridPartition})
"""
cells_map_type(::CellGridPartition{T}) where {T} = Dict{Int,T}
cells_map_type(::Type{CellGridPartition{T}}) where {T} = Dict{Int,T}

# TODO if we want to move beyond a structured grid, we have to redo this method. I have no idea how to do this.
# TODO how slow is this function? we may be wasting a lot of time partitioning that we don't recover by multithreading. Certainly memory use goes up.

function partition_cell_list(
    global_active_cells,
    global_cell_ids,
    tasks_per_axis;
    show_info = false,
)
    # minimum partition size includes i - 1 and i + 1 neighbors
    grid_size = size(global_cell_ids)
    (all_part_x, all_part_y) = split_axis.(grid_size, tasks_per_axis)

    res =
        map(enumerate(Iterators.product(all_part_x, all_part_y))) do (id, (part_x, part_y))
            # adust slice width...
            task_x, task_working_x = expand_to_neighbors(part_x..., grid_size[1])
            task_y, task_working_y = expand_to_neighbors(part_y..., grid_size[2])
            if show_info
                @info "Creating cell partition on grid ids..." id = id global_ids =
                    (range(task_x...), range(task_y...)) compute_ids =
                    (range(task_working_x...), range(task_working_y...))
            end
            # cells copied for this task
            # we want to copy this...?
            task_cell_ids = global_cell_ids[range(task_x...), range(task_y...)]
            # total number of cells this task has a copy of
            task_cell_count = count(>(0), task_cell_ids)
            cell_type = valtype(global_active_cells)
            cell_ids_map = Dict{Int,cell_type}()
            n_update_components = update_size(cell_type)
            update_dtype = SVector{4,numeric_dtype(cell_type)}
            cell_updates_map = Dict{Int,NTuple{n_update_components,update_dtype}}()
            sizehint!(cell_ids_map, task_cell_count)
            sizehint!(cell_updates_map, task_cell_count)
            for i ∈ eachindex(task_cell_ids)
                cell_id = task_cell_ids[i]
                cell_id == 0 && continue
                get!(cell_ids_map, cell_id, global_active_cells[cell_id])
                get!(
                    cell_updates_map,
                    cell_id,
                    ntuple(Returns(zeros(update_dtype)), n_update_components),
                )
            end
            return CellGridPartition(
                id,
                (task_x, task_y),
                (part_x, part_y),
                (task_working_x, task_working_y),
                task_cell_ids,
                cell_ids_map,
                cell_updates_map,
            )
        end
    return res
end

function departition_cell_list(cell_partitions, global_cell_ids)
    u_global = empty(cell_partitions[1].cells_map)
    sizehint!(u_global, count(≠(0), global_cell_ids))
    for part ∈ cell_partitions
        computation_region = view(
            part.cells_copied_ids,
            range(part.computation_indices[1]...),
            range(part.computation_indices[2]...),
        )
        for id ∈ computation_region
            id == 0 && continue
            get!(u_global, id, part.cells_map[id])
        end
    end
    return u_global
end

function _iface_speed(iface::Tuple{Int,T,T}, gas) where {T<:QuadCell}
    uL = state_to_vector(iface[2].u)
    uR = state_to_vector(iface[3].u)
    return max(abs.(interface_signal_speeds(uL, uR, iface[1], gas))...)
end

function maximum_cell_signal_speeds(
    interfaces::NamedTuple{(:north, :south, :east, :west)},
    gas::CaloricallyPerfectGas,
)
    # doing this with map allocated?!
    return SVector(
        max(_iface_speed(interfaces.north, gas), _iface_speed(interfaces.south, gas)),
        max(_iface_speed(interfaces.east, gas), _iface_speed(interfaces.west, gas)),
    )
end

function compute_cell_update_and_max_Δt(
    cell::PrimalQuadCell,
    active_cells,
    boundary_conditions,
    gas,
)
    neighbors = neighbor_cells(cell, active_cells, boundary_conditions, gas)
    ifaces = (
        north = (2, cell, neighbors.north),
        south = (2, neighbors.south, cell),
        east = (1, cell, neighbors.east),
        west = (1, neighbors.west, cell),
    )
    a = maximum_cell_signal_speeds(ifaces, gas)
    Δt_max = min((cell.extent ./ a)...)

    ϕ = map(ifaces) do (dim, cell_L, cell_R)
        uL = state_to_vector(cell_L.u)
        uR = state_to_vector(cell_R.u)
        return ϕ_hll(uL, uR, dim, gas)
    end

    Δx = map(ifaces) do (dim, cell_L, cell_R)
        (cell_L.extent[dim] + cell_R.extent[dim]) / 2
    end

    Δu = (
        inv(Δx.west) * ϕ.west - inv(Δx.east) * ϕ.east + inv(Δx.south) * ϕ.south -
        inv(Δx.north) * ϕ.north
    )

    return (Δt_max, Δu)
end

function compute_cell_update_and_max_Δt(
    cell::TangentQuadCell,
    active_cells,
    boundary_conditions,
    gas,
)
    neighbors = neighbor_cells(cell, active_cells, boundary_conditions, gas)
    ifaces = (
        north = (2, cell, neighbors.north),
        south = (2, neighbors.south, cell),
        east = (1, cell, neighbors.east),
        west = (1, neighbors.west, cell),
    )
    a = maximum_cell_signal_speeds(ifaces, gas)
    Δt_max = min((cell.extent ./ a)...)
    ϕ = map(ifaces) do (dim, cell_L, cell_R)
        uL = state_to_vector(cell_L.u)
        uR = state_to_vector(cell_R.u)
        return (ϕ_hll(uL, uR, dim, gas), ϕ_hll_jvp(uL, cell_L.u̇, uR, cell_R.u̇, dim, gas))
    end

    Δx = map(ifaces) do (dim, cell_L, cell_R)
        (cell_L.extent[dim] + cell_R.extent[dim]) / 2
    end

    Δu = ntuple(2) do i
        inv(Δx.west) * ϕ.west[i] - inv(Δx.east) * ϕ.east[i] + inv(Δx.south) * ϕ.south[i] - inv(Δx.north) * ϕ.north[i]
    end

    return (Δt_max, Δu)
end

# no longer allocates since we pre-allocate the update dict in the struct itself!
function compute_partition_update_and_max_Δt!(
    cell_partition::CellGridPartition{T},
    boundary_conditions,
    gas::CaloricallyPerfectGas,
) where {T}
    computation_region = view(
        cell_partition.cells_copied_ids,
        range(cell_partition.computation_indices[1]...),
        range(cell_partition.computation_indices[2]...),
    )
    Δt_max = typemax(numeric_dtype(T))
    for cell_id ∈ computation_region
        cell_id == 0 && continue
        cell_Δt_max, cell_Δu = compute_cell_update_and_max_Δt(
            cell_partition.cells_map[cell_id],
            cell_partition.cells_map,
            boundary_conditions,
            gas,
        )
        Δt_max = min(Δt_max, cell_Δt_max)
        get!(cell_partition.cells_update, cell_id, cell_Δu)
    end

    return Δt_max
end

"""
    propagate_updates_to!(dest, src, global_cell_ids)

After computing and applying the cell updates for the regions 
that a partition is responsible for, propagate the updates 
to other partitions.

Returns the number of cells updated.
"""
function propagate_updates_to!(
    dest::CellGridPartition{T},
    src::CellGridPartition{T},
) where {T}
    src_region = view(
        src.cells_copied_ids,
        range(src.computation_indices[1]...),
        range(src.computation_indices[2]...),
    )
    count = 0
    for idx ∈ eachindex(src_region)
        if haskey(dest.cells_map, src_region[idx])
            count += 1
            dest.cells_map[src_region[idx]] = src.cells_map[src_region[idx]]
        end
    end
    return count
end

function _update_cprops(u::ConservedProps{2,T}, Δu::SVector{4,T}, Δt) where {T}
    # preserve units?
    return convert(typeof(u), ConservedProps(state_to_vector(u) + Δt * Δu))
end

function apply_partition_update!(
    partition::CellGridPartition{T,U,V},
    Δt,
) where {T<:PrimalQuadCell,U,V}
    for (k, v) ∈ partition.cells_update
        cell = partition.cells_map[k]
        u_next = _update_cprops(cell.u, v[1], Δt)
        partition.cells_map[k] = @set cell.u = u_next
        partition.cells_update[k] = ntuple(Returns(zeros(SVector{4,V})), U)
    end
end

function apply_partition_update!(
    partition::CellGridPartition{T},
    Δt,
) where {T<:TangentQuadCell}
    for (k, v) ∈ partition.cells_update
        u_next = _update_cprops(partition.cells_map[k].u, v[1], Δt)
        @reset partition.cells_map[k].u = u_next
        @reset partition.cells_map[k].u̇ = partition.cells_map[k].u̇ + Δt * v[2]
    end
end

function step_cell_simulation!(
    cell_partitions,
    Δt_maximum,
    boundary_conditions,
    cfl_limit,
    gas::CaloricallyPerfectGas,
)
    T = numeric_dtype(eltype(cell_partitions))
    # compute Δu from flux functions
    compute_partition_update_tasks = map(cell_partitions) do cell_partition
        Threads.@spawn begin
            # not sure what to interpolate here
            compute_partition_update_and_max_Δt!(cell_partition, $boundary_conditions, $gas)
        end
    end
    partition_max_Δts::Array{T,length(size(compute_partition_update_tasks))} =
        fetch.(compute_partition_update_tasks)
    # find Δt
    Δt = mapreduce(min, partition_max_Δts; init = Δt_maximum) do val
        cfl_limit * val
    end
    # apply update, does this really need to be threaded?
    
    for p_idx ∈ eachindex(cell_partitions)
        # propagate updates to other partitions
        for dest_idx ∈ eachindex(cell_partitions)
            dest_idx == p_idx && continue
            propagate_updates_to!(cell_partitions[dest_idx], cell_partitions[dest_idx])
        end
    end

    update_tasks = map(cell_partitions) do p
        Threads.@spawn begin
            apply_partition_update!(p, Δt)
        end
    end
    wait.(update_tasks)

    return Δt
end

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
                u_cur = departition_cell_list(cell_partitions, global_cell_ids)
                put!(writer_channel, (t, u_cur))
                push!(u_history, u_cur)
                push!(t_history, t)
            elseif write_result
                put!(
                    writer_channel,
                    (t, departition_cell_list(cell_partitions, global_cell_ids)),
                )
            elseif history_in_memory
                push!(u_history, departition_cell_list(cell_partitions, global_cell_ids))
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
        [departition_cell_list(cell_partitions, global_cell_ids)],
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
            # @info "Reading..." k t_k = time_steps[k] n_active
            read!(f, temp_cell_vals)
            cell_vals[k] = Dict{Int,CellDType}()
            sizehint!(cell_vals[k], n_active)
            for cell ∈ temp_cell_vals
                get!(cell_vals[k], cell.id, cell)
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