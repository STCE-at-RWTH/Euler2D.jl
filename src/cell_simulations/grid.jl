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

"""
    FVMCell

Abstract data type for all cells in a Cartesian grid.

All FVMCells _must_ provide the following methods:

 - `numeric_dtype(::FVMCell)`
 - `update_dtype(::FVMCell)`
 - `cell_points(::FVMCell)`
"""
abstract type FVMCell end

function is_cell_contained_by(cell::FVMCell, closed_poly)
    return all(eachcol(cell_points(cell))) do p
        point_inside(closed_poly, p)
    end
end

function is_cell_overlapping(cell::FVMCell, closed_poly)
    v = map(eachcol(cell_points(cell))) do p
        point_inside(closed_poly, p)
    end
    return any(v) && !all(v)
end

function overlapping_cell_area(cell::FVMCell, poly)
    isect = poly_intersection(cell_points(cell), poly)
    return polygon_area(isect)
end

"""
    PrimalQuadCell{T} <: FVMCell

QuadCell data type for a primal computation.

Type Parameters
---
 - `T`: Numeric data type.

Fields
---
 - `id`: Which quad cell is this?
 - `idx`: Which grid cell does this data represent?
 - `center`: Where is the center of this quad cell?
 - `extent`: How large is this quad cell?
 - `u`: What are the cell-averaged non-dimensionalized conserved properties in this cell?
 - `neighbors`: What are this cell's neighbors?
"""
struct PrimalQuadCell{T} <: FVMCell
    id::Int
    idx::CartesianIndex{2}
    center::SVector{2,T}
    extent::SVector{2,T}
    u::SVector{4,T}
    # either (:boundary, :cell)
    # and then the ID of the appropriate boundary
    neighbors::NamedTuple{
        (:north, :south, :east, :west),
        NTuple{4,Tuple{CellNeighboring,Int}},
    }
end

"""
    TangentQuadCell{T, NSEEDS,PARAMCOUNT} <: FVMCell

QuadCell data type for a primal computation. Pushes forward `NSEEDS` seed values through the JVP of the flux function.
`PARAMCOUNT` determines the "length" of the underlying `SMatrix` for `u̇`.

Fields
---
 - `id`: Which quad cell is this?
 - `idx`: Which grid cell does this data represent?
 - `center`: Where is the center of this quad cell?
 - `extent`: How large is this quad cell?
 - `u`: What are the cell-averaged non-dimensionalized conserved properties in this cell?
 - `u̇`: What are the cell-averaged pushforwards in this cell?
 - `neighbors`: What are this cell's neighbors?
"""
struct TangentQuadCell{T,NSEEDS,PARAMCOUNT} <: FVMCell
    id::Int
    idx::CartesianIndex{2}
    center::SVector{2,T}
    extent::SVector{2,T}
    u::SVector{4,T}
    u̇::SMatrix{4,NSEEDS,T,PARAMCOUNT}
    neighbors::NamedTuple{
        (:north, :south, :east, :west),
        NTuple{4,Tuple{CellNeighboring,Int}},
    }
end

"""
    PolyCell{T, NV}


"""
struct TangentPolyCell{T,NV,NSEEDS,NTANGENTS} <: FVMCell
    id::Int
    boundary::SClosedPolygon{T,NV}
    u::SVector{4,T}
    u̇::SMatrix{4,NSEEDS,T,NTANGENTS}
    neighbors::NTuple{NV,Tuple{CellNeighboring,Int}}
end

numeric_dtype(::PrimalQuadCell{T}) where {T} = T
numeric_dtype(::Type{PrimalQuadCell{T}}) where {T} = T

numeric_dtype(::TangentQuadCell{T,N,P}) where {T,N,P} = T
numeric_dtype(::Type{TangentQuadCell{T,N,P}}) where {T,N,P} = T
n_seeds(::TangentQuadCell{T,N,P}) where {T,N,P} = N
n_seeds(::Type{TangentQuadCell{T,N,P}}) where {T,N,P} = N

numeric_dtype(::TangentQuadCell{T,N,P}) where {T,N,P} = T
numeric_dtype(::Type{TangentQuadCell{T,N,P}}) where {T,N,P} = T
n_seeds(::TangentQuadCell{T,N,P}) where {T,N,P} = N
n_seeds(::Type{TangentQuadCell{T,N,P}}) where {T,N,P} = N

function cell_points(cell::PrimalQuadCell)
    c = cell.center
    dx, dy = cell.extent / 2
    return make_closed(
        SVector(
            c + SVector(dx, -dy),
            c + SVector(-dx, -dy),
            c + SVector(-dx, dy),
            c + SVector(dx, dy),
        ),
    )
end

function cell_points(cell::TangentQuadCell)
    c = cell.center
    dx, dy = cell.extent / 2
    return make_closed(
        SVector(
            c + SVector(dx, -dy),
            c + SVector(-dx, -dy),
            c + SVector(-dx, dy),
            c + SVector(dx, dy),
        ),
    )
end

@doc """
        numeric_dtype(cell)
        numeric_dtype(::Type{CELL_TYPE})

    Get the numeric data type associated with this cell.
    """ numeric_dtype

update_dtype(::Type{T}) where {T<:PrimalQuadCell} = NTuple{2,SVector{4,numeric_dtype(T)}}
function update_dtype(::Type{TangentQuadCell{T,N,P}}) where {T,N,P}
    return Tuple{SVector{4,T},SVector{4,T},SMatrix{4,N,T,P},SMatrix{4,N,T,P}}
end

@doc """
    update_dtype(::Type{T<:QuadCell})

Get the tuple of update data types that must be enforced upon fetch-ing results out of the worker tasks.
""" update_dtype

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

cell_volume(cell) = *(cell.extent...)

function phantom_neighbor(cell::PrimalQuadCell, dir, bc, gas)
    # HACK use nneighbors as intended.
    @assert dir ∈ (:north, :south, :east, :west) "dir is not a cardinal direction..."
    @assert nneighbors(bc) == 1 "dirty hack alert, this function needs to be extended for bcs with more neighbors"
    phantom = @set cell.id = 0

    @inbounds begin
        reverse_phantom = _dirs_bc_is_reversed[dir] && reverse_right_edge(bc)
        @reset phantom.center = cell.center + outward_normals(cell)[dir] .* cell.extent
        @reset phantom.neighbors =
            NamedTuple{(:north, :south, :east, :west)}(ntuple(Returns((IS_PHANTOM, 0)), 4))

        u = if _dirs_bc_is_reversed[dir]
            flip_velocity(cell.u, _dirs_dim[dir])
        else
            cell.u
        end
        phantom_u = phantom_cell(bc, u, _dirs_dim[dir], gas)
        if reverse_phantom
            @reset phantom.u = flip_velocity(phantom_u, _dirs_dim[dir])
        else
            @reset phantom.u = phantom_u
        end
    end
    return phantom
end

function phantom_neighbor(
    cell::TangentQuadCell{T,NSEEDS,PARAMCOUNT},
    dir,
    bc,
    gas,
) where {T,NSEEDS,PARAMCOUNT}
    # HACK use nneighbors as intended.
    @assert dir ∈ (:north, :south, :east, :west) "dir is not a cardinal direction..."
    @assert nneighbors(bc) == 1 "dirty hack alert, this function needs to be extended for bcs with more neighbors"
    phantom = @set cell.id = 0

    @inbounds begin
        reverse_phantom = _dirs_bc_is_reversed[dir] && reverse_right_edge(bc)
        @reset phantom.center = cell.center + outward_normals(cell)[dir] .* cell.extent
        @reset phantom.neighbors =
            NamedTuple{(:north, :south, :east, :west)}(ntuple(Returns((IS_PHANTOM, 0)), 4))

        # TODO there must be a way to do this with Accessors.jl and "lenses" that makes sense
        # HACK is this utter nonsense????? I do not know. 
        dim = _dirs_dim[dir]
        u = _dirs_bc_is_reversed[dir] ? flip_velocity(cell.u, dim) : cell.u
        u̇ = _dirs_bc_is_reversed[dir] ? flip_velocity(cell.u̇, dim) : cell.u̇
        phantom_u = phantom_cell(bc, u, _dirs_dim[dir], gas)
        J_phantom = ForwardDiff.jacobian(u) do u
            phantom_cell(bc, u, _dirs_dim[dir], gas)
        end
        phantom_u̇ = J_phantom * u̇
        if reverse_phantom
            @reset phantom.u = flip_velocity(phantom_u, _dirs_dim[dir])
            @reset phantom.u̇ = flip_velocity(phantom_u̇, _dirs_dim[dir])
        else
            @reset phantom.u = phantom_u
            @reset phantom.u̇ = phantom_u̇
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

"""
Takes a range `[a, b]` and an axis size `n`, and
tries to expand the range to `[a-1, b+1]`. 

Will clamp `a-1` to `1` and`b+1` to `n`.

Returns `(c, d) = (max(1,a-1), min(n, b+1)` (the cells needed to copy into a partition) 
and `(e,f)` (the cells that a partition will be responsible for updating.
"""
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

# TODO could we speed up the broadcast operation by splitting the dictionaries that the "owned" cells and the "required" cells are in?
"""
  AbstractCellGridPartition{T, U}

Represents a partition of a larger FVM simulation grid. 
`T` is the cell type, `U` is the update type associated with `T`.
Methods:
- `numeric_dtype(p)`: Backing numeric data type
- `cell_type(p)`: Get the cell type.
- `update_type(p)`: Get the cell update type.

Subtypes implement:
- `partition_id(p)`: Returns the partition id
- `owned_cell_ids(p)`: Returns an iterable of cell IDs that `p` is responsible for updating
- `owns_cell(p, id)`: Test if `p` owns cell `id`.
- `copied_cells(p)`: Returns a (key,value) map of all cells that the partition has a copy of.
- `get_cell(p, id)`: Returns the partition `p`'s copy of cell `id`.
- `store_owned_cell_update!(p, id, update)`: Store an update for owned cell `id`.
- ``
"""
abstract type AbstractCellGridPartition{T,U} end

struct CellGridPartition{T,U} <: AbstractCellGridPartition{T,U}
    id::Int
    # which slice of the global grid was copied into this partition?
    global_extent::NTuple{2,NTuple{2,Int}}
    # which (global) indices is this partition responsible for updating?
    global_computation_indices::NTuple{2,NTuple{2,Int}}
    # which (local) indices is this partition responsible for updating?
    computation_indices::NTuple{2,NTuple{2,Int}}
    # what cell IDs were copied into this partition?
    cells_copied_ids::Array{Int,2}
    cells_map::Dict{Int,T}
    cells_update::Dict{Int,U}

    function CellGridPartition(
        id,
        global_extent,
        global_computation_indices,
        computation_indices,
        cells_copied_ids,
        cells_map::Dict{Int,T},
        cells_update::Dict{Int},
    ) where {T<:FVMCell}
        return new{T,update_dtype(T)}(
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

struct FastCellGridPartition{T,U} <: AbstractCellGridPartition{T,U}
    id::Int

    owned_cells::Dict{Int,T}
    owned_update::Dict{Int,U}

    neighbor_cells::Dict{Int,T}
    neighbors_update::Dict{Int,U}
end

"""
    numeric_dtype(::CellGridPartition)
    numeric_dtype(::Type{CellGridPartition})

Underlying numeric data type of this partition.
"""
numeric_dtype(::AbstractCellGridPartition{T,U}) where {T,U} = numeric_dtype(T)
numeric_dtype(::Type{AbstractCellGridPartition{T,U}}) where {T,U} = numeric_dtype(T)

cell_type(::AbstractCellGridPartition{T,U}) where {T,U} = T
cell_type(::Type{AbstractCellGridPartition{T,U}}) where {T,U} = T

update_dtype(::AbstractCellGridPartition{T,U}) where {T,U} = U
update_dtype(::Type{AbstractCellGridPartition{T,U}}) where {T,U} = U

function _computation_region_indices(cell_partition)
    return (
        range(cell_partition.computation_indices[1]...),
        range(cell_partition.computation_indices[2]...),
    )
end

"""
  owned_cell_ids(partition)

Get a collection of valid cell IDs that the given partition is responsible for updating.
None of these will be zero <=> each of these will be a valid global index of a cell.
"""
function owned_cell_ids(p::CellGridPartition)
    idxs = (range(p.computation_indices[1]...), range(p.computation_indices[2]...))
    return Iterators.filter(>(0), @view p.cells_copied_ids[idxs...])
end
owned_cell_ids(p::FastCellGridPartition) = keys(p.owned_cells)

"""
  owns_cell(partition, id)

Test if the partition `partition` owns cell `id`.
"""
function owns_cell(p::CellGridPartition, id)
    idxs = (range(p.computation_indices[1]...), range(p.computation_indices[2]...))
    return haskey(p.cells_map, id) && id ∈ @view p.cells_copied_ids[idxs...]
end
owns_cell(p::FastCellGridPartition, id) = haskey(p.owned_cells, id)

"""
  copied_cells(partition)

Get a `id=>cell` map of all cells that this partition needs to compute an update step. 
"""
copied_cells(partition::CellGridPartition) = partition.cells_map
function copied_cells(partition::FastCellGridPartition)
    return BackupDict(partition.owned_cells, partition.neighbor_cells)
end

"""
  get_cell(partition, cell_id)

Get the cell `cell_id` that this partition owns.
"""
get_cell(partition::CellGridPartition, cell_id) = partition.cells_map[cell_id]
get_cell(partition::FastCellGridPartition, cell_id) = partition.owned_cells[cell_id]

"""
  store_cell_update!(partition, cell_id, cell_update)

Store `cell_update` for `cell_id` after computing it.
"""
function store_cell_update!(partition::CellGridPartition, id, Δu)
    partition.cells_update[id] = Δu
end

function store_cell_update!(partition::FastCellGridPartition, id, Δu)
    partition.owned_update[id] = Δu
end

# TODO if we want to move beyond a structured grid, we have to redo this method. I have no idea how to do this.

function partition_cell_list(
    global_active_cells,
    global_cell_ids,
    partitions_per_axis;
    show_info = false,
)
    # minimum partition size includes i - 1 and i + 1 neighbors
    grid_size = size(global_cell_ids)
    all_part = split_axis.(grid_size, partitions_per_axis)

    cell_type = valtype(global_active_cells)
    update_type = update_dtype(cell_type)
    if show_info
        @info "Partitioning global cell grid into $(*(length.(all_part)...)) partitions." cell_type update_type
    end

    res = map(enumerate(Iterators.product(all_part...))) do (id, (part_x, part_y))
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
        cell_ids_map = Dict{Int,cell_type}()
        cell_updates_map = Dict{Int,update_type}()
        sizehint!(cell_ids_map, task_cell_count)
        sizehint!(cell_updates_map, task_cell_count)
        for i ∈ eachindex(task_cell_ids)
            cell_id = task_cell_ids[i]
            cell_id == 0 && continue
            cell_ids_map[cell_id] = global_active_cells[cell_id]
            cell_updates_map[cell_id] = zero.(fieldtypes(update_type))
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
    @assert _verify_partitioning(res) "Partition is invalid! Oh no"
    return res
end

function fast_partition_cell_list(
    global_active_cells,
    global_cell_ids,
    partitions_per_axis;
    show_info = true,
)
    grid_size = size(global_cell_ids)
    all_part = split_axis.(grid_size, partitions_per_axis)

    CELL_TYPE = valtype(global_active_cells)
    UPDATE_TYPE = update_dtype(CELL_TYPE)

    if show_info
        smalllest_partition, biggest_partition =
            extrema(Iterators.product(all_part...)) do (part_x, part_y)
                sz = (part_x[2] - part_x[1] + 1) * (part_y[2] - part_y[1] + 1)
                return sz
            end
        @info "Partioning global cell grid into $(*(length.(all_part)...)) partitions." CELL_TYPE UPDATE_TYPE smalllest_partition biggest_partition
    end
    partition_infos = Iterators.product(all_part...) |> enumerate |> collect
    partitions = tmap(partition_infos) do (partition_id, (part_x, part_y))
        # the partition covers some amount of the global grid
        # and owns a slice of that cover
        total_covered_x, locally_owned_x = expand_to_neighbors(part_x..., grid_size[1])
        total_covered_y, local_owned_y = expand_to_neighbors(part_y..., grid_size[2])

        global_covered_idxs =
            CartesianIndices((range(total_covered_x...), range(total_covered_y...)))
        partition_owned_idxs =
            CartesianIndices((range(locally_owned_x...), range(local_owned_y...)))

        partition_covered_cell_ids = @view global_cell_ids[global_covered_idxs]
        partition_owned_cell_ids =
            @view partition_covered_cell_ids[partition_owned_idxs]
        global_owned_idxs = CartesianIndices(parentindices(partition_owned_cell_ids))
        owned_active_cells = 0
        shared_active_cells = 0
        for i ∈ global_covered_idxs
            id = global_cell_ids[i]
            id == 0 && continue
            if i ∈ global_owned_idxs
                owned_active_cells += 1
            else
                shared_active_cells += 1
            end
        end
        # cells that this partition is responsible for computing the updates to
        partition_compute_cells = Dict{Int,CELL_TYPE}()
        partition_compute_cells_updates = Dict{Int,UPDATE_TYPE}()
        sizehint!(partition_compute_cells, owned_active_cells)
        sizehint!(partition_compute_cells_updates, owned_active_cells)
        # cells that OTHER partitions are responsible for but this partition needs to know about
        partition_shared_cells = Dict{Int,CELL_TYPE}()
        partition_shared_cells_updates = Dict{Int,UPDATE_TYPE}()
        sizehint!(partition_shared_cells, shared_active_cells)
        sizehint!(partition_shared_cells_updates, shared_active_cells)
        for i ∈ global_covered_idxs
            id = global_cell_ids[i]
            id == 0 && continue
            if i ∈ global_owned_idxs
                partition_compute_cells[id] = global_active_cells[id]
                partition_compute_cells_updates[id] = zero.(fieldtypes(UPDATE_TYPE))
            else
                partition_shared_cells[id] = global_active_cells[id]
                partition_shared_cells_updates[id] = zero.(fieldtypes(UPDATE_TYPE))
            end
        end
        if show_info
            @info "Creating cell partition on grid ids..." partition_id = partition_id copied_idxs =
                global_covered_idxs global_owned_idxs local_compute_idxs =
                (range(locally_owned_x...), range(local_owned_y...)) owned_active_cells shared_active_cells
        end

        return FastCellGridPartition{CELL_TYPE,UPDATE_TYPE}(
            partition_id,
            partition_compute_cells,
            partition_compute_cells_updates,
            partition_shared_cells,
            partition_shared_cells_updates,
        )
    end

    @assert _verify_fastpartitioning(partitions, global_active_cells)
    return partitions
end

"""
Test that the computation regions of the partitions in `p` do not share any cell ids.
"""
function _verify_partitioning(p)
    return all(Iterators.filter(Iterators.product(p, p)) do (p1, p2)
        p1.id != p2.id
    end) do (p1, p2)
        c1 = owned_cell_ids(p1)
        c2 = owned_cell_ids(p2)
        return !any(c1) do v1
            v1 == 0 && return false
            return any(c2) do v2
                v2 == 0 && return false
                v1 == v2
            end
        end
    end
end

function _verify_fastpartitioning(p, global_cells)
    no_overlap = all(Iterators.filter(Iterators.product(p, p)) do (p1, p2)
        p1.id != p2.id
    end) do (p1, p2)
        shared12 = intersect(keys(p1.owned_cells), keys(p2.neighbor_cells))
        shared21 = intersect(keys(p2.owned_cells), keys(p1.neighbor_cells))
        return all(owned_cell_ids(p1)) do cell_id
            cell_id ∉ owned_cell_ids(p2)
        end
    end
    computed_cells = mapreduce(owned_cell_ids, union, p)
    return no_overlap && (computed_cells == keys(global_cells))
end

function collect_cell_partition!(global_cells, partition::CellGridPartition)
    data_region = owned_cell_ids(partition)
    for id ∈ data_region
        global_cells[id] = partition.cells_map[id]
    end
end

function collect_cell_partition!(global_cells, partition::FastCellGridPartition)
    merge!(global_cells, partition.owned_cells)
end

function collect_cell_partitions(cell_partitions, n_active_cells)
    u_global = Dict{Int64,cell_type(first(cell_partitions))}()
    sizehint!(u_global, n_active_cells)
    foreach(cell_partitions) do p
        collect_cell_partition!(u_global, p)
    end
    return u_global
end

function _iface_speed(iface::Tuple{Int,T,T}, gas) where {T<:FVMCell}
    return max(abs.(interface_signal_speeds(iface[2].u, iface[3].u, iface[1], gas))...)
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

"""
    compute_cell_update_and_max_Δt(cell, active_cells, boundary_conditions, gas)

Computes the update (of type `update_dtype(typeof(cell))`) for a given cell.

Arguments
---
- `cell`
- `active_cells`: The active cell partition or simulation. Usually a `Dict` that maps `id => typeof(cell)`
- `boundary_conditions`: The boundary conditions
- `gas::CaloricallyPerfectGas`: The simulation fluid.

Returns
---
`(update, Δt_max)`: A tuple of the cell update and the maximum time step size allowed by the CFL condition.
"""
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
        return ϕ_hll(cell_L.u, cell_R.u, dim, gas)
    end

    Δx = map(ifaces) do (dim, cell_L, cell_R)
        (cell_L.extent[dim] + cell_R.extent[dim]) / 2
    end

    Δu = (
        inv(Δx.west) * ϕ.west - inv(Δx.east) * ϕ.east,
        inv(Δx.south) * ϕ.south - inv(Δx.north) * ϕ.north,
    )
    return (Δt_max, Δu)
end

function compute_cell_update_and_max_Δt(
    cell::TangentQuadCell{T,N,P},
    active_cells,
    boundary_conditions,
    gas,
) where {T,N,P}
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
        return ϕ_hll(cell_L.u, cell_R.u, dim, gas)
    end
    ϕ_jvp = map(ifaces) do (dim, cell_L, cell_R)
        return ϕ_hll_jvp(cell_L.u, cell_L.u̇, cell_R.u, cell_R.u̇, dim, gas)
    end

    Δx = map(ifaces) do (dim, cell_L, cell_R)
        (cell_L.extent[dim] + cell_R.extent[dim]) / 2
    end

    RESULT_DTYPE = update_dtype(typeof(cell))
    Δu::RESULT_DTYPE = (
        (inv(Δx.west) * ϕ.west) - (inv(Δx.east) * ϕ.east),
        (inv(Δx.south) * ϕ.south) - (inv(Δx.north) * ϕ.north),
        (inv(Δx.west) * ϕ_jvp.west) - (inv(Δx.east) * ϕ_jvp.east),
        (inv(Δx.south) * ϕ_jvp.south) - (inv(Δx.north) .* ϕ_jvp.north),
    )
    return (Δt_max, Δu)
end

function compute_partition_update_and_max_Δt!(
    partition,
    boundary_conditions,
    gas::CaloricallyPerfectGas,
)
    Δt_max = typemax(numeric_dtype(partition))
    for cell_id ∈ owned_cell_ids(partition)
        cell_Δt_max, cell_Δu = compute_cell_update_and_max_Δt(
            get_cell(partition, cell_id),
            copied_cells(partition),
            boundary_conditions,
            gas,
        )
        Δt_max = min(Δt_max, cell_Δt_max)
        store_cell_update!(partition, cell_id, cell_Δu)
    end

    return Δt_max
end

"""
    propagate_updates_to!(dest, src)

After computing the cell updates for the regions 
that a partition is responsible for, propagate the updates 
to other partitions.

Returns the number of cells updated.
"""
function propagate_updates_to!(
    dest::CellGridPartition{T},
    src::CellGridPartition{T},
) where {T}
    count = 0
    src_compute = owned_cell_ids(src)
    for src_id ∈ src_compute
        for dest_id ∈ dest.cells_copied_ids
            if src_id == dest_id
                dest.cells_update[src_id] = src.cells_update[src_id]
                count += 1
            end
        end
    end
    return count
end

function propagate_updates_to!(dest::FastCellGridPartition, src::FastCellGridPartition)
    count = 0
    shared_keys =
        Iterators.filter(k -> haskey(src.owned_update, k), keys(dest.neighbors_update))
    for k ∈ shared_keys
        count += 1
        dest.neighbors_update[k] = src.owned_update[k]
    end
    return count
end

function _update_cell(cell::PrimalQuadCell, Δu, Δt, dim)
    return @set cell.u = cell.u + Δt * Δu[dim]
end

function _update_cell(cell::TangentQuadCell, Δu, Δt, dim)
    @reset cell.u = cell.u + Δt * Δu[dim]
    return @set cell.u̇ = cell.u̇ + Δt * Δu[2+dim]
end

# zeroing out the update is not technically necessary, but it's also very cheap
# ( I hope )
function apply_partition_update!(partition::CellGridPartition, dim, Δt)
    for (k, v) ∈ partition.cells_update
        partition.cells_map[k] = _update_cell(partition.cells_map[k], v, Δt, dim)
        partition.cells_update[k] = zero.(fieldtypes(update_dtype(partition)))
    end
end

function apply_partition_update!(partition::FastCellGridPartition, dim, Δt)
    for (k, v) ∈ partition.owned_update
        partition.owned_cells[k] = _update_cell(partition.owned_cells[k], v, Δt, dim)
        partition.owned_update[k] = zero.(fieldtypes(update_dtype(partition)))
    end
    for (k, v) ∈ partition.neighbors_update
        partition.neighbor_cells[k] = _update_cell(partition.neighbor_cells[k], v, Δt, dim)
        partition.neighbors_update[k] = zero.(fieldtypes(update_dtype(partition)))
    end
end

function step_cell_simulation!(
    cell_partitions,
    Δt_maximum,
    boundary_conditions,
    cfl_limit,
    gas::CaloricallyPerfectGas,
)
    T = numeric_dtype(first(cell_partitions))
    # TODO
    # there has to be a cleverer way to do this...
    # perhaps a dict of id=>list of ids to pull from?
    adjacent_partition_pairs = collect(
        Iterators.filter(
            ((a, b),) -> a ≠ b,
            ((a.id, b.id) for (a, b) ∈ Iterators.product(cell_partitions, cell_partitions)),
        ),
    )
    partition_locks = [Base.Lockable(p) for p ∈ cell_partitions]

    # 1. Calculate updates
    # 2. Share update
    # 3. apply update with appropriate time step

    # first x
    # compute Δu from flux functions
    # find Δt
    Δt =
        cfl_limit * tmapreduce(
            min,
            cell_partitions;
            outputtype = T,
            init = Δt_maximum,
        ) do cell_partition
            compute_partition_update_and_max_Δt!(cell_partition, boundary_conditions, gas)
        end
    tforeach(adjacent_partition_pairs) do (id1, id2)
        lock(partition_locks[id1])
        propagate_updates_to!(partition_locks[id1][], cell_partitions[id2])
        unlock(partition_locks[id1])
    end
    tforeach(cell_partitions) do p
        apply_partition_update!(p, 1, Δt / 2)
    end
    # then in y
    tforeach(cell_partitions) do cell_partition
        compute_partition_update_and_max_Δt!(cell_partition, boundary_conditions, gas)
    end
    tforeach(adjacent_partition_pairs) do (id1, id2)
        lock(partition_locks[id1])
        propagate_updates_to!(partition_locks[id1][], cell_partitions[id2])
        unlock(partition_locks[id1])
    end
    tforeach(cell_partitions) do p
        apply_partition_update!(p, 2, Δt)
    end
    # then in x again
    tforeach(cell_partitions) do cell_partition
        compute_partition_update_and_max_Δt!(cell_partition, boundary_conditions, gas)
    end
    tforeach(adjacent_partition_pairs) do (id1, id2)
        lock(partition_locks[id1])
        propagate_updates_to!(partition_locks[id1][], cell_partitions[id2])
        unlock(partition_locks[id1])
    end
    tforeach(cell_partitions) do p
        apply_partition_update!(p, 1, Δt / 2)
    end
    return Δt
end

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

function active_cell_ids_from_mask(active_mask)::Array{Int,2}
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
    primal_cell_list_and_id_grid(u0, bounds, ncells, obstacles)

Computes a collection of active cells and their locations in a grid determined by `bounds` and `ncells`.
`Obstacles` can be placed into the simulation grid.
"""
function primal_cell_list_and_id_grid(u0, params, bounds, ncells, scaling, obstacles)
    centers = map(zip(bounds, ncells)) do (b, n)
        v = range(b...; length = n + 1)
        return v[1:end-1] .+ step(v) / 2
    end
    extent = SVector{2}(step.(centers)...)
    pts = Iterators.product(centers...)

    # u0 is probably cheap, right?
    _u0_func(x) = nondimensionalize(u0(x, params), scaling)

    u0_grid = map(_u0_func, pts)
    active_mask = active_cell_mask(centers..., obstacles)
    active_ids = active_cell_ids_from_mask(active_mask)
    @assert sum(active_mask) == last(active_ids)
    cell_list = Dict{Int,PrimalQuadCell{eltype(eltype(u0_grid))}}()
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

"""
    tangent_cell_list_and_id_grid(u0, bounds, ncells, obstacles)

Computes a collection of active cells and their locations in a grid determined by `bounds` and `ncells`.
`Obstacles` can be placed into the simulation grid.
"""
function tangent_cell_list_and_id_grid(u0, params, bounds, ncells, scaling, obstacles)
    centers = map(zip(bounds, ncells)) do (b, n)
        v = range(b...; length = n + 1)
        return v[1:end-1] .+ step(v) / 2
    end
    pts = Iterators.product(centers...)
    extent = SVector{2}(step.(centers)...)

    # u0 is probably cheap, right?
    _u0_func(x) = nondimensionalize(u0(x, params), scaling)
    _u̇0_func(x) = begin
        J = ForwardDiff.jacobian(params) do p
            nondimensionalize(u0(x, p), scaling)
        end
        return J * I
    end

    u0_type = typeof(_u0_func(first(pts)))
    T = eltype(u0_type)
    u̇0_type = typeof(_u̇0_func(first(pts)))

    NSEEDS = ncols_smatrix(u̇0_type)

    u0_grid = map(_u0_func, pts)
    u̇0_grid = map(_u̇0_func, pts)
    active_mask = active_cell_mask(centers..., obstacles)
    active_ids = active_cell_ids_from_mask(active_mask)
    @assert sum(active_mask) == last(active_ids)

    cell_list = Dict{Int,TangentQuadCell{T,NSEEDS,4 * NSEEDS}}()
    sizehint!(cell_list, sum(active_mask))
    for i ∈ eachindex(IndexCartesian(), active_ids, active_mask)
        active_mask[i] || continue
        cell_id = active_ids[i]
        (m, n) = Tuple(i)
        x_i = centers[1][m]
        y_j = centers[2][n]
        neighbors = cell_neighbor_status(i, active_ids)
        cell_list[cell_id] = TangentQuadCell(
            cell_id,
            i,
            SVector(x_i, y_j),
            extent,
            u0_grid[i],
            u̇0_grid[i],
            neighbors,
        )
    end
    return cell_list, active_ids
end
