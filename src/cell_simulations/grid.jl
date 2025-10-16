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
- `get_cell_update(p, id)`: Returns the partition `p`'s copy of the update for cell `id`.
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

    owned_ids::Vector{Int}
    owned_cells::Dict{Int,T}
    owned_update::Dict{Int,U}

    neighbor_ids::Vector{Int}
    neighbor_cells::Dict{Int,T}
    neighbors_update::Dict{Int,U}

    function FastCellGridPartition(
        id,
        owned_ids,
        owned_cells::Dict{Int,T},
        owned_update,
        neighbor_ids,
        neighbor_cells::Dict{Int,T},
        neighbors_update,
    ) where {T<:FVMCell}
        return new{T,update_dtype(T)}(
            id,
            owned_ids,
            owned_cells,
            owned_update,
            neighbor_ids,
            neighbor_cells,
            neighbors_update,
        )
    end
end

"""
    FastCellGridPartition(id, owned_cells, neighbor_cells)

Create a partition with fast lookup and initialize the update dictionaries to the appropriate zeros.
"""
function FastCellGridPartition(
    id,
    owned_cells::Dict{Int,T},
    neighbor_cells::Dict{Int,T},
) where {T<:FVMCell}
    U = update_dtype(T)
    owned_ids = sort!(collect(keys(owned_cells)))
    neighbor_ids = sort!(collect(keys(neighbor_cells)))
    owned_update = Dict{Int,U}()
    neighbor_update = Dict{Int,U}()
    sizehint!(owned_update, length(owned_ids))
    sizehint!(neighbor_update, length(neighbor_ids))
    foreach(id -> owned_update[id] = zero_cell_update(U), owned_ids)
    foreach(id -> neighbor_update[id] = zero_cell_update(U), neighbor_ids)
    return FastCellGridPartition(
        id,
        owned_ids,
        owned_cells,
        owned_update,
        neighbor_ids,
        neighbor_cells,
        neighbor_update,
    )
end

"""
    numeric_dtype(::CellGridPartition)
    numeric_dtype(::Type{CellGridPartition})

Underlying numeric data type of this partition.
"""
numeric_dtype(::AbstractCellGridPartition{T,U}) where {T,U} = numeric_dtype(T)
numeric_dtype(::Type{<:AbstractCellGridPartition{T,U}}) where {T,U} = numeric_dtype(T)

cell_type(::AbstractCellGridPartition{T,U}) where {T,U} = T
cell_type(::Type{<:AbstractCellGridPartition{T,U}}) where {T,U} = T

update_dtype(::AbstractCellGridPartition{T,U}) where {T,U} = U
update_dtype(::Type{<:AbstractCellGridPartition{T,U}}) where {T,U} = U

owned_cell_count(p::AbstractCellGridPartition) = count(≠(0), owned_cell_ids(p))

"""
   owned_cell_ids(partition)

Get a collection of valid cell IDs that the given partition is responsible for updating.
None of these will be zero <=> each of these will be a valid global index of a cell.
"""
function owned_cell_ids(p::CellGridPartition)
    idxs = (range(p.computation_indices[1]...), range(p.computation_indices[2]...))
    return Iterators.filter(>(0), @view p.cells_copied_ids[idxs...])
end
owned_cell_ids(p::FastCellGridPartition) = p.owned_ids

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
    has_cell_as_neighbor(partition, id)

Test if the partition `partition` will need the update from cell `id` to perform the update step.
"""
function has_cell_as_neighbor(p::CellGridPartition, id)
    # this is slow
    return haskey(p.cells_map, id) && !owns_cell(p, id)
end
has_cell_as_neighbor(p::FastCellGridPartition, id) = haskey(p.neighbor_cells, id)

"""
    copied_cells(partition)

Get a `id=>cell` map of all cells that this partition needs in order to compute an update step. 
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
   get_cell_update(partition, cell_id)

Get the update for cell `cell_id` that this partition owns.
"""
get_cell_update(partition::CellGridPartition, cell_id) = partition.cells_update[cell_id]
get_cell_update(partition::FastCellGridPartition, cell_id) = partition.owned_update[cell_id]

"""
    store_cell_update!(partition, cell_id, cell_update)

Store `cell_update` for `cell_id` after computing it.

    store_cell_update!(partition, cell_id, cell_update, dim, s)

Store `cell_update` for `cell_id` only in dimension `dim` at splitting level `s` after computing it.
"""
function store_cell_update!(partition::CellGridPartition, id, Δu)
    partition.cells_update[id] = Δu
end

function store_cell_update!(partition::FastCellGridPartition, id, Δu)
    partition.owned_update[id] = Δu
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
        sizehint!(partition_compute_cells, owned_active_cells)
        # cells that OTHER partitions are responsible for but this partition needs to know about
        partition_shared_cells = Dict{Int,CELL_TYPE}()
        sizehint!(partition_shared_cells, shared_active_cells)
        for i ∈ global_covered_idxs
            id = global_cell_ids[i]
            id == 0 && continue
            if i ∈ global_owned_idxs
                partition_compute_cells[id] = global_active_cells[id]
            else
                partition_shared_cells[id] = global_active_cells[id]
            end
        end
        if show_info
            @info "Creating cell partition on grid ids..." partition_id = partition_id copied_idxs =
                global_covered_idxs global_owned_idxs local_compute_idxs =
                (range(locally_owned_x...), range(local_owned_y...)) owned_active_cells shared_active_cells
        end
        return FastCellGridPartition(
            partition_id,
            partition_compute_cells,
            partition_shared_cells,
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
            !owns_cell(p2, cell_id)
        end
    end
    computed_cells = mapreduce(p -> keys(p.owned_cells), union, p)
    return no_overlap && (computed_cells == keys(global_cells))
end

"""
    partition_neighbor_map(partitions)

Returns a `partition id => list of neighboring partition indices` given a vector of partitions.
"""
function partition_neighbor_map(partitions)
    return Dict([
        p.id => [
            p2.id for p2 ∈ Iterators.filter(partitions) do other
                return any(cell_id -> owns_cell(p, cell_id), other.neighbor_ids)
            end
        ] for p ∈ partitions
    ])
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

"""
    collect_cell_partitions!(global_cells, partitions)

Merges all of the updates for a collection of partitions into one dict for shared lookup.
Takes an existing dict and returns it after updating its contents.
"""
function collect_cell_partitions!(global_cells, partitions)
    foreach(partitions) do p
        collect_cell_partition!(global_cells, p)
    end
    return global_cells
end

"""
    collect_cell_partitions(cell_partitions, n_active_cells)

Merges a collection of partitions into one dict for shared lookup.
"""
function collect_cell_partitions(cell_partitions, n_active_cells)
    u_global = Dict{Int64,cell_type(first(cell_partitions))}()
    sizehint!(u_global, n_active_cells)
    return collect_cell_partitions!(u_global, cell_partitions)
end

function collect_cell_partition_update!(Δu_global, partition::CellGridPartition)
    data_region = owned_cell_ids(partition)
    for id ∈ data_region
        Δu_global[id] = partition.cells_update[id]
    end
end

function collect_cell_partition_update!(Δu_global, partition::FastCellGridPartition)
    merge!(Δu_global, partition.owned_update)
end

"""
    collect_cell_partition_updates!(Δu_global, n_active_cells)

Merges all of the updates for a collection of partitions into one dict for shared lookup.
Takes an existing dict and returns it after updating its contents.
"""
function collect_cell_partition_updates!(Δu_global, cell_partitions)
    foreach(cell_partitions) do p
        collect_cell_partition_update!(Δu_global, p)
    end
    return Δu_global
end

"""
    collect_cell_partition_updates(cell_paritions, n_active_cells)

Merges all of the updates for a collection of partitions into one dict for shared lookup.
"""
function collect_cell_partition_updates(cell_partitions, n_active_cells)
    Δu_global = Dict{Int64,update_dtype(first(cell_partitions))}()
    sizehint!(Δu_global, n_active_cells)
    return collect_cell_partition_updates!(Δu_global, cell_partitions)
end

# does not allocate as of now
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

function compute_partition_update_and_max_Δt!(
    partition::AbstractCellGridPartition,
    dim,
    splitting_level,
    boundary_conditions,
    gas::CaloricallyPerfectGas,
)
    Δt_max = typemax(numeric_dtype(partition))
    for cell_id ∈ owned_cell_ids(partition)
        cell = get_cell(partition, cell_id)
        nbrs = neighbor_cells(cell, copied_cells(partition), boundary_conditions, gas)
        cell_Δt_max, cell_Δu = compute_cell_update_and_max_Δt(cell, dim, nbrs, gas)
        Δt_max = min(Δt_max, cell_Δt_max)
        current_update = get_cell_update(partition, cell_id)
        store_cell_update!(
            partition,
            cell_id,
            partial_cell_update(current_update, cell_Δu, dim, splitting_level),
        )
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
    shared_keys = Iterators.filter(k -> owns_cell(src, k), dest.neighbor_ids)
    for k ∈ shared_keys
        count += 1
        dest.neighbors_update[k] = src.owned_update[k]
    end
    return count
end

function apply_partition_update!(partition::CellGridPartition, dim, Δt, split_level)
    for (k, v) ∈ partition.cells_update
        partition.cells_map[k] =
            update_cell(partition.cells_map[k], v, Δt, dim, split_level)
    end
end

function apply_partition_update!(partition::FastCellGridPartition, dim, Δt, split_level)
    for id ∈ partition.owned_ids
        Δu = partition.owned_update[id]
        partition.owned_cells[id] =
            update_cell(partition.owned_cells[id], Δu, Δt, dim, split_level)
    end
    for id ∈ partition.neighbor_ids
        Δu = partition.neighbors_update[id]
        partition.neighbor_cells[id] =
            update_cell(partition.neighbor_cells[id], Δu, Δt, dim, split_level)
    end
end

"""
    function step_cell_simulation_with_strang_splitting!(
       cell_partitions,
       partition_neighboring,
       Δt_maximum,
       boundary_conditions,
       cfl_limit,
       gas::CaloricallyPerfectGas,
    )

Advance the simulation that has been partitioned into `cell_partitions` one time step, limited by `Δt_maximum` and `cfl_limit`.
Requires a dict of `id=>idx` for partitions that share neighbor cells.

Performs the update step via Strang splitting!

Returns the time step size and the maximum update size in each dimension.
"""
function step_cell_simulation_with_strang_splitting!(
    cell_partitions,
    partition_neighboring,
    Δt_maximum,
    boundary_conditions,
    cfl_limit,
    gas::CaloricallyPerfectGas,
)
    T = numeric_dtype(eltype(cell_partitions))
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
            compute_partition_update_and_max_Δt!(
                cell_partition,
                1,
                1,
                boundary_conditions,
                gas,
            )
        end
    # propagate between neighboring partitions
    # and apply
    tforeach(cell_partitions) do p
        for src_idx ∈ partition_neighboring[p.id]
            propagate_updates_to!(p, cell_partitions[src_idx])
        end
        apply_partition_update!(p, 1, Δt / 2, 1)
    end
    # then in y
    tforeach(cell_partitions) do cell_partition
        compute_partition_update_and_max_Δt!(cell_partition, 2, 1, boundary_conditions, gas)
    end
    tforeach(cell_partitions) do p
        for src_idx ∈ partition_neighboring[p.id]
            propagate_updates_to!(p, cell_partitions[src_idx])
        end
        apply_partition_update!(p, 2, Δt, 1)
    end
    # then in x again
    tforeach(cell_partitions) do cell_partition
        compute_partition_update_and_max_Δt!(cell_partition, 1, 2, boundary_conditions, gas)
    end
    tforeach(cell_partitions) do p
        for src_idx ∈ partition_neighboring[p.id]
            propagate_updates_to!(p, cell_partitions[src_idx])
        end
        apply_partition_update!(p, 1, Δt / 2, 2)
    end
    return Δt
end

struct PartitionConvergenceMeta{T}
    partition_id::Int
    Δu_relative::NTuple{4,Vector{T}}
end

function PartitionConvergenceMeta(partition)
    T = numeric_dtype(partition)
    ncells = owned_cell_count(partition)
    bufs = ntuple(Returns(Vector{T}(undef, ncells)), 4)
    return PartitionConvergenceMeta(partition.id, bufs)
end

function compute_partition_convergence_measures!(meta, partition, Δt)
    for (i, id) ∈ enumerate(owned_cell_ids(partition))
        u = get_cell(partition, id).u
        Δu = total_update(get_cell_update(partition, id))[1]
        relative_abs_update = abs.((Δt * Δu) ./ u)
        for j ∈ 1:4
            meta.Δu_relative[j][i] = relative_abs_update[j]
        end
    end
    measures = map(meta.Δu_relative) do v
        sort!(v)
        i_guess = findfirst(isnan, v)
        v_notnan = isnothing(i_guess) ? @view(v[begin:end]) : @view(v[begin:i_guess-1])
        if isempty(v_notnan)
            return ntuple(Returns(zero(eltype(v))), 2)
        end
        mean_upd = mean(v_notnan)
        max_upd = maximum(v_notnan)
        return (mean_upd, max_upd)
    end
    mean_update = SVector(ntuple(i -> measures[i][1], 4))
    max_update = SVector(ntuple(i -> measures[i][2], 4))
    return mean_update, max_update
end

function test_for_convergence(mean_upd, max_upd)
    return norm(mean_upd) < 0.0001
end

##GLOBAL CELL GRID STUFF

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
