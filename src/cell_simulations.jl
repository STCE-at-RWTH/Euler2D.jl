using Euler2D
using Euler2D: nneighbors, phantom_cell, reverse_right_edge, ϕ_hll
using LinearAlgebra
using ShockwaveProperties
using ShockwaveProperties: MomentumDensity, EnergyDensity
using StaticArrays
using Unitful
using Unitful: Density

struct RegularQuadCell{T,Q1<:Density,Q2<:MomentumDensity,Q3<:EnergyDensity}
    id::Int
    idx::CartesianIndex{2}
    center::Tuple{T,T}
    u::ConservedProps{2,T,Q1,Q2,Q3}
    # either (:boundary, :cell)
    # and then the ID of the appropriate boundary
    neighbors::NamedTuple{(:north, :south, :east, :west),NTuple{4,Tuple{Symbol,Int}}}
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

abstract type Obstacle end

struct CircularObstacle{T} <: Obstacle
    center::SVector{2,T}
    radius::T
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
        (p2 - p1) ⋅ (pt - p1) > 0
    end
end

function active_cell_mask(cell_centers_x, cell_centers_y, obstacles)
    return map(Iterators.product(cell_centers_x, cell_centers_y)) do pt
        p = SVector(pt...)
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
            return (:boundary, 1)
        elseif neighbor[1] > size(cell_ids)[1]
            return (:boundary, 2)
        elseif neighbor[2] < 1
            return (:boundary, 3)
        elseif neighbor[2] > size(cell_ids)[2]
            return (:boundary, 4)
        elseif active_mask[neighbor]
            return (:cell, cell_ids[neighbor])
        else
            return (:boundary, 5)
        end
    end
end

props_dtype(::ConservedProps{N,T,U1,U2,U3}) where {N,T,U1,U2,U3} = T
props_unitstypes(::ConservedProps{N,T,U1,U2,U3}) where {N,T,U1,U2,U3} = (U1, U2, U3)
function quadcell_dtype(::ConservedProps{N,T,U1,U2,U3}) where {N,T,U1,U2,U3}
    return RegularQuadCell{T,U1,U2,U3}
end

function quadcell_list_and_id_grid(u0, bounds, ncells, obstacles)
    centers = map(zip(bounds, ncells)) do (b, n)
        v = range(b...; length = n + 1)
        return v[1:end-1] .+ step(v) / 2
    end

    dx, dy = step.(centers)
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
        cell_list[j] = RegularQuadCell(j, i, (x_i, y_j), u0_grid[i], neighbors)
    end
    return cell_list, active_ids
end

function phantom_neighbor(id, active_cells, dir, bc, gas)
    cardinal_dirs_opposites = (north = :south, south = :north, east = :west, west = :east)
    cardinal_dirs_reverse_bcs = (north = true, south = false, east = true, west = false)
    cardinal_dirs_dims = (north = 2, south = 2, east = 1, west = 1)
    opposite_dir = cardinal_dirs_opposites[dir]

    T = dtype(active_cells[id])
    dim = cardinal_dirs_dims[dir]
    reverse_bc = cardinal_dirs_reverse_bcs[dir] && reverse_right_edge(bc)
    velocity_scaling = SVector(ntuple(Returns(one(T)), 4))
    if reverse_bc
        setindex(velocity_scaling, -one(T), dim + 1)
    end

    boundary_neighbors_u = MMatrix{4,nneighbors(bc),T}(undef)
    cur = Ref(active_cells[id])
    for j ∈ 1:nneighbors(bc)
        boundary_neighbors_u[:, j] .= state_to_vector(cur[].u) .* velocity_scaling
        cur = Ref(active_cells[cur[].neighbors[opposite_dir][2]])
    end

    phantom = phantom_cell(bc, boundary_neighbors_u, dim, gas) .* velocity_scaling
    return SVector(phantom)
end

function single_cell_neighbor_data(
    id,
    active_cells,
    boundary_conditions,
    gas::CaloricallyPerfectGas,
)
    neighbors = active_cells[id].neighbors
    map(ntuple(i -> ((keys(neighbors)[i], neighbors[i])), 4)) do (dir, (kind, id))
        if kind == :boundary
            return phantom_neighbor(id, active_cells, dir, boundary_conditions[id], gas)
        else
            return state_to_vector(active_cells[id].u)
        end
    end |> NamedTuple{(:north, :south, :east, :west)}
end

function compute_cell_update(cell_data, neighbor_data, Δx, Δy, gas)
    ifaces = (
        north = (2, Ref(cell_data), Ref(neighbor_data.north)),
        south = (2, Ref(neighbor_data.south), Ref(cell_data)),
        east = (1, Ref(cell_data), Ref(neighbor_data.east)),
        west = (1, Ref(neighbor_data.west), Ref(cell_data)),
    )

    maximum_signal_speed = mapreduce(max, ifaces) do (dim, uL, uR)
        max(abs.(interface_signal_speeds(uL[], uR[], dim, gas))...)
    end

    ϕ = map(ifaces) do (dim, uL, uR)
        ϕ_hll(uL[], uR[], dim, gas)
    end
    # we want to write this as u_next = u + Δt * diff
    Δu = inv(Δx) * (ϕ.west - ϕ.east) + inv(Δy) * (ϕ.south - ϕ.north)

    return (cell_speed = maximum_signal_speed, cell_Δu = Δu)
end

function compute_next_u(cell, Δt, Δu)
    u_next = state_to_vector(active_cells[i]) + Δt * update_infos[i].cell_Δu
    RegularQuadCell(cell.id, cell.idx, cell.center, u_next, cell.neightbors)
end

function step_cell_simulation!(
    cells_next,
    active_cells,
    boundary_conditions,
    cfl_limit,
    Δx,
    Δy,
    gas::CaloricallyPerfectGas;
)
    T = dtype(active_cells[1])
    step_tasks = map(enumerate(active_cells)) do (idx, cell)
        n_data = single_cell_neighbor_data($idx, active_cells, $boundary_conditions, $gas)
        c_data = state_to_vector(cell.u)
        Threads.@spawn begin
            return compute_cell_update(c_data, n_data, $Δx, $Δy, $gas)
        end
    end
    update_infos::Vector{Tuple{Int,T,SVector{4,T}}} = fetch.(step_tasks)
    a_max = mapreduce(el -> el.cell_speed, max, update_infos)
    Δt = cfl_limit * Δx / a_max
    for i ∈ eachindex(update_infos, cells_next, active_cells)
        u_next = state_to_vector(active_cells[i]) + Δt * update_infos[i].cell_Δu
        cells_next[i] = RegularQuadCell(active_cells[i].id)
    end
end

struct CellBasedEulerSim{T}
    ncells::Tuple{T,T}
    n_active_cells::Int
    bounds::NTuple{2,Tuple{T,T}}
    n_tsteps::Int
    tsteps::Vector{T}
    cells::Array{RegularQuadCell,2}
end
