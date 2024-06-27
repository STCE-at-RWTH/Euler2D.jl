using LinearAlgebra
using ShockwaveProperties: MomentumDensity, EnergyDensity
using Unitful: Density

mutable struct RegularQuadCell{T,Q1<:Density,Q2<:MomentumDensity,Q3<:EnergyDensity}
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

const _cardinal_dirs_dims = (north = 2, south = 2, east = 1, west = 1)
const _cardinal_dirs_opposites =
    (north = :south, south = :north, east = :west, west = :east)
const _cardinal_dirs_reverse_bcs = (north = true, south = false, east = false, west = true)

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

const _cell_neighbor_offsets = (
    north = CartesianIndex(0, 1),
    south = CartesianIndex(0, -1),
    east = CartesianIndex(1, 0),
    west = CartesianIndex(-1, 0),
)

function cell_neighbor_status(i, cell_ids, active_mask)
    idx = CartesianIndices(size(cell_ids))[i]
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

function quadcell_list_and_id_grid(u0, bounds, ncells, obstacles)
    faces = map(zip(bounds, ncells)) do (b, n)
        range(b...; length = n + 1)
    end
    centers = map(faces) do f
        f[1:(end-1)] .+ step(f) / 2
    end
    dx, dy = step.(centers)
    active_mask = active_cell_mask(centers..., obstacles)
    active_ids = active_cell_ids_from_mask(active_mask)
    cell_list = map(eachindex(IndexCartesian(), active_ids)) do idx
        (i, j) = Tuple(idx)
        x_i = centers[1][i]
        y_j = centers[2][j]
        u = u0(x_i, y_j)
        neighbors = cell_neighbor_status(idx, active_ids, active_mask)

        RegularQuadCell(active_ids[idx], idx, (x_i, y_j), u, neighbors)
    end
    return cell_list, active_ids
end

function phantom_neighbor(cell, active_cells, dir, bc, gas)
    boundary_neighbors_u = Matrix{dtype(cell),2}(undef, (4, nneighbors(bc)))
    opposite_dir = _cardinals_dirs_opposites[dir]
    cur = Ref(cell)
    for col ∈ eachcol(boundary_neighbors_u)
        col .= state_to_vector(cur[].u)
        cur = Ref(active_cells[cell.neighbors[opposite_dir]])
    end
    dim = _cardinal_dirs_dims[dir]
    reverse_bc = _cardinal_dirs_reverse_bcs[dir] && reverse_right_edge(bc)
    if reverse_bc
        boundary_neighbors_u[dim+1, :] .*= -1
    end
    phantom = phantom_cell(bc, boundary_neighbors_u, dim, gas)
    if reverse_bc
        phantom[dim+1] *= -1
    end
    return SVector{4}(phantom)
end

function single_cell_neighbor_data(
    id,
    active_cells,
    boundary_conditions,
    gas::CaloricallyPerfectGas,
)
    center_cell = active_cells[id]
    return map(pairs(center_cell.neighbors)) do (dir, (kind, id))
        if kind == :boundary
            return phantom_neighbor(
                center_cell,
                active_cells,
                dir,
                boundary_conditions[id],
                gas,
            )
        else
            return state_to_vector(active_cells[id].u)
        end
    end
end

function compute_cell_update(cell, neighbors)

struct CellBasedEulerSim{T}
    ncells::Tuple{T,T}
    n_active_cells::Int
    bounds::NTuple{2,Tuple{T,T}}
    n_tsteps::Int
    tsteps::Vector{T}
    cells::Array{RegularQuadCell,2}
end
