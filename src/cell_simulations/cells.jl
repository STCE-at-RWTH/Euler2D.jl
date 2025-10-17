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
    FVMCell{T}

Abstract data type for all cells in a Cartesian grid.
The underlying numeric data type is `T`.

All FVMCells _must_ provide the following methods:

 - `update_dtype(::Type{<:FVMCell})`
 - `cell_boundary_polygon(::FVMCell)`
 - `cell_volume(::FVMCell)` (defaults to `poly_area(cell_boundary_polygon(cell))`)
 - `phantom_neighbor`
 - `compute_cell_update_and_maximum_Δt(cell, neighbors, gas)`
 - `update_cell(cell, Δ, Δt)`
"""
abstract type FVMCell{T} end

numeric_dtype(::FVMCell{T}) where {T} = T
numeric_dtype(::Type{<:FVMCell{T}}) where {T} = T

update_dtype(::T) where {T<:FVMCell} = update_dtype(T)

cell_volume(cell::FVMCell) = poly_area(cell_boundary_polygon(cell))

"""
    is_cell_contained_by(cell1, poly)
    is_cell_contained_by(cell1, cell2)

Does `cell2` (or `poly`) contain `cell1`?
"""
function is_cell_contained_by(cell1, poly)
    cell_poly = cell_boundary_polygon(cell1)
    if cell_poly isa PlanePolygons.SizedClockwiseOrientedPolygon
        # help inference out a bit here
        return all(Tuple(edge_starts(cell_poly))) do pt
            return PlanePolygons.point_inside_strict(poly, pt)
        end
    else
        return all(edge_starts(cell_poly)) do pt
            return PlanePolygons.point_inside_strict(poly, pt)
        end
    end
end

function is_cell_contained_by(cell1, cell2::FVMCell)
    return is_cell_contained_by(cell1, cell_boundary_polygon(cell2))
end

"""
    is_cell_overlapping(cell1, poly)
    is_cell_overlapping(cell1, cell2)

Does `cell1` merely overlap `cell2` (or `poly`) (rather than being contained by it)?
"""
function is_cell_overlapping(cell1, poly)
    contained = is_cell_contained_by(cell1, poly)
    return (!contained && are_polygons_intersecting(cell_boundary_polygon(cell1), poly))
end

function is_cell_overlapping(cell1, cell2::FVMCell)
    return is_cell_overlapping(cell1, cell_boundary_polygon(cell2))
end

"""
    overlapping_cell_area(cell1, cell2)
    overlapping_cell_area(cell1, poly)

Get the overlapping area between two finite volume cells.
"""
function overlapping_cell_area(cell1, poly)
    isect = poly_intersection(cell_boundary_polygon(cell1), poly)
    return poly_area(isect)
end

function overlapping_cell_area(cell1, cell2::FVMCell)
    return overlapping_cell_area(cell1, cell_boundary_polygon(cell2))
end

"""
    RectangularFVMCell{T}

An FVM cell that is a rectangle.
Has fields `center` and `extent`.
"""
abstract type RectangularFVMCell{T} <: FVMCell{T} end

cell_volume(cell::RectangularFVMCell) = *(cell.extent...)

function cell_boundary_polygon(cell::RectangularFVMCell)
    c = cell.center
    dx, dy = cell.extent / 2
    return SClosedPolygon(
        c + SVector(dx, -dy),
        c + SVector(-dx, -dy),
        c + SVector(-dx, dy),
        c + SVector(dx, dy),
    )
end

"""
    FVMCellUpdate{T}

The data required to perform the update to an FVMCell.

Subtypes must provide:

- `get_update_info(Δ::FVMCellUpdate, dim)`
"""
abstract type FVMCellUpdate{T} end

function zero_cell_update(::T) where {T<:FVMCellUpdate}
    return zero_cell_update(T)
end

"""
    PrimalQuadCell{T}

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
 - `neighbors`: What are IDs of this cell's neighbors?
"""
struct PrimalQuadCell{T} <: RectangularFVMCell{T}
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

struct PrimalQuadCellStrangUpdate{T} <: FVMCellUpdate{T}
    Δu_x::NTuple{2,SVector{4,T}}
    Δu_y::SVector{4,T}
end

function zero_cell_update(::Type{<:PrimalQuadCellStrangUpdate{T}}) where {T}
    U = SVector{4,T}
    return PrimalQuadCellStrangUpdate((zero(U), zero(U)), zero(U))
end

function partial_cell_update(previous_Δu::PrimalQuadCellStrangUpdate, partial_Δu, dim, s)
    if dim == 1
        return @set previous_Δu.Δu_x[s] = partial_Δu
    end
    return @set previous_Δu.Δu_y = partial_Δu
end

"""
    TangentQuadCell{T, NSEEDS,PARAMCOUNT} 

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
 - `neighbors`: What are IDs of this cell's neighbors?
"""
struct TangentQuadCell{T,NSEEDS,PARAMCOUNT} <: RectangularFVMCell{T}
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

struct TangentQuadCellStrangUpdate{T,NSEEDS,NPARAMS}
    Δu_x::NTuple{2,SVector{4,T}}
    Δu_y::SVector{4,T}
    Δu̇_x::NTuple{2,SMatrix{4,NSEEDS,T,NPARAMS}}
    Δu̇_y::SMatrix{4,NSEEDS,T,NPARAMS}
end

function zero_cell_update(::Type{<:TangentQuadCellStrangUpdate{T,NS,NP}}) where {T,NS,NP}
    U = SVector{4,T}
    V = SMatrix{4,NS,T,NP}
    return TangentQuadCellStrangUpdate(
        (zero(U), zero(U)),
        zero(U),
        (zero(V), zero(V)),
        zero(V),
    )
end

function partial_cell_update(previous_Δu::TangentQuadCellStrangUpdate, partial_Δu, dim, s)
    if dim == 1
        @reset previous_Δu.Δu_x[s] = partial_Δu[1]
        return @set previous_Δu.Δu̇_x[s] = partial_Δu[2]
    end
    @reset previous_Δu.Δu_y = partial_Δu[1]
    return @set previous_Δu.Δu̇_y = partial_Δu[2]
end

n_seeds(::TangentQuadCell{T,N,P}) where {T,N,P} = N
n_seeds(::Type{TangentQuadCell{T,N,P}}) where {T,N,P} = N

@doc """
        numeric_dtype(cell)
        numeric_dtype(::Type{CELL_TYPE})

    Get the numeric data type associated with this cell.
    """ numeric_dtype

function update_dtype(::Type{PrimalQuadCell{T}}) where {T}
    return PrimalQuadCellStrangUpdate{T}
end

function update_dtype(::Type{TangentQuadCell{T,N,P}}) where {T,N,P}
    return TangentQuadCellStrangUpdate{T,N,P}
end

@doc """
    update_dtype(::Type{T<:FVMCell})

Get the cell update data type that must be enforced upon fetching the result of the computation tasks.
""" update_dtype

function update_cell(cell::PrimalQuadCell, Δu::PrimalQuadCellStrangUpdate, Δt, dim, s)
    if dim == 1 # x
        return @set cell.u = cell.u + Δt * Δu.Δu_x[s]
    end
    return @set cell.u = cell.u + Δt * Δu.Δu_y
end

function update_cell(cell::TangentQuadCell, Δu::TangentQuadCellStrangUpdate, Δt, dim, s)
    if dim == 1
        @reset cell.u = cell.u + Δt * Δu.Δu_x[s]
        return @set cell.u̇ = cell.u̇ + Δt * Δu.Δu̇_x[s]
    end
    @reset cell.u = cell.u + Δt * Δu.Δu_y
    return @set cell.u̇ = cell.u̇ + Δt * Δu.Δu̇_y
end

"""
    total_update(Δu)

Sum together all of the subcomponents of an update and return them as one value.
"""
function total_update(Δu::PrimalQuadCellStrangUpdate)
    return (sum(Δu.Δu_x) / length(Δu.Δu_x) + Δu.Δu_y,)
end

function total_update(Δu::TangentQuadCellStrangUpdate)
    return (
        sum(Δu.Δu_x) / length(Δu.Δu_x) + Δu.Δu_y,
        sum(Δu.Δu̇_x) / length(Δu.Δu̇_x) + Δu.Δu̇_y,
    )
end

@doc """
    update_cell(cell, Δu, Δt, dim, s)

Apply the update `Δu` in dimension `dim` at splitting level 's' (Strang splitting has 2 `x`-axis levels and 1 `y`-axis level)
""" update_cell

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

inward_normals(cell::RectangularFVMCell) = inward_normals(numeric_dtype(cell))
outward_normals(cell::RectangularFVMCell) = outward_normals(numeric_dtype(cell))

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

function _iface_speed(iface::Tuple{Int,T,T}, gas) where {T<:FVMCell}
    return max(abs.(interface_signal_speeds(iface[2].u, iface[3].u, iface[1], gas))...)
end

function maximum_cell_signal_speeds(
    interfaces::NamedTuple{(:north, :south, :east, :west)},
    gas::CaloricallyPerfectGas,
)
    # doing this with map allocated?!
    return (
        max(_iface_speed(interfaces.north, gas), _iface_speed(interfaces.south, gas)),
        max(_iface_speed(interfaces.east, gas), _iface_speed(interfaces.west, gas)),
    )
end

"""
    compute_cell_update_and_max_Δt(cell, dim, boundary_conditions, gas)

Computes the update (of type `update_dtype(typeof(cell))`) for a given cell.

Arguments
---
- `cell`
- `nbrs`: A `NamedTuple{(:north, :south, :east, :west)}` of (phantom) neighbors to `cell`
- `dim`: The dimension to compute the update in.

Returns
---
`(update, Δt_max)`: A tuple of the cell update in direction `dim` and the maximum time step size allowed by the CFL condition.
"""
function compute_cell_update_and_max_Δt(cell::PrimalQuadCell, dim, neighbors, gas)
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
    return (Δt_max, Δu[dim])
end

function compute_cell_update_and_max_Δt(
    cell::TangentQuadCell{T,N,P},
    dim,
    neighbors,
    gas,
) where {T,N,P}
    ifaces = (
        north = (2, cell, neighbors.north),
        south = (2, neighbors.south, cell),
        east = (1, cell, neighbors.east),
        west = (1, neighbors.west, cell),
    )
    a = maximum_cell_signal_speeds(ifaces, gas)
    Δt_max = min((cell.extent ./ a)...)
    ϕ_jvp = map(ifaces) do (dim, cell_L, cell_R)
        value, jvp = ϕ_hll_and_jvp(cell_L.u, cell_L.u̇, cell_R.u, cell_R.u̇, dim, gas)
        return value, hcat(jvp...)
    end
    Δx = map(ifaces) do (dim, cell_L, cell_R)
        (cell_L.extent[dim] + cell_R.extent[dim]) / 2
    end
    Δu = (
        inv(Δx.west) * ϕ_jvp.west[1] - inv(Δx.east) * ϕ_jvp.east[1],
        inv(Δx.south) * ϕ_jvp.south[1] - inv(Δx.north) * ϕ_jvp.north[1],
    )
    Δu̇ = (
        inv(Δx.west) * ϕ_jvp.west[2] - inv(Δx.east) * ϕ_jvp.east[2],
        inv(Δx.south) * ϕ_jvp.south[2] - inv(Δx.north) * ϕ_jvp.north[2],
    )
    return (Δt_max, (Δu[dim], Δu̇[dim]))
end
