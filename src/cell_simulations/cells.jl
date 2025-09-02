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

 - `update_dtype(::FVMCell)`
 - `cell_boundary_polygon(::FVMCell)`
 - `cell_volume(::FVMCell)` (defaults to `poly_area(cell_boundary_polygon(cell))`)
"""
abstract type FVMCell{T} end

numeric_dtype(::FVMCell{T}) where {T} = T
numeric_dtype(::Type{FVMCell{T}}) where {T} = T

cell_volume(cell::FVMCell) = poly_area(cell_boundary_polygon(cell))

"""
    is_cell_contained_by(cell1, poly)
    is_cell_contained_by(cell1, cell2)

Does `cell2` (or `poly`) contain `cell1`?
"""
function is_cell_contained_by(cell1, poly)
    return all(edge_starts(cell_boundary_polygon(cell1))) do pt
        return PlanePolygons.point_inside_strict(poly, pt)
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

abstract type RectangularFVMCell{T} <: FVMCell{T} end

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

n_seeds(::TangentQuadCell{T,N,P}) where {T,N,P} = N
n_seeds(::Type{TangentQuadCell{T,N,P}}) where {T,N,P} = N

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

inward_normals(cell::RectangularFVMCell) = inward_normals(numeric_dtype(cell))
outward_normals(cell::RectangularFVMCell) = outward_normals(numeric_dtype(cell))

"""
    TangentPolyCell{T, NV}


"""
struct TangentPolyCell{T,NV,NSEEDS,NTANGENTS} <: FVMCell{T}
    id::Int
    boundary::SClosedPolygon{T,NV}
    u::SVector{4,T}
    u̇::SMatrix{4,NSEEDS,T,NTANGENTS}
    neighbors::NTuple{NV,Tuple{CellNeighboring,Int}}
end

cell_boundary_polygon(cell::TangentPolyCell) = cell.boundary

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
