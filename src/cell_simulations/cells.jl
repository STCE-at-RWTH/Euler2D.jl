using Unitful
using Unitful: Density
using ShockwaveProperties
using ShockwaveProperties: MomentumDensity, EnergyDensity
using StaticArrays

const _cell_iface_dirs = (:north, :south, :east, :west)
const _cell_iface_self = (north = :north, south = :south, east = :east, west = :west)
const _cell_iface_pairs = (north = :south, south = :north, east = :west, west = :east)
const _cell_iface_vtxs =
    (north = (:nw, :ne), south = (:sw, :se), east = (:se, :ne), west = (:sw, :nw))
const _cell_neighbor_offsets = (
    north = CartesianIndex(0, 1),
    south = CartesianIndex(0, -1),
    east = CartesianIndex(1, 0),
    west = CartesianIndex(-1, 0),
)

merge_values_tuple(arg1, arg2) = (arg1, arg2)
merge_values_tuple(arg1::Tuple, arg2) = (arg1..., arg2)
merge_values_tuple(arg1, arg2::Tuple) = (arg1, arg2...)
merge_values_tuple(arg1::Tuple, arg2::Tuple) = (arg1..., arg2...)

"""
    merge_named_tuples(nt1::NamedTuple{NAMES}, nt2::NamedTuple{NAMES}, nts::NamedTuple{NAMES}...)

Merge the values of the provided named tuples. Will flatten any tuple fields.
"""
function merge_named_tuples(nt1::NamedTuple{NAMES}, nt2::NamedTuple{NAMES}) where {NAMES}
    new_values = ntuple(length(NAMES)) do i
        merge_values_tuple(nt1[NAMES[i]], nt2[NAMES[i]])
    end
    return NamedTuple{NAMES}(new_values)
end

function merge_named_tuples(nt1::NamedTuple{NAMES}, nts::NamedTuple{NAMES}...) where {NAMES}
    return merge_named_tuples(merge_named_tuples(nt1, nts[1]), nts[2:end]...)
end

@enum CellIFaceKind::UInt8 begin
    NOT_ACTIVE
    CELL_BOUNDARY_FULL
    CELL_BOUNDARY_CUT
    CELL_CELL_FULL
    CELL_CELL_CUT_LEFT_ACTIVE
    CELL_CELL_CUT_RIGHT_ACTIVE
end

struct CellIFace{T}
    iface_kind::CellIFaceKind

    left_id::Int
    right_id::Int

    iface_area::T
end

function CellIFace(
    iface_direction::Symbol,
    kind::CellIFaceKind,
    self_id::Int,
    other_id::Int,
    iface_area::T,
) where {T}
    if iface_direction == :north || iface_direction == :east
        return CellIFace(kind, self_id, other_id, iface_area)
    else
        return CellIFace(kind, other_id, self_id, iface_area)
    end
end

is_iface_empty(cif) = cif.iface_kind == NOT_ACTIVE
is_iface_cut(cif) = cif.iface_kind == CELL_CELL_CUT

struct SketchedQuadCell{T}
    id::Int
    center::SVector{2,T}
    extent::SVector{2,T}
end

struct SketchedQuadCellWithBoundary{T}
    id::Int
    V::T
    center::SVector{2,T}
    extent::SVector{2,T}
    ifaces::NamedTuple{_cell_iface_dirs,NTuple{4,CellIFace{T}}}
end

function SketchedQuadCellWithBoundary(cell::SketchedQuadCell{T}, ifaces, volume) where {T}
    return SketchedQuadCellWithBoundary(cell.id, volume, cell.center, cell.extent, ifaces)
end

numeric_dtype(::SketchedQuadCell{T}) where {T} = T
numeric_dtype(::Type{SketchedQuadCell{T}}) where {T} = T

numeric_dtype(::SketchedQuadCellWithBoundary{T}) where {T} = T
numeric_dtype(::Type{SketchedQuadCellWithBoundary{T}}) where {T} = T

struct QuadCell{T,Q1<:Density{T},Q2<:MomentumDensity{T},Q3<:EnergyDensity{T}}
    id::Int
    V::T
    u::ConservedProps{2,T,Q1,Q2,Q3}

    center::SVector{2,T}
    extent::SVector{2,T}
    ifaces::NamedTuple{_cell_iface_dirs,NTuple{4,CellIFace{T}}}
end

numeric_dtype(::QuadCell{T,Q1,Q2,Q3}) where {T,Q1,Q2,Q3} = T
numeric_dtype(::Type{QuadCell{T,Q1,Q2,Q3}}) where {T,Q1,Q2,Q3} = T

function is_point_in(pt, cell)
    A = abs.(pt - cell.center)
    return all(A .< cell.extent / 2)
end

function is_point_on(pt, cell)
    A = abs.(pt - cell.center)
    return (
        (A[1] ≈ cell.extent[1] / 2 && A[2] <= cell.extent[2] / 2) ||
        (A[2] ≈ cell.extent[2] / 2 && A[1] <= cell.extent[1] / 2)
    )
end

struct ImmersedBoundarySection{T}
    parent_id::Int
    segment_length::T
    cell_avg_normal::SVector{2,T}
end

function vertices(cell)
    (dx, dy) = cell.extent / 2
    return (
        se = cell.center + @SVector([dx, -dy]),
        sw = cell.center + @SVector([-dx, -dy]),
        nw = cell.center + @SVector([-dx, dy]),
        ne = cell.center + @SVector([dx, dy]),
    )
end

extent(cell) = cell.extent

function full_iface_area(cell, dir)
    if dir == :north || dir == :south
        return cell.extent[1]
    else
        return cell.extent[2]
    end
end

function intersection_point(vtx_left, vtx_right, dir, immersed_boundary)
    if dir == :north || dir == :south
        @assert vtx_left[2] == vtx_right[2]
        x = find_intersection_x(vtx_left[2], immersed_boundary, vtx_left[1], vtx_right[1])
        return SVector(x, vtx_left[2])
    else
        @assert vtx_left[1] == vtx_right[1]
        y = find_intersection_y(vtx_left[1], immersed_boundary, vtx_left[2], vtx_right[2])
        return SVector(vtx_left[1], y)
    end
end

function interface_area(iface_L, iface_R, dir)
    if dir == :north || dir == :south
        return iface_R[1] - iface_L[1]
    else
        return iface_R[2] - iface_R[2]
    end
end

struct Circular{T}
    id::Int
    center::SVector{2,T}
    rsqr::T
end

function path_fn(obs::Circular)
    return (x, y) -> sum((SVector(x, y) - obs.center) .^ 2) - obs.rsqr
end

function is_point_in(pt, circle::Circular)
    return sum((pt - circle.center) .^ 2) < circle.rsqr
end

function find_intersection_x(y, c::Circular, xmin, xmax)
    x1 = xmin - c.center[1]
    x2 = xmax - c.center[1]
    y_tick = y - c.center[2]
    x = sqrt(c.rsqr - y_tick^2)
    if x1 < x < x2
        return x + c.center[1]
    else
        return -x + c.center[1]
    end
end

function find_intersection_y(x, c::Circular, ymin, ymax)
    y1 = ymin - c.center[2]
    y2 = ymax - c.center[2]
    x_tick = x - c.center[1]
    y = sqrt(c.rsqr - x_tick^2)
    if y1 < y < y2
        return y + c.center[2]
    else
        return -y + c.center[2]
    end
end

function central_angle_of(circ, pt)
    A = pt - circ.center
    θ = atan(A[2], A[1]) # ranges from -π to π, according to Julia docs
    # switch to (0, 2π]
    if θ < 0
        θ += 2π
    end
    return θ
end

function average_normal_on(curve::Circular, pt1, pt2)
    α = central_angle_of(curve, pt1)
    β = central_angle_of(curve, pt2)
    res = sincos((α + β) / 2)
    return SVector(res[2], res[1])
end

##

function is_point_on(pt, obs)
    return path_fn(obs)(pt...) ≈ 0
end

##

function create_cell_sketch(
    bounds_x::Tuple{T,T},
    bounds_y::Tuple{T,T},
    ncells_x,
    ncells_y,
) where {T}
    bounds = (bounds_x, bounds_y)
    ncells = (ncells_x, ncells_y)
    centers = map(zip(bounds, ncells)) do (b, n)
        v = range(b...; length = n + 1)
        return v[1:end-1] .+ step(v) / 2
    end
    extent = SVector{2}(step.(centers)...)
    sketch = Array{SketchedQuadCell{T},2}(undef, ncells)
    for (i, j) ∈ Iterators.product(1:ncells_x, 1:ncells_y)
        idx = CartesianIndex(i, j)
        pt = SVector(centers[1][i], centers[2][j])
        sketch[i, j] = SketchedQuadCell(LinearIndices(ncells)[idx], pt, extent)
    end
    return sketch
end

function is_neighbor_outside(neighbor_idx, cells_grid)
    return (
        neighbor_idx[1] < 1 ||
        neighbor_idx[1] > size(cells_grid)[1] ||
        neighbor_idx[2] < 1 ||
        neighbor_idx[2] > size(cells_grid)[2]
    )
end

function neighbor_id(neighbor_idx, cells_grid)
    if neighbor_idx[1] < 1
        # WEST
        return 4
    elseif neighbor_idx[1] > size(cells_grid)[1]
        # EAST
        return return 3
    elseif neighbor_idx[2] < 1
        # SOUTH
        return 2
    elseif neighbor_idx[2] > size(cells_grid)[2]
        # NORTH
        return 1
    else
        return cells_grid[neighbor_idx].id
    end
end

"""
    cell_interface

Create the appropriate cell interface given:
- `parent`: The cell to which this interface "belongs"
- `dir`: which face of `parent`?
- `neighbor_out`: Is the neighboring cell in direction `dir` inside or outside the computational grid?
- `n_id`: what is the ID of the neighboring object in direction `dir`?
- `vtxs`: What are the endpoints of this interface _before_ any cutting takes place?
- `ons`: Are the vertices `vtxs` *ON* any of the boundary structures in `immersed_boundaries`?
- `ins`: Aare the vertices `vtxs` *IN* any of the boundary structures in `immersed_boundaries`?

"""
function cell_interface(
    parent,
    dir,
    neighbor_out,
    n_id,
    vtxs,
    ons,
    ins,
    immersed_boundaries,
)
    on_L, on_R = ons
    in_L, in_R = ins
    vtx_L, vtx_R = vtxs
    #@info "Making cell interface for..." id=parent.id on_L on_R in_L in_R vtx_L vtx_R
    # interface is full size and active
    # we just need to get the neighbor ID
    if in_L == 0 && in_R == 0 && on_L == 0 && on_R == 0
        iface_area = full_iface_area(parent, dir)
        iface_type = if neighbor_out
            CELL_BOUNDARY_FULL
        else
            CELL_CELL_FULL
        end
        return CellIFace(dir, iface_type, parent.id, n_id, iface_area)
    end
    # cell is not ON any boundary
    if (on_L == 0 && on_R == 0)
        if in_L == 0 && in_R > 0
            pt = intersection_point(vtx_L, vtx_R, dir, immersed_boundaries[in_R])
            A = interface_area(vtx_L, pt, dir)
            return CellIFace(dir, CELL_CELL_CUT_LEFT_ACTIVE, parent.id, n_id, A)
        elseif in_R == 0 && in_L > 0
            pt = intersection_point(vtx_L, vtx_R, dir, immersed_boundaries[in_L])
            A = interface_area(pt, vtx_R, dir)
            return CellIFace(dir, CELL_CELL_CUT_LEFT_ACTIVE, parent.id, n_id, A)
        else #in_R > 0 && in_L > 0
            other_id = if in_R ≠ in_L
                # HACK we assume that if both vertices are inside of different obstacles
                # then the cell face is behind the boundary
                # this could get weird
                @error(
                    "Cell is intersected by multiple obstacles... refine grid or correct obstacle definitions.",
                    cell_id = parent.id,
                    in_R = in_R,
                    in_L = in_L,
                )
                0
            else
                in_R
            end
            # NOT_ACTIVE + OTHER_ID == 0 means we do a linear interpolation later
            A = full_iface_area(parent, dir)
            return CellIFace(dir, NOT_ACTIVE, parent.id, other_id, A)
        end
    else # cell is ON at least one boundary
        if on_L > 0 && in_R == on_L
            A = full_iface_area(parent, dir)
            return CellIFace(dir, NOT_ACTIVE, parent.id, on_L, A)
        elseif on_R > 0 && in_L == on_R
            A = full_iface_area(parent, dir)
            return CellIFace(dir, NOT_ACTIVE, parent.id, on_R, A)
        else
            # both corners on boundary
        end
    end
end

function immerse_boundaries(cells_sketch, immersed_boundaries)
    n_active_cells = mapreduce(+, cells_sketch) do c
        v = map(vertices(c)) do pt
            return any(immersed_boundaries) do b
                is_point_in(pt, b)
            end
        end
        !all(v)
    end
    @info n_active_cells
    T = numeric_dtype(eltype(cells_sketch))
    cells = Dict{Int,SketchedQuadCellWithBoundary{T}}()
    sizehint!(cells, n_active_cells)
    for idx ∈ CartesianIndices(cells_sketch)
        vtxs = vertices(cells_sketch[idx])
        vtx_ins = map(vtxs) do vtx
            in_id = 0
            for ib ∈ immersed_boundaries
                if is_point_in(vtx, ib)
                    in_id = ib.id
                end
            end
            return in_id
        end

        if all(>(0), vtx_ins)
            continue
        end

        vtx_ons = map(vtxs) do vtx
            on_id = 0
            for ib ∈ immersed_boundaries
                if is_point_on(vtx, ib)
                    on_id = ib.id
                end
            end
            return on_id
        end

        cell_ifaces = map(
            merge_named_tuples(_cell_iface_self, _cell_neighbor_offsets, _cell_iface_vtxs),
        ) do (dir, offset, sym_L, sym_R)
            ons = vtx_ons[(sym_L, sym_R)]
            ins = vtx_ins[(sym_L, sym_R)]
            neighbor_out = is_neighbor_outside(idx + offset, cells_sketch)
            n_id = neighbor_id(idx + offset, cells_sketch)
            return cell_interface(
                cells_sketch[idx],
                dir,
                neighbor_out,
                n_id,
                vtxs[(sym_L, sym_R)],
                ons,
                ins,
                immersed_boundaries,
            )
        end
        cells[cells_sketch[idx].id] = SketchedQuadCellWithBoundary(cells_sketch[idx], cell_ifaces, 0.0)
    end
    return cells
end

##

c1 = Circular(1, SVector(0.0, 0.0), 1.0)
t = SketchedQuadCell(1, SVector(0.0, 0.0), SVector(1.0, 1.0))

map(Base.Fix1(central_angle_of, c1), vertices(t))

tgrid = create_cell_sketch((-2.0, 2.0), (-2.0, 2.0), 10, 10)

test_cells = immerse_boundaries(tgrid, [c1])