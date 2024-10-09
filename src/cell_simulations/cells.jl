const _cell_iface_dirs = (:north, :south, :east, :west)
const _cell_iface_pairs = (north = :south, south = :north, east = :west, west = :east)
const _cell_iface_vtxs =
    (north = (:nw, :ne), south = (:sw, :se), east = (:se, :ne), west = (:sw, :nw))

function fuse_named_tuples(
    nt1::NamedTuple{NAMES,T1},
    nt2::NamedTuple{NAMES,T2},
) where {NAMES,T1,T2}
    new_values = ntuple(length(NAMES)) do i
        (nt1[NAMES[i]], nt2[NAMES[i]])
    end
    return NamedTuple{NAMES}(new_values)
end

@enum CellIFaceKind::UInt8 begin
    UNINITIALIZED
    NOT_ACTIVE
    CELL_BOUNDARY
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

_uninit_cell_iface(T) = CellIFace(UNINITIALIZED, 0, 0, zero(T))

is_iface_empty(cif) = cif.iface_kind == NOT_ACTIVE
is_iface_cut(cif) = cif.iface_kind == CELL_CELL_CUT

struct SketchedQuadCell{T}
    id::Int
    center::SVector{2,T}
    extent::SVector{2,T}
end

struct QuadCell{T,Q1<:Density,Q2<:MomentumDensity,Q3<:EnergyDensity}
    id::Int
    V::T
    u::ConservedProps{2,T,Q1,Q2,Q3}

    center::SVector{2,T}
    extent::SVector{2,T}
    ifaces::NamedTuple{_cell_iface_dirs,NTuple{4,CellIFace{T}}}
end

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
    @show α, β
    res = sincos((α + β) / 2)
    return SVector(res[2], res[1])
end

##

function is_point_on(pt, obs)
    return path_fn(obs)(pt...) ≈ 0
end

function classify_interfaces(cell, obstacles)
    vtxs = vertices(cell)
    pts_inside = map(Base.Fix2(is_point_in, obstacle), vtxs)
    @show pts_inside
    pts_on = map(Base.Fix2(is_point_on, obstacle), vtxs)
    @show pts_on
    return map(_cell_iface_vtxs) do (v1, v2)
        if getfield(pts_inside, v1) && getfield(pts_inside, v2)
            return NOT_ACTIVE
        elseif getfield(pts_on, v1) && getfield(pts_on, v2)
            return CELL_BOUNDARY # is this correct.?
        elseif !getfield(pts_inside, v1) && !getfield(pts_inside, v2)
            return CELL_CELL_FULL
        else
            return CELL_CELL_CUT
        end
    end
end

##

function create_cell_sketch(bounds_x::T, bounds_y::T, ncells_x, ncells_y) where {T}
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
    else
        return cells_grid[neighbor_idx]
    end
end

function immerse_boundaries(cells_sketch, boundary_conditions, immersed_boundaries)
    ncells = size(cells_sketch)

    _cell_neighbor_offsets = (
        north = CartesianIndex(0, 1),
        south = CartesianIndex(0, -1),
        east = CartesianIndex(1, 0),
        west = CartesianIndex(-1, 0),
    )

    for idx ∈ CartesianIndices(cells_sketch)
        (i, j) = Tuple(idx)
        vtx_ins = map(vertices(cells_sketch[idx])) do vtx
            in_id = 0
            for ib ∈ immersed_boundaries
                if is_point_in(vtx, ib)
                    in_id = ib.id
                end
            end
        end

        vtx_ons = map(vertices(cells_sketch[idx])) do vtx
            on_id = 0
            for ib ∈ immersed_boundaries
                if is_point_on(pt, ib)
                    on_id = ib.id
                end
            end
        end

        cell_ifaces = map(
            fuse_named_tuples(_cell_neighbor_offsets, _cell_iface_vtxs),
        ) do (offset, sym_L, sym_R)

            in_L = getfield(vtx_ins, sym_L)
            in_R = getfield(vtx_ins, sym_R)
            on_L = getfield(vtx_ons, sym_L)
            on_R = getfield(vtx_ons, sym_R)

            # 
            if in_L == 0 && in_R == 0 && on_L == 0 && on_R == 0

            end
        end
    end
end

function create_quadcell_grid(cells_sketch, immersed_boundaries)
    for i in CartesianIndices(cells_sketch)
        pts_in_boundary = map(vertices(ce))
    end
end

##

c1 = Circular(1, SVector(0.0, 0.0), 1.0)
t = SketchedQuadCell(1, SVector(0.5, 0.0), SVector(2.0, 0.125))
map(Base.Fix1(central_angle_of, c1), vertices(t))
ifaces_t = classify_interfaces(t, c1)

# HACK we assume that if both vertices are _inside_ of obstacles
# then the cell face is a hard wall

function make_ifaces() end