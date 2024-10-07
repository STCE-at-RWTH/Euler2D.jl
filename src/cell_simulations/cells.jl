const _cell_iface_dirs = (:north, :south, :east, :west)
const _cell_iface_pairs = (north = :south, south = :north, east = :west, west = :east)
const _cell_iface_vtxs =
    (north = (:nw, :ne), south = (:sw, :se), east = (:se, :ne), west = (:sw, :nw))

@enum CellIFaceKind::UInt8 begin
    NO_IFACE
    CELL_CELL_FULL
    CELL_CELL_CUT
    CELL_BOUNDARY
end

struct CellIFace{T}
    iface_kind::CellIFaceKind

    left_id::Int
    right_id::Int

    iface_area::T
end

is_iface_empty(cif) = cif.iface_kind == NO_IFACE
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
    center::SVector{2,T}
    rsqr::T
end

function is_point_in(pt, circle::Circular)
    return sum((pt - circle.center) .^ 2) < circle.rsqr
end

function is_point_on(pt, circle::Circular)
    return (sum(pt - circle.center) .^ 2) ≈ circle.rsqr
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
    res = sincos((α+β)/2)
    return SVector(res[2], res[1])
end

##

function classify_interfaces(cell, obstacle)
    vtxs = vertices(cell)
    pts_inside = map(Base.Fix2(is_point_in, obstacle), vtxs)
    pts_on = map(Base.Fix2(is_point_on, obstacle), vtxs)
    return map(_cell_iface_vtxs) do (v1, v2)
        if getfield(pts_inside, v1) && getfield(pts_inside, v2)
            return NO_IFACE
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

c1 = Circular(SVector(0., 0.), 1.0)
central_angle_of(c1, SVector(-1., -0.5)) |> rad2deg

