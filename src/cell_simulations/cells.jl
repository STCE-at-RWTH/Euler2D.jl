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
    A = (pt - circ.center)
    θ = atan(A[2] / A[1]) # ranges from -π/2 to π/2
    # switch to (0, 2π]
    if θ < 0
        θ += 2π
    end
    if A[1] < 0
        if A[2] >= 0
            # quadrant 2
            θ += π / 2
        else
            θ -= π / 2
        end
    end
    return θ
end

#