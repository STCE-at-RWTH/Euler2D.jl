abstract type CellBoundary end

struct FacesCell{T} <: CellBoundary
    normal::SVector{2, T}
    other::Ref{QuadCell{T}}
end

struct FacesBoundary{T} <: CellBoundary 
    normal::SVector{2, T}
    boundary_condition::EdgeBoundary
end

NeighborsList = @NamedTuple begin
    north::CellBoundary
    south::CellBoundary
    east::CellBoundary
    west::CellBoundary
end

struct QuadCell
    id::Int
    u::ConservedProps
    neighbors::NeighborsList
end

