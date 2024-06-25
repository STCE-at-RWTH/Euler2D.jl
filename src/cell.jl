abstract type QuadCellFace end

struct CellCellFace{T} <: QuadCellFace
    normal::SVector{2, T}
    other::Ref{QuadCell{T}}
end

struct CellBoundaryFace{T} <: QuadCellFace 
    normal::SVector{2, T}
    boundary_condition::EdgeBoundary
end

QuadCellIFaces = NamedTuple{(:north, :south, :east, :west), <:NTuple{4, QuadCellFace}}

struct Neighbor{T}
    id
end

struct QuadCell{T}
    id::Int
    u::ConservedProps
    neighbors::NTuple{4, Int}
end

inward_normals(QuadCell{})
