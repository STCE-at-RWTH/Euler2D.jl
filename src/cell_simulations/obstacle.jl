abstract type Obstacle end

struct CircularObstacle{T} <: Obstacle
    center::SVector{2,T}
    radius::T
end

function CircularObstacle(center, radius)
    CircularObstacle(SVector{2}(center...), radius)
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
    return all(abs.(Δx) .<= s.extent / 2)
end

struct TriangularObstacle{T} <: Obstacle
    points::NTuple{3,SVector{2,T}}
end

"""
    TriangularObstacle(points)

Create a triangular obstacle from a clockwise-oriented list of its vertices.
"""
function TriangularObstacle(pts...)
    return TriangularObstacle(tuple((SVector{2}(p) for p ∈ pts)...))
end

function point_inside(s::TriangularObstacle, pt)
    return all(zip(s.points, s.points[[2, 3, 1]])) do (p1, p2)
        R = SMatrix{2,2}(0, -1, 1, 0)
        return (R * (p2 - p1)) ⋅ (pt - p1) > 0 # test if inward normal faces towards the point
    end
end
